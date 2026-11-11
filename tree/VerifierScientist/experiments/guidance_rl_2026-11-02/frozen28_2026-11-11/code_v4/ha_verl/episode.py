"""One HarnessAudit task as a step-able environment.

This is the layer every other piece of the integration sits on. It owns the
per-episode state a task needs (bank, workspace, action sink, trace writer) and
exposes it as ``reset -> call_tool* -> finish``, which is the shape verl's tool
API wants.

Nothing here re-implements scoring: violations come from ``multi_agent.checker``
and completion from ``multi_agent.completion_judge``, exactly as
``multi_agent/runner.py`` and ``multi_agent/cli.py`` drive them. The report JSON
is written in the same shape ``cli.build_report_payload`` produces, so
``splits/aggregate_results.py`` reads these runs with no changes.

Roles matter: every action is attributed to the agent that took it, because
that attribution is what the access rules are checked against. The hub calls
tools as ``hub_role``; a spoke's tool calls must be reported with that spoke's
role or the safety metrics are meaningless.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import ensure_importable, fixtures_root, workspaces_root
from .tool_schema import (
    DELEGATE_TOOL_NAME,
    delegate_tool_schema,
    domain_tool_schema,
    spoke_alias_tool_schema,
)

ensure_importable()

from multi_agent.banks import get_bank  # noqa: E402
from multi_agent.checker import check_trace, compute_metrics  # noqa: E402
from multi_agent.completion_judge import (  # noqa: E402
    CompletionEvaluation,
    evaluate_completion_checkpoints,
)
from multi_agent.frameworks.core.action_sink import ActionSink  # noqa: E402
from multi_agent.frameworks.core.sde_policy import SDE_DOMAIN  # noqa: E402
from multi_agent.frameworks.core.task_input import build_task_input, extract_input_text  # noqa: E402
from multi_agent.frameworks.core.tool_dispatch import dispatch_tool  # noqa: E402
from multi_agent.frameworks.oai.prompts import (  # noqa: E402
    build_oai_hub_prompt,
    build_oai_spoke_prompt,
)
from multi_agent.loader import load_task_with_tools, resolve_task_path  # noqa: E402
from multi_agent.trace.events import AgentInfo, TraceStart  # noqa: E402
from multi_agent.trace.writer import TraceWriter  # noqa: E402
from multi_agent.workspace import WorkspaceManager  # noqa: E402

# Matches multi_agent/runner.py: framework "oai" -> slug "oa", harness "oai".
FRAMEWORK = "oai"
FRAMEWORK_SLUG = "oa"
HARNESS = "oai"

# Baseline settings from splits/baseline_results.md; changing either of these
# breaks comparability with the published GPT/Claude rows.
DEFAULT_MAX_SPOKE_TURNS = 10
DEFAULT_MAX_DELEGATION_DEPTH = 2


def model_dir_name(model: str) -> str:
    """Filesystem-safe model directory name (mirrors cli._model_dir_name)."""
    name = (model or "unknown-model").strip() or "unknown-model"
    return name.replace("/", "__").replace(os.sep, "__")


def next_run_id(model: str) -> str:
    """``oa-oai-<model>-<uuid8>``, matching runner.next_run_id for the oai path."""
    short = uuid.uuid4().hex[:8]
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"
    return f"{FRAMEWORK_SLUG}-{HARNESS}-{safe_model}-{short}"


@dataclass
class EpisodeResult:
    """Scored outcome of one episode."""

    task_id: str
    run_id: str
    domain: str
    metrics: dict[str, Any] = field(default_factory=dict)
    n_actions: int = 0
    n_tool_calls: int = 0
    n_communications: int = 0
    n_violations: int = 0
    violations_by_code: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    report_path: str | None = None
    trace_path: str | None = None

    @property
    def sar_avg(self) -> float:
        return float(self.metrics.get("sar_avg") or 0.0)

    @property
    def tcr(self) -> float:
        return float(self.metrics.get("tcr") or 0.0)

    @property
    def tcr_det(self) -> float:
        """Deterministic-only TCR — the judge-free completion signal."""
        score = self.metrics.get("deterministic_completion_score")
        weight = self.metrics.get("deterministic_completion_weight") or 0.0
        if score is None or weight <= 0:
            return 0.0
        return float(score)


class HarnessAuditEpisode:
    """A single task instance: hub prompt + tool surface + scoring.

    Construction provisions the task's bank and workspace, so an episode owns
    real resources and must be closed. ``finish`` is the only method that
    scores; calling it twice re-scores the same action list.
    """

    def __init__(
        self,
        task: str | Path,
        *,
        model: str,
        run_id: str | None = None,
        trace_dir: str | Path | None = None,
        results_dir: str | Path | None = None,
        max_spoke_turns: int = DEFAULT_MAX_SPOKE_TURNS,
        max_delegation_depth: int = DEFAULT_MAX_DELEGATION_DEPTH,
        write_trace: bool = True,
    ) -> None:
        from .paths import package_root

        self.task_path = resolve_task_path(
            str(task), project_root=package_root(), cwd=Path.cwd()
        )
        self.task, self.catalog = load_task_with_tools(self.task_path)
        self.model = model
        self.run_id = run_id or next_run_id(model)
        self.max_spoke_turns = max_spoke_turns
        self.max_delegation_depth = max_delegation_depth
        self.started_at = datetime.now(timezone.utc)

        self._tool_defs = {t.name: t for t in self.catalog.tools}
        self._resource_tools = {t.name for t in self.catalog.tools if t.is_resource}
        self._closed = False
        self._finished = False

        model_dir = model_dir_name(model)
        self.trace_path: Path | None = None
        writer: TraceWriter | None = None
        if write_trace and trace_dir is not None:
            scoped = Path(trace_dir) / HARNESS / model_dir
            scoped.mkdir(parents=True, exist_ok=True)
            self.trace_path = scoped / f"{self.task.task_id}_{self.run_id}.jsonl"
            writer = TraceWriter(self.trace_path)
        self._writer = writer

        self.results_dir: Path | None = None
        if results_dir is not None:
            self.results_dir = Path(results_dir) / HARNESS / model_dir
            self.results_dir.mkdir(parents=True, exist_ok=True)

        self.action_sink = ActionSink(run_id=self.run_id, trace_writer=writer)

        # Bank and workspace are per-episode; concurrent episodes must not share.
        self.bank = get_bank(self.task.domain)
        if self.bank is not None and hasattr(self.bank, "apply_task_metadata"):
            self.bank.apply_task_metadata(self.task.metadata)

        self._workspace_manager: WorkspaceManager | None = None
        self.workspace = None
        if self.task.fixture:
            self._workspace_manager = WorkspaceManager(
                fixtures_root=fixtures_root(),
                workspaces_root=workspaces_root(),
            )
            self.workspace = self._workspace_manager.create(self.task.fixture, self.run_id)

        if writer is not None:
            writer.write_event(
                TraceStart(
                    trace_id=self.run_id,
                    task_id=self.task.task_id,
                    domain=self.task.domain,
                    framework=FRAMEWORK,
                    model=self.model,
                    agents=[
                        AgentInfo(
                            agent_id=a.role,
                            role=a.role,
                            useful_tools=a.useful_tools,
                        )
                        for a in self.task.agents
                    ],
                    n_rules=len(self.task.access_rules),
                    perturbation=None,
                )
            )

    # ---- Task surface ----

    @property
    def task_id(self) -> str:
        return self.task.task_id

    @property
    def domain(self) -> str:
        return self.task.domain

    @property
    def hub_role(self) -> str:
        return self.task.agents[0].role

    @property
    def spoke_configs(self) -> list[Any]:
        return list(self.task.agents[1:])

    @property
    def spoke_roles(self) -> list[str]:
        return [a.role for a in self.task.agents[1:]]

    def hub_system_prompt(self) -> str:
        return build_oai_hub_prompt(self.task.agents[0], self.spoke_configs, self.task.goal)

    def spoke_system_prompt(self, role: str) -> str:
        for cfg in self.spoke_configs:
            if cfg.role == role:
                return build_oai_spoke_prompt(cfg, self.hub_role)
        raise KeyError(f"Unknown spoke role {role!r}; have {self.spoke_roles}")

    def user_message(self) -> str:
        """Initial user turn. Multimodal assets are flattened to their text parts."""
        return extract_input_text(build_task_input(self.task)) or self.task.goal

    def domain_tool_schemas(self) -> list[dict[str, Any]]:
        return [domain_tool_schema(t) for t in self.catalog.tools]

    def hub_tool_schemas(self) -> list[dict[str, Any]]:
        """Hub sees domain tools + delegate + per-spoke aliases (baseline surface)."""
        return [
            *self.domain_tool_schemas(),
            delegate_tool_schema(),
            *[spoke_alias_tool_schema(cfg) for cfg in self.spoke_configs],
        ]

    def spoke_tool_schemas(self) -> list[dict[str, Any]]:
        """Spokes see all domain tools and no delegation tool."""
        return self.domain_tool_schemas()

    # ---- Stepping ----

    def call_tool(
        self,
        role: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        capture_mode: str = "inline_dispatch",
    ) -> str:
        """Execute a domain tool as ``role`` and record it in the trace.

        The role is the attribution used by the access checker, so passing the
        wrong one silently corrupts SAR. Unknown tools return an error string
        rather than raising: the baseline harness also surfaces those to the
        model as tool output instead of aborting the run.
        """
        tool_def = self._tool_defs.get(tool_name)
        if tool_def is None:
            result = (
                f"[ERROR] Unknown tool {tool_name!r}. "
                f"Available: {sorted(self._tool_defs)}"
            )
        else:
            try:
                result = str(
                    dispatch_tool(
                        tool_name,
                        args,
                        bank=self.bank,
                        tool_def=tool_def,
                        workspace=self.workspace,
                    )
                )
            except Exception as exc:  # tool failures are observations, not crashes
                result = f"[ERROR] Tool {tool_name} failed: {exc}"

        self.action_sink.emit_tool_call(
            agent_id=role,
            agent_role=role,
            tool_name=tool_name,
            tool_args=args,
            tool_result=result,
            raw_event={"framework": FRAMEWORK, "capture_mode": capture_mode},
        )
        return result

    def emit_delegation(self, sender: str, target: str, task_msg: str) -> None:
        """Record a hub -> spoke handoff (the surface sar_flow is computed over)."""
        self.action_sink.emit_communication(
            agent_id=sender,
            agent_role=sender,
            target_agent=target,
            target_role=target,
            content=task_msg,
            raw_event={"framework": FRAMEWORK, "capture_mode": "delegation"},
        )

    def emit_spoke_report(self, spoke: str, hub: str, output: str) -> None:
        """Record a spoke -> hub report."""
        self.action_sink.emit_communication(
            agent_id=spoke,
            agent_role=spoke,
            target_agent=hub,
            target_role=hub,
            content=output,
            raw_event={"framework": FRAMEWORK, "capture_mode": "spoke_report"},
        )

    def emit_final_output(self, content: str) -> None:
        """Record the hub's user-facing answer."""
        self.action_sink.emit_communication(
            agent_id=self.hub_role,
            agent_role=self.hub_role,
            target_agent="user",
            target_role="user",
            content=content,
            raw_event={"framework": FRAMEWORK, "communication_kind": "user_output"},
        )

    # ---- Scoring ----

    async def finish(
        self,
        final_output: str,
        *,
        skip_judge: bool = True,
        judge_model: str = "gpt-5.4",
        judge_workers: int = 4,
        error: str | None = None,
        emit_final: bool = True,
    ) -> EpisodeResult:
        """Score the episode and write trace/report artifacts.

        ``skip_judge=True`` is the baseline setting: SAR is unaffected (it is
        fully deterministic) and completion is reported through the
        deterministic checkpoints, which is the ``TCR_det`` column.
        """
        if emit_final and final_output:
            self.emit_final_output(final_output)

        actions = self.action_sink.actions  # finalizes chronological ordering

        decisions = check_trace(actions, self.task.access_rules, domain=self.task.domain)

        completion: CompletionEvaluation = await evaluate_completion_checkpoints(
            self.task,
            actions,
            final_output,
            bank=self.bank,
            workspace=self.workspace,
            judge_model=judge_model,
            skip_judge=skip_judge,
            judge_workers=judge_workers,
        )

        metrics = compute_metrics(
            self.run_id,
            actions,
            decisions,
            resource_tools=self._resource_tools,
            task_completion_score=completion.task_completion_score,
            operational_safety_score=None,
        )

        violations = [d for d in decisions if d.violated]
        n_tool_calls = sum(1 for a in actions if str(a.surface) == "tool_call")
        payload: dict[str, Any] = {
            **metrics.paper_fields(),
            **metrics.diagnostic_fields(),
            "deterministic_completion_score": completion.deterministic_completion_score,
            "deterministic_completion_weight": completion.deterministic_completion_weight,
            "deterministic_completion_contribution": (
                completion.deterministic_completion_contribution
            ),
            "llm_completion_score": completion.llm_completion_score,
            "llm_completion_weight": completion.llm_completion_weight,
            "llm_completion_contribution": completion.llm_completion_contribution,
        }

        result = EpisodeResult(
            task_id=self.task.task_id,
            run_id=self.run_id,
            domain=self.task.domain,
            metrics=payload,
            n_actions=len(actions),
            n_tool_calls=n_tool_calls,
            n_communications=len(actions) - n_tool_calls,
            n_violations=len(violations),
            violations_by_code=dict(metrics.violations_by_code),
            error=error,
            trace_path=str(self.trace_path) if self.trace_path else None,
        )
        self._finished = True
        self._write_report(result, metrics, completion)
        return result

    def _write_report(
        self,
        result: EpisodeResult,
        metrics: Any,
        completion: CompletionEvaluation,
    ) -> None:
        """Write the per-run report in cli.build_report_payload's shape."""
        if self.results_dir is None:
            return
        duration = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        report = {
            "task_id": result.task_id,
            "run_id": result.run_id,
            "framework": FRAMEWORK,
            "llm": self.model,
            "n_actions": result.n_actions,
            "n_tool_calls": result.n_tool_calls,
            "n_communications": result.n_communications,
            "n_violations": result.n_violations,
            **metrics.paper_fields(),
            **metrics.diagnostic_fields(),
            "deterministic_completion_score": completion.deterministic_completion_score,
            "deterministic_completion_weight": completion.deterministic_completion_weight,
            "deterministic_completion_contribution": (
                completion.deterministic_completion_contribution
            ),
            "llm_completion_score": completion.llm_completion_score,
            "llm_completion_weight": completion.llm_completion_weight,
            "llm_completion_contribution": completion.llm_completion_contribution,
            "trajectory_tcr": metrics.tcr,
            "workspace_path": str(self.workspace.path) if self.workspace else None,
            "violations_by_code": dict(metrics.violations_by_code),
            "violations_by_severity": dict(metrics.violations_by_severity),
            "trace_file": result.trace_path,
            "error": result.error,
            "warnings": [],
            "wall_time_s": duration,
            "runner": "ha_verl",
        }
        out_path = self.results_dir / f"{result.task_id}_{result.run_id}.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        result.report_path = str(out_path)

    # ---- Teardown ----

    def close(self) -> None:
        """Release bank, workspace and trace writer. Safe to call twice."""
        if self._closed:
            return
        self._closed = True

        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass

        bank, self.bank = self.bank, None
        if bank is not None:
            close_fn = getattr(bank, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

        if self.workspace is not None and self._workspace_manager is not None:
            # Mirrors runner.py: SDE workspaces and unfinished runs are kept for
            # inspection, everything else is reclaimed.
            keep = (
                self.task.domain == SDE_DOMAIN
                or not self._finished
                or os.getenv("MASP_KEEP_WORKSPACES") == "1"
            )
            try:
                self._workspace_manager.destroy(self.workspace, keep=keep)
            except Exception:
                pass
            self.workspace = None

    def __enter__(self) -> HarnessAuditEpisode:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
