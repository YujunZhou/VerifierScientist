"""Hand an episode's trace from the rollout worker to the reward function.

Scoring is split across two processes because neither one has everything:

- The **worker** owns the bank and workspace, so completion checkpoints
  (``sql_assert``, ``tool_called``, ``pytest_passes``) must be evaluated there,
  at release time. It never sees the hub's final answer.
- The **driver** running the reward function has the final answer in
  ``solution_str`` but no bank. It needs the final answer because that message
  is scanned for leaked secrets, which is what ``sar_flow`` measures.

So the worker dumps actions plus its completion numbers, and the driver reloads
them, appends the final-answer communication, and computes violations and
metrics -- both of which are pure functions of actions, rules and catalog.

This split is only sound while no completion checkpoint reads the final output.
``scripts/build_dataset.py`` asserts that at build time (``final_output_contains``
is unused across the test split); if that ever changes, the build fails rather
than silently mis-scoring.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .paths import ensure_importable

ensure_importable()

from multi_agent.checker import check_trace, compute_metrics  # noqa: E402
from multi_agent.loader import load_task_with_tools  # noqa: E402
from multi_agent.schemas.actions import ActionSurface, ObservableAction  # noqa: E402

DUMP_ENV_VAR = "HA_EPISODE_DIR"
DEFAULT_DUMP_DIR = "ha_episodes"


def dump_dir() -> Path:
    path = Path(os.environ.get(DUMP_ENV_VAR, DEFAULT_DUMP_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_path_for(task_id: str, request_id: str) -> Path:
    safe = str(request_id).replace("/", "_")
    return dump_dir() / f"{task_id}__{safe}.json"


def write_dump(
    episode: Any,
    *,
    request_id: str,
    completion: Any,
    error: str | None = None,
) -> Path:
    """Persist one finished episode. Written atomically: the driver may be polling."""
    payload = {
        "task_id": episode.task_id,
        "task_path": str(episode.task_path),
        "run_id": episode.run_id,
        "request_id": request_id,
        "domain": episode.domain,
        "model": episode.model,
        "hub_role": episode.hub_role,
        "trace_path": str(episode.trace_path) if episode.trace_path else None,
        "error": error,
        "completion": {
            "task_completion_score": completion.task_completion_score,
            "deterministic_completion_score": completion.deterministic_completion_score,
            "deterministic_completion_weight": completion.deterministic_completion_weight,
            "deterministic_completion_contribution": (
                completion.deterministic_completion_contribution
            ),
            "llm_completion_score": completion.llm_completion_score,
            "llm_completion_weight": completion.llm_completion_weight,
            "llm_completion_contribution": completion.llm_completion_contribution,
        },
        "actions": [a.model_dump(mode="json") for a in episode.action_sink.actions],
    }
    target = dump_path_for(episode.task_id, request_id)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def find_dumps(task_id: str) -> list[Path]:
    """All dumps for a task, newest last."""
    return sorted(dump_dir().glob(f"{task_id}__*.json"), key=lambda p: p.stat().st_mtime)


def load_dump(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _actions_from(payload: dict[str, Any]) -> list[ObservableAction]:
    return [ObservableAction(**a) for a in payload["actions"]]


def rescore(
    payload: dict[str, Any],
    final_output: str,
) -> dict[str, Any]:
    """Recompute violations and metrics with the hub's final answer included.

    Returns a report dict in the shape ``cli.build_report_payload`` produces, so
    ``splits/aggregate_results.py`` can read it unchanged.
    """
    task, catalog = load_task_with_tools(Path(payload["task_path"]))
    actions = _actions_from(payload)

    if final_output:
        from datetime import datetime, timezone

        hub_role = payload.get("hub_role") or task.agents[0].role
        last = actions[-1] if actions else None
        actions = actions + [
            ObservableAction(
                # pydantic >= 2.13 (verl08 env) rejects None here; a zero-action
                # episode (hub answered without any successful tool call) still
                # needs the synthetic user_output action to be scoreable.
                timestamp=last.timestamp if last else datetime.now(timezone.utc),
                sequence_num=(last.sequence_num + 1) if last else 1,
                run_id=payload["run_id"],
                surface=ActionSurface.COMMUNICATION,
                agent_id=hub_role,
                agent_role=hub_role,
                target_agent="user",
                target_role="user",
                content=final_output,
                raw_event={"framework": "oai", "communication_kind": "user_output"},
            )
        ]

    decisions = check_trace(actions, task.access_rules, domain=task.domain)
    completion = payload.get("completion") or {}
    metrics = compute_metrics(
        payload["run_id"],
        actions,
        decisions,
        resource_tools={t.name for t in catalog.tools if t.is_resource},
        task_completion_score=float(completion.get("task_completion_score") or 0.0),
        operational_safety_score=None,
    )

    n_tool_calls = sum(1 for a in actions if str(a.surface) == "tool_call")
    violations = [d for d in decisions if d.violated]
    return {
        "task_id": payload["task_id"],
        "run_id": payload["run_id"],
        "framework": "oai",
        "llm": payload.get("model", ""),
        "n_actions": len(actions),
        "n_tool_calls": n_tool_calls,
        "n_communications": len(actions) - n_tool_calls,
        "n_violations": len(violations),
        **metrics.paper_fields(),
        **metrics.diagnostic_fields(),
        **completion,
        "trajectory_tcr": metrics.tcr,
        "violations_by_code": dict(metrics.violations_by_code),
        "violations_by_severity": dict(metrics.violations_by_severity),
        "trace_file": payload.get("trace_path"),
        "error": payload.get("error"),
        "warnings": [],
        "runner": "ha_verl",
    }
