"""verl tool binding for HarnessAudit.

One class backs every tool name a task exposes -- the domain tools,
``delegate_to_agent``, and the per-role delegation aliases. verl instantiates it
once per registered name (the schema comes from the tool config), and dispatch
is by ``self.name``.

Loaded by dotted path through ``verl/tools/base_tool.py::initialize_tools_from_config``,
so this file lives outside the verl checkout and nothing there is modified.

Registration constraint worth knowing: verl builds its function-call parser once
from *all* registered tools, but looks up ``_req.tools_kwargs[name]`` per sample.
A name that is registered globally but missing from a sample's ``tools_kwargs``
raises KeyError mid-rollout. The dataset builder therefore emits one job per
domain and gives every sample in it the same key set.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from .paths import ensure_verl_importable

ensure_verl_importable()

from verl.tools.base_tool import BaseTool  # noqa: E402
from verl.tools.schemas import OpenAIFunctionToolSchema  # noqa: E402

from .persist import write_dump  # noqa: E402
from .registry import REGISTRY
from .spoke import SpokeConfig, run_spoke
from .tool_schema import DELEGATE_TOOL_NAME

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Baseline settings (splits/baseline_results.md); see harnessaudit-baseline-protocol.
DEFAULT_MAX_SPOKE_TURNS = 10
DEFAULT_MAX_DELEGATION_DEPTH = 2


class HarnessAuditTool(BaseTool):
    """A single HarnessAudit tool name, backed by a shared per-rollout episode."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        cfg = config or {}
        self.trace_dir = cfg.get("trace_dir")
        self.results_dir = cfg.get("results_dir")
        self.skip_judge = bool(cfg.get("skip_judge", True))
        self.judge_model = cfg.get("judge_model", "gpt-5.4")
        self.judge_workers = int(cfg.get("judge_workers", 4))
        self.max_spoke_turns = int(cfg.get("max_spoke_turns", DEFAULT_MAX_SPOKE_TURNS))
        self.max_delegation_depth = int(
            cfg.get("max_delegation_depth", DEFAULT_MAX_DELEGATION_DEPTH)
        )
        self.spoke_cfg = SpokeConfig(
            base_url=cfg.get("spoke_base_url"),
            model=cfg.get("spoke_model"),
            api_key=cfg.get("spoke_api_key"),
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    # ---- lifecycle ----

    async def create(
        self,
        instance_id: Optional[str] = None,
        task_yaml: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Attach to (or build) this rollout's episode.

        Called once per registered tool per rollout; all of them carry the same
        ``task_yaml``, and the registry makes sure only the first one builds.
        """
        instance_id = instance_id or str(uuid4())
        if not task_yaml:
            raise ValueError(
                f"tool {self.name!r} needs create_kwargs.task_yaml; "
                "regenerate the dataset with scripts/build_dataset.py"
            )
        await REGISTRY.acquire(
            instance_id,
            task=task_yaml,
            model=model or "unknown-model",
            trace_dir=self.trace_dir,
            results_dir=None,  # the driver writes the report, after rescoring
            max_spoke_turns=self.max_spoke_turns,
            max_delegation_depth=self.max_delegation_depth,
        )
        return instance_id

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[str, float, dict]:
        """Run this tool call inside the rollout's episode.

        Returns the tool text the model sees. The step reward is always 0: the
        HarnessAudit score is trajectory-level and computed once, after the
        episode ends.
        """
        episode = REGISTRY.get(instance_id)
        if episode is None:
            return f"[ERROR] No active episode for {self.name}.", 0.0, {"ha_error": 1}

        if not isinstance(parameters, dict):
            parameters = {}

        lock = REGISTRY.lock_for(instance_id)
        try:
            if lock is not None:
                async with lock:
                    response = await self._dispatch(episode, parameters)
            else:
                response = await self._dispatch(episode, parameters)
        except Exception as exc:  # never let a tool fault kill the rollout
            logger.warning("HarnessAudit tool %s failed: %s", self.name, exc)
            return f"[ERROR] Tool {self.name} failed: {exc}", 0.0, {"ha_error": 1}

        return response, 0.0, {"ha_tool": self.name}

    async def _dispatch(self, episode: Any, parameters: dict[str, Any]) -> str:
        # delegate_to_agent(agent_name, task)
        if self.name == DELEGATE_TOOL_NAME:
            target = str(parameters.get("agent_name") or "")
            task_msg = str(parameters.get("task") or "")
            if not target or not task_msg:
                return "[ERROR] Invalid delegation arguments: need agent_name and task."
            return await self._delegate(episode, target, task_msg)

        # Role-name alias: <role>(task). Same flow as delegate_to_agent.
        if self.name in episode.spoke_roles or self._looks_like_role(episode):
            task_msg = str(
                parameters.get("task")
                or parameters.get("message")
                or parameters.get("input")
                or ""
            )
            if not task_msg:
                return "[ERROR] Missing 'task' argument for delegation."
            return await self._delegate(episode, self.name, task_msg)

        # Domain tool, called by the hub.
        return episode.call_tool(episode.hub_role, self.name, parameters)

    def _looks_like_role(self, episode: Any) -> bool:
        """True when this tool is a role alias that this task does not have.

        Per-domain registration means a task can be offered an alias for a role
        belonging to a sibling task. Routing it into delegation produces the
        baseline's "Unknown agent" message instead of dispatching a nonexistent
        domain tool.
        """
        return self.name not in {t.name for t in episode.catalog.tools}

    async def _delegate(self, episode: Any, target: str, task_msg: str) -> str:
        return await run_spoke(
            episode,
            target_role=target,
            task_msg=task_msg,
            sender_role=episode.hub_role,
            cfg=self.spoke_cfg,
            max_turns=self.max_spoke_turns,
            max_depth=self.max_delegation_depth,
        )

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        """Always 0 -- the trajectory score is computed by the reward function.

        The bank-dependent half of scoring happens in ``release``; the half that
        needs the final answer happens in the driver. Neither can produce a
        number here.
        """
        return 0.0

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        """Drop this tool's reference; the last one scores and persists."""
        await REGISTRY.release(instance_id, on_last=self._finalize_factory(instance_id))

    def _finalize_factory(self, instance_id: str):
        async def finalize(episode: Any) -> None:
            error = None
            try:
                from multi_agent.completion_judge import evaluate_completion_checkpoints

                # Evaluated here because the bank and workspace are still alive.
                # final_output="" is sound only while no checkpoint reads it --
                # asserted at dataset build time.
                completion = await evaluate_completion_checkpoints(
                    episode.task,
                    episode.action_sink.actions,
                    "",
                    bank=episode.bank,
                    workspace=episode.workspace,
                    judge_model=self.judge_model,
                    skip_judge=self.skip_judge,
                    judge_workers=self.judge_workers,
                )
            except Exception as exc:
                logger.warning("completion evaluation failed: %s", exc)
                from multi_agent.completion_judge import CompletionEvaluation

                completion = CompletionEvaluation(task_completion_score=0.0)
                error = f"completion_evaluation_failed: {exc}"

            try:
                write_dump(
                    episode,
                    request_id=instance_id,
                    completion=completion,
                    error=error,
                )
            except Exception as exc:
                logger.error("failed to persist episode %s: %s", episode.task_id, exc)

        return finalize
