"""verl >= 0.8 tool binding for HarnessAudit (agent-loop lifecycle).

verl 0.8 replaced the in-engine sglang multi-turn rollout with the agent loop
(`verl/experimental/agent_loop/tool_agent_loop.py`), which changes the BaseTool
contract in three ways this subclass absorbs:

1. ``create`` is called once per *tool call* (not once per rollout), receives
   the sample's kwargs as a single ``create_kwargs`` dict, and must return
   ``(instance_id, ToolResponse)``.
2. ``execute`` must return ``(ToolResponse, reward, metrics)`` and receives
   ``agent_data``, whose ``request_id`` is the only stable per-trajectory key.
3. ``release`` runs after every call, so it can no longer mean "the episode is
   over". There is no end-of-trajectory hook at all.

The episode therefore lives in the registry keyed by ``agent_data.request_id``,
and the dump is refreshed after every tool call instead of once at the end:
with ``--skip-judge`` the completion checkpoints are deterministic functions of
bank/workspace state, and that state only changes through tool calls, so the
snapshot taken after the last call equals what end-of-episode scoring would
have produced. The reward function claims the final dump exactly as before.

Loaded by dotted path via ``rollout.multi_turn.tool_config_path``; nothing in
the verl install or HarnessAudit is modified. Run with
``VERL_ROOT=<env>/lib/python3.11/site-packages`` so ``paths.ensure_verl_importable``
resolves the *installed* verl 0.8 rather than the old 0.4 checkout.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from .paths import ensure_verl_importable

ensure_verl_importable()

from verl.tools.schemas import ToolResponse  # noqa: E402

from .persist import write_dump  # noqa: E402
from .registry import REGISTRY
from .tool import HarnessAuditTool

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class HarnessAuditToolV08(HarnessAuditTool):
    """HarnessAudit tool for verl 0.8's per-call create/execute/release cycle."""

    # instance_id -> create_kwargs, alive between create() and release().
    # Class-level on purpose: verl instantiates one object per tool name, and
    # create/release of one call always hit the same instance, but a shared map
    # costs nothing and survives any future instancing change.
    _pending: dict[str, dict[str, Any]] = {}

    async def create(
        self,
        instance_id: Optional[str] = None,
        create_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> tuple[str, ToolResponse]:
        instance_id = instance_id or str(uuid4())
        self._pending[instance_id] = dict(create_kwargs or {})
        return instance_id, ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        agent_data: Any = None,
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict]:
        create_kwargs = self._pending.get(instance_id, {})
        request_id = getattr(agent_data, "request_id", None) or instance_id

        episode = REGISTRY.get(request_id)
        if episode is None:
            task_yaml = create_kwargs.get("task_yaml")
            if not task_yaml:
                msg = (
                    f"[ERROR] tool {self.name!r} has no create_kwargs.task_yaml; "
                    "regenerate the dataset with scripts/build_dataset.py"
                )
                return ToolResponse(text=msg), 0.0, {"ha_error": 1}
            # The parquet's create_kwargs bake in the model name the dataset
            # was built for; the env override keeps one dataset usable across
            # models without mislabeling dumps/reports (the Qwen3.5 run would
            # otherwise file its results as Qwen3-4B).
            episode = await REGISTRY.acquire(
                request_id,
                task=task_yaml,
                model=os.getenv("HA_MODEL_NAME")
                or create_kwargs.get("model")
                or "unknown-model",
                trace_dir=self.trace_dir,
                results_dir=None,  # the driver writes the report, after rescoring
                max_spoke_turns=self.max_spoke_turns,
                max_delegation_depth=self.max_delegation_depth,
            )

        if not isinstance(parameters, dict):
            parameters = {}

        lock = REGISTRY.lock_for(request_id)
        try:
            if lock is not None:
                async with lock:
                    response = await self._dispatch(episode, parameters)
                    await self._snapshot(episode, request_id)
            else:
                response = await self._dispatch(episode, parameters)
                await self._snapshot(episode, request_id)
        except Exception as exc:  # never let a tool fault kill the rollout
            logger.warning("HarnessAudit tool %s failed: %s", self.name, exc)
            return ToolResponse(text=f"[ERROR] Tool {self.name} failed: {exc}"), 0.0, {"ha_error": 1}

        return ToolResponse(text=response), 0.0, {"ha_tool": self.name}

    async def _snapshot(self, episode: Any, request_id: str) -> None:
        """Refresh this trajectory's dump so the latest write is the final state."""
        error = None
        try:
            from multi_agent.completion_judge import evaluate_completion_checkpoints

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
            write_dump(episode, request_id=request_id, completion=completion, error=error)
        except Exception as exc:
            logger.error("failed to persist episode %s: %s", episode.task_id, exc)

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        """Per-call cleanup only; the episode outlives every individual call."""
        self._pending.pop(instance_id, None)
