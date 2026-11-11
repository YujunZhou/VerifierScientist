"""HarnessAuditToolV08 that also tells the reward function WHICH dump is its own.

The v08 tool writes one dump per trajectory, ``<task_id>__<request_id>.json``,
but verl only hands the reward function the task id, so ``reward.py`` paired a
rollout with the oldest unclaimed dump of its task. With GRPO's n rollouts of a
task running concurrently that pairing is wrong whenever another rollout's last
tool call is older than this one's (and the dump is rewritten after every call,
so a still-running sibling's partial trace can be taken): per-sample credit is
scrambled inside the group (oracle run 1540778, 2026-10-15).

verl 0.8 carries ``agent_data.extra_fields`` through ``AgentLoopOutput`` into
``non_tensor_batch["tool_extra_fields"]``, which the reward manager merges into
``extra_info``. This subclass stores the request id there on every call, and
``reward_v2.py`` loads exactly ``<task_id>__<request_id>.json``. A rollout that
never calls a tool has no request id and no dump (scored as an empty trace).
"""

from __future__ import annotations

from typing import Any

from .tool_v08 import HarnessAuditToolV08

REQUEST_ID_KEY = "ha_request_id"


class HarnessAuditToolV09(HarnessAuditToolV08):
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        agent_data: Any = None,
        **kwargs: Any,
    ):
        request_id = getattr(agent_data, "request_id", None)
        extra = getattr(agent_data, "extra_fields", None)
        if request_id and isinstance(extra, dict):
            extra[REQUEST_ID_KEY] = str(request_id)
        return await super().execute(instance_id, parameters, agent_data=agent_data, **kwargs)
