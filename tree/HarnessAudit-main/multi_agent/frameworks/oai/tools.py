"""Tool factories for the OpenAI Agents SDK adapter.

Creates ``FunctionTool`` wrappers that bridge multi-agent ToolCatalog definitions
to the OpenAI Agents SDK, dispatching through Bank / fixture / mock and
emitting ActionSink events.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from agents import FunctionTool, Runner

try:
    from agents import RunContextWrapper as ToolContext
except ImportError:  # older SDKs exposed the function-tool context here
    from agents.tool_context import ToolContext

if TYPE_CHECKING:
    from multi_agent.schemas.task import ToolCatalog, ToolDef


# ---- Parameter schema helpers ----

_TYPE_PREFIX_RE = re.compile(r"^\((int|float|str|string|bool)\)\s*")


def _json_type_from_desc(desc: str) -> str:
    """Extract JSON Schema type from ToolDef param description like '(int) Client ID'."""
    m = _TYPE_PREFIX_RE.match(desc)
    if not m:
        return "string"
    t = m.group(1)
    if t == "int":
        return "integer"
    if t == "float":
        return "number"
    if t == "bool":
        return "boolean"
    return "string"


def _build_params_schema(td: ToolDef) -> dict[str, Any]:
    """Build a strict JSON Schema from ToolDef.params."""
    properties: dict[str, Any] = {}
    for name, desc in td.params.items():
        properties[name] = {"type": _json_type_from_desc(desc), "description": desc}
    return {
        "type": "object",
        "properties": properties,
        "required": list(td.params.keys()),
        "additionalProperties": False,  # required for strict_json_schema=True
    }


# ---- Domain tool factory ----

def build_domain_tools(
    catalog: ToolCatalog,
    workspace: Any = None,
) -> list[FunctionTool]:
    """Create one ``FunctionTool`` per ToolDef in the catalog.

    Each tool dispatches through the Bank→real→fixture→mock cascade and emits
    an ``ActionSink.emit_tool_call`` via the shared ``MultiAgentContext``.
    """
    tools: list[FunctionTool] = []
    for td in catalog.tools:
        tools.append(FunctionTool(
            name=td.name,
            description=td.description or td.name,
            params_json_schema=_build_params_schema(td),
            on_invoke_tool=_make_domain_handler(td, workspace),
            strict_json_schema=True,
        ))
    return tools


def _make_domain_handler(td: ToolDef, workspace: Any):
    """Return an async handler closure for a single domain tool."""

    async def handler(ctx: ToolContext[Any], args_json: str) -> str:
        run_ctx = ctx.context  # MultiAgentContext
        agent_role = run_ctx.agent_role
        try:
            args = json.loads(args_json)
        except Exception as exc:
            args = {}
            result = f"[ERROR] Invalid JSON arguments for {td.name}: {exc}"
            run_ctx.action_sink.emit_tool_call(
                agent_id=agent_role,
                agent_role=agent_role,
                tool_name=td.name,
                tool_args=args,
                tool_result=result,
                raw_event={"framework": "oai", "capture_mode": "inline_dispatch"},
            )
            return result

        from ..core.tool_dispatch import dispatch_tool

        try:
            result = dispatch_tool(
                td.name, args, bank=run_ctx.bank, tool_def=td, workspace=workspace,
            )
        except Exception as exc:
            result = f"[ERROR] Tool {td.name} failed: {exc}"
        result = str(result)

        run_ctx.action_sink.emit_tool_call(
            agent_id=agent_role,
            agent_role=agent_role,
            tool_name=td.name,
            tool_args=args,
            tool_result=result,
            raw_event={"framework": "oai", "capture_mode": "inline_dispatch"},
        )
        return result

    return handler


# ---- Delegate-to-agent tool (hub → spoke, enables parallel execution) ----

async def _delegate(run_ctx: Any, target: str, task_msg: str, max_spoke_turns: int) -> str:
    """Shared hub→spoke delegation flow (used by delegate_to_agent and role aliases)."""
    sender = run_ctx.agent_role
    sink = run_ctx.action_sink

    # ---- recursion guard ----
    if run_ctx.depth >= 2:
        return "[ERROR] Max delegation depth reached."

    spoke = run_ctx.spoke_agents.get(target)
    if spoke is None:
        return f"[ERROR] Unknown agent '{target}'. Available: {list(run_ctx.spoke_agents.keys())}"

    # ---- emit hub → spoke communication ----
    sink.emit_communication(
        agent_id=sender,
        agent_role=sender,
        target_agent=target,
        target_role=target,
        content=task_msg,
        raw_event={"framework": "oai", "capture_mode": "delegation"},
    )

    # ---- run spoke agent (async → concurrent when parallel_tool_calls) ----
    from .adapter import MultiAgentContext  # deferred to avoid circular import

    spoke_ctx = MultiAgentContext(
        action_sink=sink,
        bank=run_ctx.bank,
        workspace=run_ctx.workspace,
        agent_role=target,
        spoke_agents=run_ctx.spoke_agents,
        depth=run_ctx.depth + 1,
    )
    try:
        runner_kwargs: dict[str, Any] = {
            "context": spoke_ctx,
            "max_turns": max_spoke_turns,
        }
        if run_ctx.run_config is not None:
            runner_kwargs["run_config"] = run_ctx.run_config
        result = await Runner.run(
            spoke,
            task_msg,
            **runner_kwargs,
        )
        output = str(result.final_output) if result.final_output else ""
    except Exception as exc:
        output = f"[ERROR] {exc}"

    # ---- emit spoke → hub communication ----
    sink.emit_communication(
        agent_id=target,
        agent_role=target,
        target_agent=sender,
        target_role=sender,
        content=output,
        raw_event={"framework": "oai", "capture_mode": "spoke_report"},
    )
    return output


def build_delegate_tool(max_spoke_turns: int = 10) -> FunctionTool:
    """Create the ``delegate_to_agent`` tool used by the hub.

    When the hub LLM proposes multiple delegate calls in a single turn
    (``parallel_tool_calls=True``), the SDK executes them concurrently.
    Each call runs ``Runner.run`` for the target spoke agent.
    """

    async def delegate_to_agent(ctx: ToolContext[Any], args_json: str) -> str:
        run_ctx = ctx.context  # MultiAgentContext
        try:
            args = json.loads(args_json)
            target = str(args["agent_name"])
            task_msg = str(args["task"])
        except Exception as exc:
            return f"[ERROR] Invalid delegation arguments: {exc}"
        return await _delegate(run_ctx, target, task_msg, max_spoke_turns)

    return FunctionTool(
        name="delegate_to_agent",
        description=(
            "Delegate a sub-task to a specialist agent by name. "
            "The agent will execute the task using domain tools and return findings. "
            "Multiple calls in one turn execute concurrently."
        ),
        params_json_schema={
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Name of the specialist agent to delegate to.",
                },
                "task": {
                    "type": "string",
                    "description": (
                        "Sub-task description with all necessary IDs, "
                        "parameters, and context for the specialist."
                    ),
                },
            },
            "required": ["agent_name", "task"],
            "additionalProperties": False,
        },
        on_invoke_tool=delegate_to_agent,
        strict_json_schema=True,
    )


def build_spoke_alias_tools(spoke_cfgs: list, max_spoke_turns: int = 10) -> list[FunctionTool]:
    """Per-spoke delegation aliases (tool name == role name).

    Some models call the specialist's name directly as a tool instead of
    ``delegate_to_agent``; the OpenAI Agents SDK treats an unknown tool name as
    a fatal ModelBehaviorError that kills the whole run. These aliases route
    such calls through the same delegation flow, so the trace is identical to a
    ``delegate_to_agent`` call.
    """

    def _make(role: str):
        async def alias(ctx: ToolContext[Any], args_json: str) -> str:
            run_ctx = ctx.context  # MultiAgentContext
            try:
                args = json.loads(args_json) if args_json else {}
            except Exception as exc:
                return f"[ERROR] Invalid delegation arguments: {exc}"
            task_msg = str(args.get("task") or args.get("message") or args.get("input") or "")
            if not task_msg:
                return "[ERROR] Missing 'task' argument for delegation."
            return await _delegate(run_ctx, role, task_msg, max_spoke_turns)

        return alias

    tools: list[FunctionTool] = []
    for cfg in spoke_cfgs:
        tools.append(FunctionTool(
            name=cfg.role,
            description=(
                f"Delegate a sub-task to the {cfg.role} specialist. "
                f"{cfg.description or ''} Equivalent to "
                f"delegate_to_agent(agent_name='{cfg.role}', task=...)."
            ),
            params_json_schema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "Sub-task description with all necessary IDs, "
                            "parameters, and context for the specialist."
                        ),
                    },
                },
                "required": ["task"],
                "additionalProperties": False,
            },
            on_invoke_tool=_make(cfg.role),
            strict_json_schema=True,
        ))
    return tools
