"""OpenAI function schemas for HarnessAudit domain tools.

These mirror ``multi_agent/frameworks/oai/tools.py`` so a verl rollout sees the
exact same tool surface the GPT/Claude baselines saw. That file cannot simply be
imported here because it pulls in the ``agents`` SDK at module scope, so the
schema builders are duplicated -- ``tests/test_schema_parity.py`` diffs the two
implementations whenever ``openai-agents`` is installed, which is what keeps
this copy honest.
"""

from __future__ import annotations

import re
from typing import Any

# Source: multi_agent/frameworks/oai/tools.py::_TYPE_PREFIX_RE
_TYPE_PREFIX_RE = re.compile(r"^\((int|float|str|string|bool)\)\s*")


def json_type_from_desc(desc: str) -> str:
    """Extract JSON Schema type from a ToolDef param description like '(int) Client ID'."""
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


def build_params_schema(tool_def: Any) -> dict[str, Any]:
    """Build a strict JSON Schema from ToolDef.params."""
    properties: dict[str, Any] = {}
    for name, desc in tool_def.params.items():
        properties[name] = {"type": json_type_from_desc(desc), "description": desc}
    return {
        "type": "object",
        "properties": properties,
        "required": list(tool_def.params.keys()),
        "additionalProperties": False,
    }


def domain_tool_schema(tool_def: Any) -> dict[str, Any]:
    """One OpenAI-format function schema for a domain tool."""
    return {
        "type": "function",
        "function": {
            "name": tool_def.name,
            "description": tool_def.description or tool_def.name,
            "parameters": build_params_schema(tool_def),
        },
    }


DELEGATE_TOOL_NAME = "delegate_to_agent"

DELEGATE_TOOL_DESCRIPTION = (
    "Delegate a sub-task to a specialist agent by name. "
    "The agent will execute the task using domain tools and return findings. "
    "Multiple calls in one turn execute concurrently."
)

_DELEGATE_PARAMS: dict[str, Any] = {
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
}

_SPOKE_ALIAS_PARAMS: dict[str, Any] = {
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
}


def delegate_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": DELEGATE_TOOL_NAME,
            "description": DELEGATE_TOOL_DESCRIPTION,
            "parameters": dict(_DELEGATE_PARAMS),
        },
    }


def spoke_alias_tool_schema(agent_cfg: Any) -> dict[str, Any]:
    """Per-spoke delegation alias; tool name == role name.

    Mirrors ``build_spoke_alias_tools``: models sometimes call a specialist's
    role name directly instead of ``delegate_to_agent``, and the baseline treats
    those as equivalent delegations.
    """
    role = agent_cfg.role
    return {
        "type": "function",
        "function": {
            "name": role,
            "description": (
                f"Delegate a sub-task to the {role} specialist. "
                f"{agent_cfg.description or ''} Equivalent to "
                f"delegate_to_agent(agent_name='{role}', task=...)."
            ),
            "parameters": dict(_SPOKE_ALIAS_PARAMS),
        },
    }
