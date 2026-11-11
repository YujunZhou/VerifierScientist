"""Run a spoke (specialist) agent inside the hub's delegation tool.

The policy under training plays the hub only. When it delegates, the named
specialist has to actually run: read its own system prompt, call domain tools,
and report back. That loop happens here, against an OpenAI-compatible endpoint
(an SGLang server holding the same weights).

This mirrors ``multi_agent/frameworks/oai/tools.py::_delegate`` -- same
delegation/report trace events, same 10-turn cap, same error strings for an
unknown target -- but drives the loop with the plain ``openai`` client instead
of the Agents SDK, so nothing has to be installed into the shared verl env.

Configure with:
  HA_SPOKE_BASE_URL   OpenAI-compatible endpoint (e.g. http://host:30000/v1)
  HA_SPOKE_MODEL      model name to request
  HA_SPOKE_API_KEY    optional; SGLang ignores it but the client requires one
"""

from __future__ import annotations

import json
import os
from typing import Any

DEFAULT_TIMEOUT_S = 180.0
UNKNOWN_AGENT_TEMPLATE = "[ERROR] Unknown agent '{target}'. Available: {available}"
MAX_DEPTH_MESSAGE = "[ERROR] Max delegation depth reached."


class SpokeConfig:
    """Endpoint settings for spoke execution, read from the environment."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 2048,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.base_url = base_url or os.environ.get("HA_SPOKE_BASE_URL", "")
        self.model = model or os.environ.get("HA_SPOKE_MODEL", "")
        self.api_key = api_key or os.environ.get("HA_SPOKE_API_KEY", "EMPTY")
        env_temp = os.environ.get("HA_SPOKE_TEMPERATURE")
        self.temperature = temperature if temperature is not None else (
            float(env_temp) if env_temp else 1.0
        )
        self.max_tokens = int(os.environ.get("HA_SPOKE_MAX_TOKENS", max_tokens))
        self.timeout_s = float(os.environ.get("HA_SPOKE_TIMEOUT_S", timeout_s))

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)

    def describe(self) -> str:
        return f"{self.model} @ {self.base_url or '<unset>'}"


def _client(cfg: SpokeConfig):
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key or "EMPTY",
        timeout=cfg.timeout_s,
        max_retries=2,
    )


async def run_spoke(
    episode: Any,
    *,
    target_role: str,
    task_msg: str,
    sender_role: str,
    cfg: SpokeConfig,
    max_turns: int = 10,
    depth: int = 0,
    max_depth: int = 2,
) -> str:
    """Execute one delegation and return the specialist's report.

    Trace side effects are the point: the delegation, every tool call the spoke
    makes (attributed to the spoke's role), and the report back all land in the
    episode's action sink, because that is what the access rules are checked
    against.
    """
    if depth >= max_depth:
        return MAX_DEPTH_MESSAGE

    if target_role not in episode.spoke_roles:
        # Same message the baseline's delegate_to_agent returns.
        return UNKNOWN_AGENT_TEMPLATE.format(
            target=target_role, available=list(episode.spoke_roles)
        )

    episode.emit_delegation(sender_role, target_role, task_msg)

    if not cfg.configured:
        output = (
            "[ERROR] No spoke endpoint configured; set HA_SPOKE_BASE_URL and "
            "HA_SPOKE_MODEL."
        )
        episode.emit_spoke_report(target_role, sender_role, output)
        return output

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": episode.spoke_system_prompt(target_role)},
        {"role": "user", "content": task_msg},
    ]
    tools = episode.spoke_tool_schemas()
    client = _client(cfg)
    output = ""

    try:
        for _ in range(max_turns):
            resp = await client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                tools=tools,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            choice = resp.choices[0]
            msg = choice.message
            tool_calls = getattr(msg, "tool_calls", None) or []

            if not tool_calls:
                output = (msg.content or "").strip()
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except Exception:
                    args = {}
                result = episode.call_tool(
                    target_role,
                    tc.function.name,
                    args,
                    capture_mode="spoke_dispatch",
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
        else:
            # Turn budget exhausted with the spoke still calling tools.
            output = output or "[ERROR] Spoke reached its turn limit without a summary."
    except Exception as exc:
        output = f"[ERROR] {exc}"

    episode.emit_spoke_report(target_role, sender_role, output)
    return output
