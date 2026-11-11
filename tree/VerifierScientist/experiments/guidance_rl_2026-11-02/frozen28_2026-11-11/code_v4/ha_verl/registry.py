"""Per-rollout episode registry.

verl calls ``create``/``release`` once per *tool* per rollout, but a HarnessAudit
task is a single episode shared by all of its tools -- one bank, one workspace,
one action trace. This registry keys episodes by verl's ``request_id`` and
refcounts them, so the first ``create`` builds the episode and the last
``release`` tears it down.

Everything here is process-local. That is correct for verl's sglang rollout,
where a request's tool calls all run in one worker process and one event loop.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .episode import HarnessAuditEpisode


@dataclass
class _Entry:
    episode: HarnessAuditEpisode
    refs: int = 0
    final_completion: Any = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class EpisodeRegistry:
    """Refcounted map of request_id -> live episode."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._guard = asyncio.Lock()

    async def acquire(self, request_id: str, **episode_kwargs: Any) -> HarnessAuditEpisode:
        """Return the episode for this request, constructing it on first use."""
        async with self._guard:
            entry = self._entries.get(request_id)
            if entry is None:
                episode = HarnessAuditEpisode(**episode_kwargs)
                entry = _Entry(episode=episode)
                self._entries[request_id] = entry
            entry.refs += 1
            return entry.episode

    def get(self, request_id: str) -> HarnessAuditEpisode | None:
        entry = self._entries.get(request_id)
        return entry.episode if entry else None

    def lock_for(self, request_id: str) -> asyncio.Lock | None:
        """Serialize tool calls within one episode (bank/workspace are shared)."""
        entry = self._entries.get(request_id)
        return entry.lock if entry else None

    async def release(self, request_id: str, on_last: Any = None) -> None:
        """Drop one reference; run ``on_last(episode)`` and close at zero.

        ``on_last`` is where end-of-episode work belongs -- scoring completion
        while the bank is still alive, and persisting the trace for the driver.
        """
        async with self._guard:
            entry = self._entries.get(request_id)
            if entry is None:
                return
            entry.refs -= 1
            if entry.refs > 0:
                return
            self._entries.pop(request_id, None)

        try:
            if on_last is not None:
                result = on_last(entry.episode)
                if asyncio.iscoroutine(result):
                    await result
        finally:
            entry.episode.close()

    def __len__(self) -> int:
        return len(self._entries)


REGISTRY = EpisodeRegistry()
