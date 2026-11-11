"""HarnessAudit <-> verl integration.

New code only: HarnessAudit and verl_sglang are both used as-is, with nothing
installed into them and no file inside them modified. Every attachment point is
a path or dotted name passed through config.
"""

from .episode import DELEGATE_TOOL_NAME, EpisodeResult, HarnessAuditEpisode

__all__ = ["HarnessAuditEpisode", "EpisodeResult", "DELEGATE_TOOL_NAME"]
