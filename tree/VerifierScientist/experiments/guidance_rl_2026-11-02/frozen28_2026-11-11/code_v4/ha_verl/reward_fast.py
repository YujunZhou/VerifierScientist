"""compute_score of ha_verl.reward plus ``has_violation`` (1 if the trajectory
has any rule violation), so verl's validation metrics carry the paper's
ViolRate directly: val-aux/<data_source>/has_violation/mean@1. Kept as a
separate module so runs already importing ha_verl.reward are unaffected."""
from __future__ import annotations

from typing import Any

from ha_verl.reward import compute_score as _base_compute_score


def compute_score(*args: Any, **kwargs: Any) -> dict[str, Any]:
    out = _base_compute_score(*args, **kwargs)
    out["has_violation"] = 1.0 if float(out.get("n_violations") or 0.0) > 0 else 0.0
    return out
