"""reward.py with exact rollout <-> dump pairing (use with tool_v09).

``reward.py`` claimed the oldest unclaimed dump of the task, which mis-pairs
GRPO's concurrent rollouts (see tool_v09.py). Here the rollout's own request id
arrives in ``extra_info["ha_request_id"]`` and the dump read is exactly
``<task_id>__<request_id>.json``. Everything after the lookup (rescore, report,
score = SAR_WEIGHT*SAR_avg + TCR_WEIGHT*TCR_det) is reward.py's.

Extra per-sample metrics (same keys on every path, as verl 0.8 requires):
  ha_dump_missing  request id known but its dump absent -> scored 0, logged
  has_violation    as in reward_fast.py (ViolRate in val metrics)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

_PKG_PARENT = str(Path(__file__).resolve().parents[1])
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from ha_verl import reward as _base  # noqa: E402

logger = logging.getLogger(__name__)

REQUEST_ID_KEY = "ha_request_id"   # = tool_v09.REQUEST_ID_KEY (not imported: that pulls in verl)


def _result(**overrides: float) -> dict[str, Any]:
    out = _base._result(ha_dump_missing=0.0, has_violation=0.0)
    out.update(overrides)
    return out


def _empty_episode(task_id: str, info: dict) -> dict[str, Any] | None:
    """reward.py's synthetic episode for a rollout that never called a tool."""
    task_yaml = str(info.get("task_yaml") or "")
    if not task_yaml:
        return None
    model = os.getenv("HA_MODEL_NAME") or str(info.get("model") or "unknown-model")
    return {
        "task_id": task_id,
        "task_path": task_yaml,
        "run_id": f"oa-oai-{model.replace('/', '-')}-noep-{task_id}",
        "domain": str(info.get("domain") or ""),
        "model": model,
        "hub_role": str(info.get("hub_role") or ""),
        "trace_path": None,
        "error": None,
        "completion": {"task_completion_score": 0.0},
        "actions": [],
    }


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: str = "",
    extra_info: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    from ha_verl.persist import dump_path_for, load_dump, rescore

    info = extra_info if isinstance(extra_info, dict) else {}
    task_id = str(ground_truth or info.get("task_id") or "").strip()
    if not task_id:
        return _result(ha_missing_task_id=1.0)

    request_id = str(info.get(REQUEST_ID_KEY) or "")
    no_episode = not request_id
    if no_episode:
        payload = _empty_episode(task_id, info)
        if payload is None:
            logger.warning("no tool call and no task_yaml for %s; scoring 0", task_id)
            return _result(ha_no_episode=1.0)
    else:
        path = dump_path_for(task_id, request_id)
        try:
            payload = load_dump(path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("dump %s unreadable (%s); scoring 0", path, exc)
            return _result(ha_dump_missing=1.0)
        try:   # bookkeeping only: any dump left without .used was never scored
            path.rename(path.with_name(path.name.replace(".json", ".used.json")))
        except OSError:
            pass

    try:
        report = rescore(payload, solution_str or "")
    except Exception as exc:
        logger.error("rescore failed for %s: %s", task_id, exc)
        return _result(ha_rescore_failed=1.0)

    results_dir = _base._results_dir(str(payload.get("model") or info.get("model") or ""))
    if results_dir is not None:
        out = results_dir / f"{report['task_id']}_{report['run_id']}.json"
        try:
            with open(out, "w") as f:
                json.dump(report, f, indent=2)
        except Exception as exc:
            logger.error("failed writing report %s: %s", out, exc)

    sar_avg = float(report.get("sar_avg") or 0.0)
    tcr_det = _base._tcr_det(report)
    n_viol = float(report.get("n_violations") or 0)
    return _result(
        score=_base.SAR_WEIGHT * sar_avg + _base.TCR_WEIGHT * tcr_det,
        sar_avg=sar_avg,
        sar_tool=float(report.get("sar_tool") or 0.0),
        sar_resource=float(report.get("sar_resource") or 0.0),
        sar_flow=float(report.get("sar_flow") or 0.0),
        tcr_det=tcr_det,
        tcr_full=float(report.get("tcr") or 0.0),
        n_violations=n_viol,
        n_tool_calls=float(report.get("n_tool_calls") or 0),
        violated=1.0 if n_viol > 0 else 0.0,
        has_violation=1.0 if n_viol > 0 else 0.0,
        ha_no_episode=1.0 if no_episode else 0.0,
    )
