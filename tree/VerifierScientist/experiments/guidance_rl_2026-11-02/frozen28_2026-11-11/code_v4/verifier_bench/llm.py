"""Thin OpenAI Responses wrapper: structured output, disk cache, cost ledger, budget cap.

Every call is keyed by sha256(model, instructions, input, schema name); a cache
hit costs nothing, so re-running an experiment only pays for what changed.
Spend is appended to $OUT_ROOT/ledger.jsonl and summed before each call; a call
that would start above the cap raises BudgetExceeded instead of running.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel

from common import OUT_ROOT, openai_client

# USD per 1M tokens (standard tier; third-party listings, 2026-10-13 -- verify on
# OpenAI's pricing page before quoting in the paper).
PRICES = {
    "gpt-6-luna": (0.10, 0.50, 0.01),     # input, output, cached input
    "gpt-6-sol": (2.00, 10.00, 0.20),
}
DEFAULT_MODEL = "gpt-6-luna"
BUDGET_USD = 100.0

CACHE_DIR = OUT_ROOT / "cache"
LEDGER = OUT_ROOT / "ledger.jsonl"
_lock = threading.Lock()
_inflight: dict[str, threading.Lock] = {}
_client = None

T = TypeVar("T", bound=BaseModel)


class BudgetExceeded(RuntimeError):
    pass


def spent_usd() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in open(LEDGER) if l.strip())


def _cost(model: str, usage) -> float:
    pin, pout, pcache = PRICES.get(model, PRICES[DEFAULT_MODEL])
    cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
    return ((usage.input_tokens - cached) * pin + cached * pcache + usage.output_tokens * pout) / 1e6


def call(instructions: str, user: str, schema: Type[T], *, model: str = DEFAULT_MODEL,
         tag: str = "", budget: float | None = None, retries: int = 4, salt: str = "") -> T:
    """``salt`` makes an independent sample of the same prompt (e.g. "rep1");
    the empty salt keeps the original cache key. Concurrent identical calls
    wait for the first one instead of paying twice."""
    key = hashlib.sha256(json.dumps([model, instructions, user, schema.__name__]
                                    + ([salt] if salt else [])).encode()).hexdigest()
    with _lock:
        key_lock = _inflight.setdefault(key, threading.Lock())
    with key_lock:
        return _call(key, instructions, user, schema, model, tag, budget, retries)


def _call(key, instructions, user, schema, model, tag, budget, retries):
    global _client
    path = CACHE_DIR / key[:2] / f"{key}.json"
    if path.exists():
        return schema.model_validate_json(json.loads(path.read_text())["output"])

    with _lock:
        budget = BUDGET_USD if budget is None else budget
        if spent_usd() >= budget:
            raise BudgetExceeded(f"spent ${spent_usd():.2f} >= cap ${budget:.2f}")
        if _client is None:
            _client = openai_client()

    for attempt in range(retries):
        try:
            resp = _client.responses.parse(model=model, instructions=instructions, input=user,
                                           text_format=schema)
            break
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001 -- rate limits / transient errors
            if attempt == retries - 1:
                raise
            time.sleep(5 * 2 ** attempt)
            last = e  # noqa: F841
    parsed = resp.output_parsed
    usd = _cost(model, resp.usage)
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": model, "tag": tag, "output": parsed.model_dump_json()}))
        with open(LEDGER, "a") as f:
            f.write(json.dumps({"t": time.time(), "model": model, "tag": tag, "usd": usd,
                                "in": resp.usage.input_tokens, "out": resp.usage.output_tokens}) + "\n")
    return parsed
