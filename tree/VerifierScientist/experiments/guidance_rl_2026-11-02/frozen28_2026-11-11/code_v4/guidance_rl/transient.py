"""Bounded retry for transient API failures, shared by reward and optimizer call paths.

Moved out of reward.py so optimize.py can use it without a circular import
(reward imports optimize.judge_record).
"""
from __future__ import annotations
import time

TRANSIENT_RETRY_DELAYS_S=(10,30,60,120,240,300)


def transient_api_error(error):
    """Server-side or network failures that a later identical request may clear.

    4xx client errors other than rate limiting are not transient and never retry.
    BudgetExceeded is never retried: the ledger cap is the hard stop.
    """
    import openai
    return isinstance(error,(openai.APIConnectionError,openai.InternalServerError,openai.RateLimitError))


def judge_with_transient_retry(call,delays=TRANSIENT_RETRY_DELAYS_S,sleep=time.sleep,log=None):
    """Retry ``call()`` only on transient API failures, with bounded backoff.

    Each failed attempt is already charged at its worst-case cost by the client
    ledger, so retries are bounded by len(delays) and cannot bypass the cap. A
    persistent outage re-raises the final error after the last delay.
    """
    attempt=0
    while True:
        try:
            return call()
        except BaseException as error:
            if not transient_api_error(error) or attempt>=len(delays):raise
            delay=delays[attempt];attempt+=1
            if log is not None:log({'transient_error':type(error).__name__,'retry_attempt':attempt,'delay_s':delay,'time':time.time()})
            sleep(delay)
