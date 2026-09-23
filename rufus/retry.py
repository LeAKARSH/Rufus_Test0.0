"""Shared retry-with-backoff utility (Phase 6 hardening).

The three upstream clients (Yahoo Finance, CurrentsAPI, Ollama) retry
transient failures with exponential backoff + jitter through this helper so
the policy lives in one place. Rate-limit budgets are consumed *once per
logical request* before the retry loop; a retried request is never re-charged
against the quota.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")

log = logging.getLogger(__name__)

# Transport-level failures worth retrying. ``requests`` exceptions subclass
# OSError/IOError, so this set catches them too.
_TRANSIENT_TYPES = (ConnectionError, TimeoutError, OSError)


def default_is_transient(exc: BaseException) -> bool:
    """Default retry predicate: network/transport-level errors only."""
    return isinstance(exc, _TRANSIENT_TYPES)


def http_status(exc: BaseException) -> int | None:
    """HTTP status attached to an exception, when the client attaches one."""
    return getattr(exc, "status", None)


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int,
    base_delay_s: float = 1.0,
    jitter_s: float = 0.25,
    is_transient: Callable[[BaseException], bool] | None = None,
    on_retry: Callable[[int, BaseException], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` up to ``attempts`` times, backing off between retries.

    ``attempts`` is the total number of tries (1 == no retry). A failure is
    retried only when ``is_transient`` says so; anything else propagates
    immediately. Backoff is ``base_delay_s * 2 ** (attempt - 1)`` plus up to
    ``jitter_s`` random seconds. The last failure is always re-raised.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    predicate = is_transient or default_is_transient
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except BaseException as exc:
            if attempt >= attempts or not predicate(exc):
                raise
            delay = base_delay_s * (2 ** (attempt - 1)) + random.uniform(0, jitter_s)
            if on_retry is not None:
                on_retry(attempt, exc)
            log.warning(
                "retrying in %.2fs after %s: %s (attempt %d/%d)",
                delay, type(exc).__name__, exc, attempt, attempts,
            )
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover