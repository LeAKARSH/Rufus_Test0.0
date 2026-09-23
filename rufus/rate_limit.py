"""Shared rate limiting utility.

The limiter enforces API budgets against the persistent ``api_usage_log``
table, which means budgets survive process restarts (Section 2.1 crash/restart
resilience) and are auditable after the fact.

Window semantics map directly onto the upstream quota reset behaviour:
- ``hourly`` -> Yahoo Finance (1000 requests per rolling calendar hour)
- ``daily``  -> CurrentsAPI (100 requests per calendar day)

This is intentionally a fixed bucket window rather than a token bucket: the
metrics that matter are "how many requests have we already spent in the
current quota window" and "when does the window reset".
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Literal

import rufus.db as db

WindowKind = Literal["hourly", "daily"]

_HOURLY_FMT = "%Y-%m-%dT%H"
_DAILY_FMT = "%Y-%m-%d"


class BudgetExhausted(Exception):
    """The provider budget for the current window has been spent."""


def bucket_key(provider: str, window: WindowKind, when: datetime | None = None) -> str:
    """Stable bucket identifier for the api_usage_log table."""
    ts = when or datetime.now(timezone.utc)
    if window == "hourly":
        return f"{provider}:{ts.strftime(_HOURLY_FMT)}"
    return f"{provider}:{ts.strftime(_DAILY_FMT)}"


class RateLimiter:
    """Enforces a persistent per-window request budget for one provider.

    Thread-safe for a single process; writes go through serialized SQLite
    transactions, so concurrent use from multiple threads is safe too.

    Example::

        limiter = RateLimiter("yahoo", settings.yahoo_max_req_per_hour, "hourly", conn)
        if limiter.try_acquire():
            _do_the_call()
    """

    def __init__(
        self,
        provider: str,
        max_requests: int,
        window: WindowKind,
        conn: sqlite3.Connection,
    ) -> None:
        self.provider = provider
        self.max_requests = int(max_requests)
        self.window = window
        self.conn = conn

    # ------------------------------------------------------------------ #
    # Introspection

    def used(self) -> int:
        """Requests already spent in the current window."""
        return db.get_api_usage(self.conn, self.provider, self.current_bucket())

    def remaining(self) -> int:
        """Requests left in the current window (never negative)."""
        return max(0, self.max_requests - self.used())

    def current_bucket(self) -> str:
        return bucket_key(self.provider, self.window)

    # ------------------------------------------------------------------ #
    # Acquiring

    def try_acquire(self, count: int = 1) -> bool:
        """Grant ``count`` requests if they fit in the current window.

        Returns ``True`` and records the usage on success, ``False`` when the
        window budget is exhausted (no partial grants).
        """
        if count <= 0:
            return True
        bucket = self.current_bucket()
        used = db.get_api_usage(self.conn, self.provider, bucket)
        if used + count > self.max_requests:
            return False
        db.increment_api_usage(self.conn, self.provider, bucket, count)
        return True