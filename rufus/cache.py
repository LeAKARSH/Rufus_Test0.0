"""A small thread-safe TTL cache used to avoid redundant upstream calls.

Repeated requests within a decision cycle (or across cycles, within the TTL)
are served from memory instead of re-hitting Yahoo Finance / CurrentsAPI,
which stretches the rate-limit budgets further (Section 3).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass
class _Entry:
    value: Any
    expires_at: float  # monotonic seconds


class TTLCache(Generic[T]):
    """Key-value store with per-entry expiry, keyed by ``str``.

    Expiry defaults to ``default_ttl`` and can be overridden per entry. A
    ``ttl`` of ``None`` (or a negative/zero duration) disables expiry.
    """

    def __init__(self, default_ttl: timedelta = timedelta(hours=1)) -> None:
        self._default_ttl = default_ttl
        self._store: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Mutation

    def set(self, key: str, value: Any, ttl: timedelta | None = None) -> None:
        if ttl is None:
            ttl = self._default_ttl
        expires = float("inf") if _disabled(ttl) else time.monotonic() + ttl.total_seconds()
        with self._lock:
            self._store[key] = _Entry(value=value, expires_at=expires)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    # ------------------------------------------------------------------ #
    # Reads

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at <= time.monotonic():
                del self._store[key]
                return None
            return entry.value

    def get_or_set(self, key: str, factory: Callable[[], T], ttl: timedelta | None = None) -> T:
        """Return the cached value or compute, store, and return it."""
        value = self.get(key)
        if value is not None:
            return value
        value = factory()
        self.set(key, value, ttl=ttl)
        return value

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def size(self) -> int:
        with self._lock:
            self._purge_locked()
            return len(self._store)

    def keys(self) -> list[str]:
        with self._lock:
            self._purge_locked()
            return list(self._store.keys())

    # ------------------------------------------------------------------ #
    # Internals

    def _purge_locked(self) -> None:
        now = time.monotonic()
        expired = [
            k for k, e in self._store.items() if e.expires_at <= now
        ]
        for k in expired:
            del self._store[k]


def _disabled(ttl: timedelta) -> bool:
    return ttl is None or ttl.total_seconds() <= 0