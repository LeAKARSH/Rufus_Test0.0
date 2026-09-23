"""CurrentsAPI news module (spec Section 3.2).

Fetch recent news articles per search-keyword block through the shared
daily rate limiter and a TTL cache. Every upstream call costs exactly one
request against the CurrentsAPI daily budget (100 on the free plan), so a
search is only issued when the budget has room *before* hitting the wire.

Upstream contract (v1, confirmed against the OpenAPI spec):
    GET  https://api.currentsapi.services/v1/search
    auth: ``Authorization: Bearer <key>`` (bare key also accepted)
    keywords support websearch syntax: quoted phrases, OR, -exclusion
    free plan: 7-day window span, 30-day lookback, 20 results per page,
    100 requests/day, quota resets 00:00 UTC; 429 carries Retry-After.

The module-level ``_get`` is factored out so tests can monkeypatch it
without making live requests.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import requests

from rufus.cache import TTLCache
from rufus.config import (
    DEFAULT_CURRENTS_RETRY_ATTEMPTS,
    DEFAULT_RETRY_BASE_DELAY_S,
    DEFAULT_RETRY_JITTER_S,
)
from rufus.rate_limit import BudgetExhausted, RateLimiter
from rufus.retry import default_is_transient, http_status, retry_call

log = logging.getLogger(__name__)

BASE_URL = "https://api.currentsapi.services/v1/search"

# Free-plan limits (see OpenAPI spec).
SEARCH_DAYS = 7  # maximum span on the free plan
RESULT_LIMIT = 20  # maximum results per page on the free plan
NEWS_TTL = timedelta(hours=6)

# Response error body shape we tolerate for diagnostics.
_ERR_KEYS = ("status", "msg", "details")


class CurrentsError(Exception):
    """Unclassified CurrentsAPI failure."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class CurrentsAuthError(CurrentsError):
    """The API key is missing, invalid, or revoked (HTTP 401)."""


class CurrentsQuotaExceeded(CurrentsError):
    """Daily quota or burst limit hit (HTTP 429). Never retry in-cycle."""

    def __init__(self, message: str, retry_after: int | None = None) -> None:
        super().__init__(message, status=429)
        self.retry_after = retry_after


class CurrentsNotConfigured(CurrentsError):
    """No API key configured for CurrentsAPI."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _get(url: str, params: Mapping[str, Any],
         headers: Mapping[str, str], timeout: float = 15.0) -> tuple[int, Any, requests.structures.CaseInsensitiveDict]:
    """Perform the HTTP GET; separated so tests can monkeypatch it."""
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    try:
        payload = resp.json()
    except (ValueError, json.JSONDecodeError):
        payload = None
    return resp.status_code, payload, resp.headers


def _parse_published(value: Any) -> str | None:
    """Normalize ``2026-09-07 14:22:08 +0000`` to ISO-8601 (UTC aware)."""
    if not value:
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S %z")
    except (TypeError, ValueError):
        return None
    return parsed.isoformat()


def _normalize_article(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Map a raw CurrentsAPI news item onto our compact article shape."""
    image = raw.get("image")
    return {
        "id": raw.get("id"),
        "title": raw.get("title"),
        "description": raw.get("description") or "",
        "url": raw.get("url"),
        "source": raw.get("author") or None,
        "published": _parse_published(raw.get("published")),
        "language": raw.get("language"),
        "categories": list(raw.get("category") or []),
    }


class CurrentsClient:
    """Searches CurrentsAPI under the shared daily rate limit and cache."""

    def __init__(
        self,
        api_key: str,
        limiter: RateLimiter,
        cache: TTLCache | None = None,
        retry_attempts: int = DEFAULT_CURRENTS_RETRY_ATTEMPTS,
        retry_base_delay_s: float = DEFAULT_RETRY_BASE_DELAY_S,
        retry_jitter_s: float = DEFAULT_RETRY_JITTER_S,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.limiter = limiter
        self.cache = cache or TTLCache(default_ttl=NEWS_TTL)
        self.retry_attempts = retry_attempts
        self.retry_base_delay_s = retry_base_delay_s
        self.retry_jitter_s = retry_jitter_s

    @property
    def available(self) -> bool:
        """True when a key is configured and the budget isn't exhausted."""
        if not self.api_key:
            return False
        return self.limiter.remaining() > 0

    def search(
        self,
        keywords: str,
        language: str = "en",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = RESULT_LIMIT,
    ) -> list[dict[str, Any]]:
        """Search recent news for ``keywords``, newest first.

        Returns a list of normalized article dicts. Raises
        :class:`CurrentsNotConfigured` when no key is set and
        :class:`BudgetExhausted` when the daily CurrentsAPI budget is spent
        (before any request is made).
        """
        if not self.api_key:
            raise CurrentsNotConfigured("CurrentsAPI key is not configured")

        cache_key = (
            f"currents:sq:{keywords}:{language}:{limit}:"
            f"{start_date and start_date.isoformat()}:{end_date and end_date.isoformat()}"
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        if not self.limiter.try_acquire():
            raise BudgetExhausted(
                f"{self.limiter.provider} budget exhausted "
                f"({self.limiter.used()}/{self.limiter.max_requests} used)"
            )

        end = end_date or _utc_now()
        start = start_date or (end - timedelta(days=SEARCH_DAYS))
        params = {
            "keywords": keywords,
            "language": language,
            "start_date": start.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "end_date": end.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "limit": min(int(limit), RESULT_LIMIT),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        payload = retry_call(
            lambda: self._search_once(params, headers),
            attempts=self.retry_attempts,
            base_delay_s=self.retry_base_delay_s,
            jitter_s=self.retry_jitter_s,
            is_transient=self._is_transient,
        )

        articles = [
            _normalize_article(raw)
            for raw in payload.get("news", [])
            if isinstance(raw, dict) and raw.get("title")
        ]
        articles.sort(
            key=lambda a: a["published"] or "",
            reverse=True,
        )
        self.cache.set(cache_key, articles, ttl=NEWS_TTL)
        log.info(
            "currents search %r -> %d articles (%s quota %s/%s)",
            keywords, len(articles), self.limiter.provider,
            self.limiter.used(), self.limiter.max_requests,
        )
        return articles

    # ------------------------------------------------------------------ #
    # Internals

    def _search_once(self, params: Mapping[str, Any], headers: Mapping[str, str]) -> Any:
        """One wire request with status handling; separated so retries stay
        inside the single budget charge acquired above."""
        status, payload, headers_resp = _get(BASE_URL, params, headers)
        self._raise_for_status(status, payload, headers_resp)
        return payload

    @staticmethod
    def _is_transient(exc: BaseException) -> bool:
        """Retry on transport failures and server errors only.

        HTTP 429 (quota/burst) and 401 (auth) are never retried in-cycle:
        a quota error can't be fixed by re-calling, and any such retry would
        burn budget that today's window can no longer grant.
        """
        if default_is_transient(exc):
            return True
        status = http_status(exc)
        return status is not None and status >= 500

    def _raise_for_status(
        self,
        status: int,
        payload: Any,
        headers: Mapping[str, str],
    ) -> None:
        found = f"{payload.get('msg') or payload.get('status') or payload}"

        if status == 401:
            raise CurrentsAuthError(f"CurrentsAPI rejected the API key: {found}")
        if status == 429:
            retry_after = headers.get("Retry-After")
            retry = int(retry_after) if str(retry_after or "").isdigit() else None
            raise CurrentsQuotaExceeded(
                f"CurrentsAPI daily quota or burst limit exceeded: {found}",
                retry_after=retry,
            )
        if status != 200:
            raise CurrentsError(f"CurrentsAPI HTTP {status}: {found}", status=status)
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise CurrentsError(f"CurrentsAPI unexpected response: {payload}")


def create_client(api_key: str, conn, max_requests: int) -> CurrentsClient:
    """Build a client wired to the persistent daily limiter (for callers)."""
    limiter = RateLimiter("currents", max_requests, "daily", conn)
    return CurrentsClient(api_key=api_key, limiter=limiter)