"""Yahoo Finance data module (spec Section 3.1).

The client fetches daily OHLCV history and quote/fundamental data per ticker,
always through the shared rate limiter and a TTL cache so repeated polls
within/ across cycles do not re-hit Yahoo unnecessarily.

The two fetch steps below are deliberately factored into module-level
functions so tests can monkeypatch them without touching yfinance:
    - ``_fetch_history``   -> yfinance ``Ticker.history``
    - ``_fetch_info``      -> yfinance ``Ticker.get_info()`` / ``.info``

Each upstream call "costs" one request against the Yahoo budget.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

import rufus.db as db
from rufus.cache import TTLCache
from rufus.config import (
    DEFAULT_RETRY_BASE_DELAY_S,
    DEFAULT_RETRY_JITTER_S,
    DEFAULT_YAHOO_RETRY_ATTEMPTS,
    Settings,
    get_settings,
)
from rufus.indicators import compute_indicators
from rufus.rate_limit import BudgetExhausted, RateLimiter
from rufus.retry import default_is_transient, retry_call

RequestBudgetExhausted = BudgetExhausted

log = logging.getLogger(__name__)

HISTORY_TTL = timedelta(minutes=15)
INFO_TTL = timedelta(hours=6)
EARNINGS_TTL = timedelta(hours=24)

# Cache marker for "no earnings date known" (TTLCache treats None as a miss).
_EARNINGS_MISS = object()

# Which fundamentals we persist in ``data_json`` (kept small on purpose).
_INFO_KEYS = (
    "sector",
    "industry",
    "marketCap",
    "trailingPE",
    "forwardPE",
    "dividendYield",
    "earningsGrowth",
    "revenueGrowth",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "regularMarketPrice",
)


def _dividend_yield(value: Any) -> float | None:
    """Newer yfinance reports ``dividendYield`` as a percent already
    (0.32 == 0.32%), not a fraction; store it verbatim."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fetch_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    import yfinance as yf

    return yf.Ticker(ticker).history(period=period, interval=interval)


def _fetch_info(ticker: str) -> dict[str, Any]:
    import yfinance as yf

    t = yf.Ticker(ticker)
    if hasattr(t, "get_info"):
        return t.get_info()
    return t.info


def _fetch_calendar(ticker: str) -> Any:
    """Raw earnings calendar from yfinance (dict or DataFrame)."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    if hasattr(t, "get_calendar"):
        return t.get_calendar()
    return getattr(t, "calendar", None)


def _as_date(value: Any) -> date | None:
    if value is None or _pd_isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        d = value if value.tzinfo is None else value.astimezone(timezone.utc)
        return d.date()
    if isinstance(value, date):
        return value
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError):
        return None


def _pd_isna(value: Any) -> bool:
    try:
        return pd.isna(value) or value is pd.NaT
    except (TypeError, ValueError):
        return False


def _parse_earnings_date(raw: Any) -> date | None:
    """Extract the next earnings date from yfinance calendar output."""
    if raw is None:
        return None
    value = raw.get("Earnings Date") if isinstance(raw, Mapping) else raw

    if isinstance(value, pd.DataFrame):
        if "Earnings Date" in value.index:
            value = value.loc["Earnings Date"]
        elif "Earnings Date" in value.columns:
            value = value["Earnings Date"]
        try:
            return _as_date(value.iloc[0])
        except (IndexError, KeyError):
            return None

    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return _as_date(value)


class YahooClient:
    """Fetches Yahoo data with shared rate limiting and response caching."""

    def __init__(
        self,
        limiter: RateLimiter,
        cache: TTLCache | None = None,
        retry_attempts: int = DEFAULT_YAHOO_RETRY_ATTEMPTS,
        retry_base_delay_s: float = DEFAULT_RETRY_BASE_DELAY_S,
        retry_jitter_s: float = DEFAULT_RETRY_JITTER_S,
    ) -> None:
        self.limiter = limiter
        self.cache = cache or TTLCache(default_ttl=HISTORY_TTL)
        self.retry_attempts = retry_attempts
        self.retry_base_delay_s = retry_base_delay_s
        self.retry_jitter_s = retry_jitter_s

    # ------------------------------------------------------------------ #
    # Public fetchers

    def fetch_history(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        ttl: timedelta | None = None,
    ) -> pd.DataFrame:
        """Daily OHLCV history, cached until the TTL expires."""
        key = f"hist:{ticker.upper()}:{period}:{interval}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        self._acquire()
        df = self._retry(lambda: _fetch_history(ticker.upper(), period=period, interval=interval))
        self.cache.set(key, df, ttl=ttl or HISTORY_TTL)
        return df

    def fetch_info(self, ticker: str) -> dict[str, Any]:
        """Quote/fundamental info dict, cached until the TTL expires."""
        key = f"info:{ticker.upper()}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        self._acquire()
        info = self._retry(lambda: _fetch_info(ticker.upper()))
        self.cache.set(key, info, ttl=INFO_TTL)
        return info

    def fetch_company_name(self, ticker: str) -> str | None:
        """Human-readable company name for seeding news search keywords."""
        info = self.fetch_info(ticker)
        return info.get("longName") or info.get("shortName") or None

    def fetch_earnings_dates(self, ticker: str) -> date | None:
        """Next reported earnings date (cached 24h); ``None`` when unknown."""
        key = f"earn:{ticker.upper()}"
        cached = self.cache.get(key)
        if cached is not None:
            return None if cached is _EARNINGS_MISS else cached
        self._acquire()
        result = _parse_earnings_date(self._retry(lambda: _fetch_calendar(ticker.upper())))
        self.cache.set(
            key,
            result if result is not None else _EARNINGS_MISS,
            ttl=EARNINGS_TTL,
        )
        return result

    def fetch_benchmark_close(
        self,
        ticker: str,
        period: str = "5d",
        ttl: timedelta | None = None,
    ) -> tuple[date, float] | None:
        """Latest daily close for a market index (e.g. ``^NSEI``).

        Returns ``(trade_date, close)`` from the most recent non-empty
        session, or ``None`` when history is unavailable. Cached alongside
        the ordinary history fetch so at most one Yahoo call/day is spent.
        """
        key = f"bench:{ticker.upper()}"
        cached = self.cache.get(key)
        if cached is not None:
            return None if cached is _EARNINGS_MISS else cached
        df = self.fetch_history(ticker, period=period, interval="1d", ttl=ttl)
        if df is None or df.empty or "Close" not in df.columns:
            self.cache.set(key, _EARNINGS_MISS, ttl=ttl or HISTORY_TTL)
            return None
        closes = df.dropna(subset=["Close"])
        if closes.empty:
            self.cache.set(key, _EARNINGS_MISS, ttl=ttl or HISTORY_TTL)
            return None
        last = closes.iloc[-1]
        result = (_as_date(last.name), float(last["Close"]))
        self.cache.set(key, result, ttl=ttl or HISTORY_TTL)
        return result

    def fetch_earnings_dates_map(
        self,
        tickers: list[str],
    ) -> dict[str, date]:
        """Upcoming earnings dates for ``tickers``; unknown ones are omitted."""
        out: dict[str, date] = {}
        for ticker in tickers:
            when = self.fetch_earnings_dates(ticker)
            if when is not None:
                out[ticker.upper()] = when
        return out

    def fetch_snapshot(self, ticker: str) -> dict[str, Any] | None:
        """Compact per-ticker snapshot for storing into ``price_snapshots``.

        Costs at most two budgeted requests (history + info); returns ``None``
        when Yahoo has neither history nor info for the ticker.
        """
        ticker = ticker.upper()
        history = self.fetch_history(ticker)
        info = self.fetch_info(ticker)

        if history.empty and not info:
            return None

        indicators = compute_indicators(history)
        price = indicators.get("price")
        if price is None:
            raw = info.get("regularMarketPrice") or info.get("currentPrice")
            price = float(raw) if raw is not None else None

        high_52w = indicators.get("high_52w") or info.get("fiftyTwoWeekHigh")
        low_52w = indicators.get("low_52w") or info.get("fiftyTwoWeekLow")

        snapshot: dict[str, Any] = {
            "price": price,
            "sma_50": indicators.get("sma_50"),
            "sma_200": indicators.get("sma_200"),
            "trend_signal": indicators.get("trend_signal"),
            "rsi_14": indicators.get("rsi_14"),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": _dividend_yield(info.get("dividendYield")),
            "high_52w": high_52w,
            "low_52w": low_52w,
            "volatility_90d": indicators.get("volatility_90d"),
            "position_vs_52w_range_pct": indicators.get("position_vs_52w_range_pct"),
        }

        info_subset = {k: info.get(k) for k in _INFO_KEYS}
        history_tail: list[dict[str, Any]] = []
        if not history.empty:
            recent = history[["Open", "High", "Low", "Close", "Volume"]].tail(5)
            history_tail = [
                {
                    "date": str(ts.date()),
                    "open": float(r["Open"]),
                    "high": float(r["High"]),
                    "low": float(r["Low"]),
                    "close": float(r["Close"]),
                    "volume": float(r["Volume"]),
                }
                for ts, r in recent.iterrows()
            ]
        snapshot["data_json"] = json.dumps(
            {
                "info": info_subset,
                "history_tail": history_tail,
                "indicators": indicators.get("indicators"),
            }
        )
        return snapshot

    # ------------------------------------------------------------------ #
    # Internals

    def _acquire(self) -> None:
        if not self.limiter.try_acquire():
            raise RequestBudgetExhausted(
                f"{self.limiter.provider} budget exhausted "
                f"({self.limiter.used()}/{self.limiter.max_requests} used)"
            )

    def _retry(self, fn):
        """Run one budgeted upstream call with the configured retry policy.

        The budget was already acquired before the retry loop, so retries
        never re-charge the limit and the failure always propagates (the poll
        loop is responsible for skip-and-continue).
        """
        return retry_call(
            fn,
            attempts=self.retry_attempts,
            base_delay_s=self.retry_base_delay_s,
            jitter_s=self.retry_jitter_s,
            is_transient=default_is_transient,
        )


def create_poll_fn(settings: Settings, conn) -> "PollFn":
    """Build the scheduler's poll callback backed by this module.

    Each pass walks the active tickers, fetches a snapshot per ticker and
    stores it. A single exhausted budget aborts the remaining pass early
    (there is no point hammering a dead budget); per-ticker failures are
    logged and skipped.
    """
    limiter = RateLimiter(
        "yahoo", settings.yahoo_max_req_per_hour, "hourly", conn
    )
    cache = TTLCache(default_ttl=HISTORY_TTL)
    client = YahooClient(
        limiter=limiter,
        cache=cache,
        retry_attempts=settings.yahoo_retry_attempts,
        retry_base_delay_s=settings.retry_base_delay_s,
        retry_jitter_s=settings.retry_jitter_s,
    )

    def poll(conn, tickers: list[str]) -> None:
        stored = 0
        for ticker in tickers:
            try:
                snapshot = client.fetch_snapshot(ticker)
                if snapshot is None:
                    log.warning("no data for %s", ticker)
                    continue
                db.insert_price_snapshot(conn, ticker, **snapshot)
                stored += 1
                seed_keywords(conn, client, ticker)
                log.info("stored snapshot for %s (price=%s)", ticker, snapshot.get("price"))
            except RequestBudgetExhausted as exc:
                log.warning("aborting poll pass: %s", exc)
                break
            except Exception:
                log.exception(
                    "poll failed for %s (yahoo budget %s/%s used)",
                    ticker, limiter.used(), limiter.max_requests,
                )
        log.info("poll pass done: %d/%d stored", stored, len(tickers))

    return poll


def _plain_stem(ticker: str) -> str:
    """Ticker without its exchange suffix (e.g. ``RELIANCE.NS`` -> ``RELIANCE``)."""
    return ticker.upper().split(".")[0]


# Legal-suffix / generic tokens that add nothing to a news keyword phrase.
_COMPANY_NOISE_WORDS = {
    "limited", "ltd", "inc", "incorporated", "corp", "corporation", "company",
    "com", "plc", "llc", "holdings", "holding", "group", "co", "sa",
}


def _company_phrase(name: str, ticker: str) -> str:
    """Compact, web-search-friendly phrase from the company long name.

    Drops legal suffixes ("Reliance Industries Limited" -> "Reliance
    Industries") because CurrentsAPI matches the quoted phrase exactly, so
    the full legal name matches nothing.
    """
    words = []
    for token in name.split():
        cleaned = token.strip(".,-&'").lower()
        if cleaned in _COMPANY_NOISE_WORDS or cleaned in {"and", "the", "of", "for"}:
            continue
        words.append(token.strip(".,"))
    return " ".join(words) or _plain_stem(ticker)


def seed_keywords(conn, client: YahooClient, ticker: str) -> None:
    """Auto-seed search keywords from the company name when not yet set.

    Uses the already-cached yfinance info (no extra budgeted call when the
    snapshot just fetched it). Stored verbatim as a user-editable CurrentsAPI
    search string, e.g. ``"Reliance Industries" OR RELIANCE``.
    """
    if db.get_ticker_keywords(conn, ticker):
        return
    name = client.fetch_company_name(ticker)
    if not name:
        return
    keywords = f'"{_company_phrase(name, ticker)}" OR {_plain_stem(ticker)}'
    db.set_ticker_keywords(conn, ticker, keywords)
    log.info("seeded news keywords for %s: %r", ticker, keywords)


def default_poll_fn(conn) -> "PollFn":
    """Convenience using the process-wide settings (for scheduler.main())."""
    return create_poll_fn(get_settings(), conn)