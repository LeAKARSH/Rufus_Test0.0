"""News polling cycle glue (spec Section 3.2 / Phase 2).

Wires the CurrentsAPI client, the Yahoo earnings lookups, the rotation
planner and the sentiment pipeline into one calendar-day cycle. The cycle is
market-hours independent: news doesn't track the intraday price cadence, so
the scheduler runs it on its own interval (``news_poll_hours``), restart-safe
through the persistent ``job_state`` marker and the ``news_snapshots``
run-date uniqueness.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Callable
from zoneinfo import ZoneInfo

import rufus.db as db
from rufus.cache import TTLCache
from rufus.config import Settings
from rufus.currents import (
    CurrentsAuthError,
    CurrentsClient,
    CurrentsError,
    CurrentsNotConfigured,
    CurrentsQuotaExceeded,
)
from rufus.ollama import OllamaClient
from rufus.rate_limit import BudgetExhausted, RateLimiter
from rufus.rotation import plan_daily_news_rotation
from rufus.sentiment import score_ticker_news
from rufus.yahoo import YahooClient

log = logging.getLogger(__name__)


def market_today(tz_name: str) -> date:
    """The current calendar date in the market's local timezone (UTC fallback)."""
    try:
        return datetime.now(ZoneInfo(tz_name)).date()
    except Exception:
        return date.today()


def run_news_cycle(
    conn,
    settings: Settings,
    currents: CurrentsClient,
    yahoo: YahooClient,
    ollama: OllamaClient,
    today: date | None = None,
) -> list[dict]:
    """Execute one news cycle: plan -> pull -> score -> store; returns stats."""
    today = today or market_today(settings.market_timezone)

    if not settings.currentsapi_key:
        log.warning("news cycle skipped: CURRENTSAPI_KEY is not configured")
        return []

    tickers = db.get_active_tickers(conn)
    if not tickers:
        log.info("news cycle skipped: empty watchlist")
        return []

    try:
        earnings = yahoo.fetch_earnings_dates_map(tickers)
    except Exception:
        log.warning("earnings lookup failed; continuing without it", exc_info=True)
        earnings = {}

    budget = currents.limiter.remaining()
    plan = plan_daily_news_rotation(
        conn, budget, today=today, earnings_dates=earnings, settings=settings
    )
    if not plan:
        log.info("news cycle planned nothing for %s (budget=%d)", today, budget)
        return []

    stats = []
    for item in plan:
        try:
            articles = currents.search(item.keywords)
        except BudgetExhausted as exc:
            log.warning("news cycle aborted: %s", exc)
            break
        except CurrentsQuotaExceeded as exc:
            retry = f" (retry after {exc.retry_after}s)" if exc.retry_after else ""
            log.error("news pull quota-exceeded for %s%s", item.ticker, retry)
            continue
        except (CurrentsNotConfigured, CurrentsAuthError,
                CurrentsError) as exc:
            log.error("news pull failed for %s: %s", item.ticker, exc)
            continue

        run_date = str(today)
        if not articles:
            log.info("no articles for %s (%s)", item.ticker, item.keywords)
            db.insert_news_snapshot(
                conn, item.ticker, run_date=run_date, query_keyword=item.keywords,
                articles_considered=0,
            )
            continue

        agg = score_ticker_news(
            ollama, conn, item.ticker, item.keywords, run_date,
            articles, settings.news_sentiment_max_articles,
        )
        stats.append(
            {
                "ticker": item.ticker,
                "articles": len(articles),
                "score": agg and agg["sentiment_score_avg"],
            }
        )
        log.info(
            "news scored for %s: %d articles, avg=%s",
            item.ticker, len(articles),
            agg and agg["sentiment_score_avg"],
        )

    log.info(
        "news cycle done for %s: %d ticker(s) scored, %d request(s) spent",
        today, len(stats),
        currents.limiter.max_requests - currents.limiter.remaining(),
    )
    return stats


def create_news_poll_fn(settings: Settings, conn) -> "Callable":
    """Build the scheduler's news-cycle callback backed by this module."""
    currents = CurrentsClient(
        api_key=settings.currentsapi_key,
        limiter=RateLimiter(
            "currents", settings.currents_max_req_per_day, "daily", conn
        ),
        retry_attempts=settings.currents_retry_attempts,
        retry_base_delay_s=settings.retry_base_delay_s,
        retry_jitter_s=settings.retry_jitter_s,
    )
    yahoo = YahooClient(
        limiter=RateLimiter(
            "yahoo", settings.yahoo_max_req_per_hour, "hourly", conn
        ),
        cache=TTLCache(),
        retry_attempts=settings.yahoo_retry_attempts,
        retry_base_delay_s=settings.retry_base_delay_s,
        retry_jitter_s=settings.retry_jitter_s,
    )
    ollama = OllamaClient(
        base_url=settings.sentiment_base_url,
        model=settings.sentiment_ollama_model,
        timeout=settings.ollama_timeout_seconds,
        max_attempts=settings.ollama_retry_attempts,
        retry_base_delay_s=settings.retry_base_delay_s,
        retry_jitter_s=settings.retry_jitter_s,
    )

    def poll(c) -> None:
        run_news_cycle(c, settings, currents, yahoo, ollama)

    return poll