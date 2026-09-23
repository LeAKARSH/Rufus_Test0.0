"""Daily news-budget rotation & prioritization (spec Section 3.2).

CurrentsAPI's free tier caps at 100 requests/day *shared across the whole
watchlist*, so not every ticker can be pulled every day. This module decides
which tickers get searched today:

- **Tier A** - tickers with a near-term catalyst: earnings within
  ``news_rotation_earnings_window_days`` days (from a supplied earnings map),
  a volatility spike in the latest ``price_snapshots`` relative to the prior
  one (``news_rotation_volatility_spike_pct``), or an overdue news pull
  (last ``news_snapshots.run_date`` older than ``news_rotation_overdue_days``,
  or never pulled). Tier A is guaranteed a pull within the budget.
- **Tier B** - everyone else, rotated with a deterministic day-to-day
  starting offset so the whole watchlist cycles through over time even when
  the budget can't cover it all in one day.

The plan only *suggests* how today's requests should be spent. The daily
``RateLimiter`` is the hard stop, so the caller passes the actual *remaining*
budget (``limiter.remaining()``), not the raw daily cap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Mapping

import rufus.db as db
from rufus.config import Settings, get_settings

log = logging.getLogger(__name__)

# Earnings within the last 2 days are still a post-earnings news catalyst.
_EARNINGS_LOOKBACK_DAYS = 2

_MISSING_RUN = 10**9  # sentinel: never pulled -> most overdue


@dataclass(frozen=True)
class RotationItem:
    """One planned news pull: search ``keywords`` for ``ticker``."""
    ticker: str
    keywords: str
    queries: int
    tier: str  # "A" (catalyst) or "B" (rotating)


def queries_per_ticker(budget: int, watchlist_size: int) -> int:
    """``100 // watchlist_size`` style per-ticker quota (minimum 0)."""
    if watchlist_size <= 0:
        return 0
    return max(0, budget // watchlist_size)


def plan_daily_news_rotation(
    conn,
    budget: int,
    today: date | None = None,
    earnings_dates: Mapping[str, str | date] | None = None,
    settings: Settings | None = None,
) -> list[RotationItem]:
    """Return today's ordered news-pull plan, Tier A first, budget-respecting."""
    today = today or date.today()
    settings = settings or get_settings()
    earnings_dates = earnings_dates or {}

    rows = db.get_active_tickers_with_keywords(conn)
    if not rows:
        return []

    budget = max(0, int(budget))
    quota = queries_per_ticker(budget, len(rows))

    tier_a: list[dict] = []
    tier_b: list[dict] = []
    for row in rows:
        ticker = row["ticker"]
        if _catalysts(
            conn, ticker, earnings_dates, today, settings
        ):
            tier_a.append(row)
        else:
            tier_b.append(row)

    # Deterministic day-to-day rotation start so Tier B cycles through the
    # whole watchlist over time even when today's budget can't cover it.
    if tier_b:
        offset = today.toordinal() % len(tier_b)
        tier_b = tier_b[offset:] + tier_b[:offset]

    plan: list[RotationItem] = []
    remaining = budget

    # Tier A first: guaranteed a pull, even when the flat quota is 0.
    for row in sorted(tier_a, key=lambda r: (-_days_since_run(conn, r["ticker"], today), r["ticker"])):
        if remaining <= 0:
            break
        allocated = min(quota if quota > 0 else 1, remaining)
        plan.append(_item(row, allocated, "A"))
        remaining -= allocated

    # Tier B second: only spends the flat quota; a leftover smaller than the
    # quota is deliberately not "wasted" on a quiet stock.
    if quota > 0:
        for row in tier_b:
            if remaining < quota:
                break
            plan.append(_item(row, quota, "B"))
            remaining -= quota

    log.info(
        "news rotation plan: %d pulls for %d tickers (budget=%d/%d, tier A=%d)",
        sum(i.queries for i in plan), len(plan),
        budget - remaining, budget, len(tier_a),
    )
    return plan


def _item(row, queries: int, tier: str) -> RotationItem:
    return RotationItem(
        ticker=row["ticker"],
        keywords=row["search_keywords"],
        queries=queries,
        tier=tier,
    )


# ---------------------------------------------------------------------------
# Catalyst detection
# ---------------------------------------------------------------------------

def _catalysts(conn, ticker: str, earnings_dates, today: date, settings: Settings) -> bool:
    if _earnings_within(earnings_dates, ticker, settings.news_rotation_earnings_window_days, today):
        return True
    if _volatility_spike(conn, ticker, settings.news_rotation_volatility_spike_pct):
        return True
    return _days_since_run(conn, ticker, today) >= settings.news_rotation_overdue_days


def _earnings_within(
    earnings_dates: Mapping[str, str | date],
    ticker: str,
    window_days: int,
    today: date,
) -> bool:
    when = earnings_dates.get(ticker)
    if not when:
        return False
    try:
        earnings = when if isinstance(when, date) else date.fromisoformat(str(when))
    except (TypeError, ValueError):
        return False
    delta = (earnings - today).days
    return -_EARNINGS_LOOKBACK_DAYS <= delta <= window_days


def _volatility_spike(conn, ticker: str, spike_pct: float) -> bool:
    rows = conn.execute(
        "SELECT volatility_90d FROM price_snapshots "
        "WHERE ticker = ? ORDER BY captured_at DESC LIMIT 2",
        (ticker.upper(),),
    ).fetchall()
    vols = [r["volatility_90d"] for r in rows if r["volatility_90d"] is not None]
    if len(vols) < 2 or not vols[1]:
        return False
    return (vols[0] / vols[1] - 1.0) >= max(0.0, spike_pct) / 100.0


def _days_since_run(conn, ticker: str, today: date) -> int:
    row = conn.execute(
        "SELECT MAX(run_date) AS last_run FROM news_snapshots WHERE ticker = ?",
        (ticker.upper(),),
    ).fetchone()
    last = row["last_run"] if row else None
    if not last:
        return _MISSING_RUN
    try:
        return (today - date.fromisoformat(last)).days
    except ValueError:
        return _MISSING_RUN