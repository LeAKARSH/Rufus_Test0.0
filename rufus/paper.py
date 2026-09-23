"""Paper-trading day orchestration (spec Section 7).

Runs the full simulation for one decision day against the default virtual
portfolio:

1. read today's stored recommendations,
2. price fills from the stored price snapshots (no new network calls),
3. execute the trades through :mod:`rufus.simulation`,
4. record the equity-curve point (with the day's benchmark close),
5. persist the benchmark index close for the buy-and-hold comparison.

Benchmark data is the *only* network touch and only happens inside
``create_paper_cycle`` (via an injected client) — the orchestration itself
(``run_paper_cycle``) is pure of network imports.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import rufus.db as db
from rufus.cache import TTLCache
from rufus.rate_limit import RateLimiter
from rufus.simulation import prices_from_snapshots, run_simulation_cycle
from rufus.sizing import make_sizer
from rufus.valuation import record_equity_snapshot, snapshot, store_benchmark_for_day
from rufus.yahoo import YahooClient

log = logging.getLogger(__name__)


def market_today(tz_name: str) -> str:
    try:
        return datetime.now(ZoneInfo(tz_name)).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def run_paper_cycle(
    conn,
    settings,
    recs=None,
    prices: dict[str, float] | None = None,
    benchmark_client=None,
    debug: dict | None = None,
    captured_at: str | None = None,
) -> dict:
    """Execute one decision day's simulation and record its equity point.

    ``recs`` defaults to the latest stored recommendations for the active
    watchlist; ``prices`` defaults to the newest stored snapshot per ticker.
    ``benchmark_client`` (a :class:`rufus.yahoo.YahooClient`) enables the
    NIFTY/benchmark close persistence for the day.
    """
    portfolio_row = db.ensure_default_portfolio(
        conn, name=settings.portfolio_name, starting_cash=settings.starting_cash
    )
    portfolio_id = portfolio_row["id"]

    if recs is None:
        tickers = db.get_active_tickers(conn)
        recs = db.get_latest_recommendations(conn, tickers)
    prices = prices if prices is not None else prices_from_snapshots(conn, recs)

    sizer = make_sizer(settings.position_sizing_strategy)
    portfolio_value = snapshot(conn, portfolio_id, prices)["total_value"]

    results = run_simulation_cycle(
        conn, portfolio_id, recs, prices, sizer,
        portfolio_value=portfolio_value,
        max_allocation_pct=settings.max_allocation_pct,
        starting_cash=settings.starting_cash,
    )

    benchmark_value = None
    if benchmark_client is not None and settings.benchmark_ticker:
        pair = benchmark_client.fetch_benchmark_close(settings.benchmark_ticker)
        if pair is not None:
            trade_date, close = pair
            store_benchmark_for_day(conn, settings.benchmark_ticker, str(trade_date), close)
            benchmark_value = float(close)

    snap = record_equity_snapshot(
        conn, portfolio_id, captured_at or db.utc_now_iso(), prices,
        benchmark_value=benchmark_value,
    )
    log.info(
        "paper cycle for %d recommendation(s): %s",
        len(results),
        {r["ticker"]: r["action"] for r in results},
    )
    return {
        "portfolio_id": portfolio_id,
        "results": results,
        "snapshot": snap,
        "benchmark_value": benchmark_value,
    }


def create_paper_cycle(settings, conn):
    """Build the scheduler's paper-trading callback (with benchmark client)."""
    client = YahooClient(
        limiter=RateLimiter(
            "yahoo", settings.yahoo_max_req_per_hour, "hourly", conn
        ),
        cache=TTLCache(),
        retry_attempts=settings.yahoo_retry_attempts,
        retry_base_delay_s=settings.retry_base_delay_s,
        retry_jitter_s=settings.retry_jitter_s,
    )

    def cycle(conn_obj) -> None:
        run_paper_cycle(conn_obj, settings, benchmark_client=client)

    return cycle


def refresh_equity(conn, settings, captured_at: str | None = None) -> None:
    """Re-record the current total value using latest stored snapshots.

    Called after each intraday price poll: re-valuates open positions at the
    freshest stored prices (zero network calls) and updates the equity curve
    point at the current timestamp.
    """
    portfolio_row = db.ensure_default_portfolio(
        conn, name=settings.portfolio_name, starting_cash=settings.starting_cash
    )
    prices = {}
    for pos in db.list_open_positions(conn, portfolio_row["id"]):
        rows = db.get_recent_price_snapshots(conn, pos["ticker"], limit=1)
        if rows and rows[0]["price"] is not None:
            prices[pos["ticker"]] = float(rows[0]["price"])
    record_equity_snapshot(
        conn, portfolio_row["id"], captured_at or db.utc_now_iso(), prices
    )


def paper_summary(conn, settings) -> dict:
    """Current paper-trading state for reports/CLI (no network calls)."""
    from rufus.valuation import benchmark_since_inception, equity_curve, snapshot

    portfolio_row = db.ensure_default_portfolio(
        conn, name=settings.portfolio_name, starting_cash=settings.starting_cash
    )
    pid = portfolio_row["id"]
    prices = {}
    for pos in db.list_open_positions(conn, pid):
        rows = db.get_recent_price_snapshots(conn, pos["ticker"], limit=1)
        if rows and rows[0]["price"] is not None:
            prices[pos["ticker"]] = float(rows[0]["price"])
    snap = snapshot(conn, pid, prices)
    return {
        "portfolio": portfolio_row["name"],
        "benchmark_ticker": settings.benchmark_ticker if settings.benchmark_ticker else None,
        "benchmark_return": benchmark_since_inception(conn, settings.benchmark_ticker),
        "equity_points": len(equity_curve(conn, pid)),
        **snap,
    }