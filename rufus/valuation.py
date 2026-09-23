"""Portfolio valuation & equity curve (spec Section 7.3).

Pure of side effects where marked: ``snapshot`` / ``benchmark_return`` compute
from data handed to them, so they are trivially testable and outside the
network/rate-limit boundary. Persistence and any benchmark price casting are
deliberately thin wrappers around the DAOs.
"""

from __future__ import annotations

import logging

import rufus.db as db
from rufus import portfolio

log = logging.getLogger(__name__)


def snapshot(conn, portfolio_id: int, prices: dict[str, float] | None = None) -> dict:
    """Compute the portfolio's current accounting picture.

    ``prices`` is a ticker -> current price map; a ticker missing from it is
    marked at its entry price (flat), so valuation never breaks on a gap.
    """
    prices = prices or {}
    row = db.get_portfolio(conn, portfolio_id)
    if row is None:
        raise ValueError(f"portfolio {portfolio_id} not found")
    cash = float(row["current_cash"])

    open_rows = db.list_open_positions(conn, portfolio_id)
    open_positions = []
    positions_value = 0.0
    unrealized = 0.0
    for pos in open_rows:
        current = float(prices.get(pos["ticker"], pos["entry_price"]))
        value = portfolio.position_value(current, pos["quantity"])
        positions_value += value
        unrealized += portfolio.unrealized_pnl(pos["entry_price"], current, pos["quantity"])
        open_positions.append({
            "id": pos["id"],
            "ticker": pos["ticker"],
            "entry_date": pos["entry_date"],
            "entry_price": pos["entry_price"],
            "quantity": pos["quantity"],
            "current_price": current,
            "market_value": value,
            "unrealized_pnl": portfolio.unrealized_pnl(pos["entry_price"], current, pos["quantity"]),
        })

    closed = db.list_positions(conn, portfolio_id)
    realized = sum(
        (p["realized_pnl"] or 0.0) for p in closed if p["status"] == "closed"
    )

    return {
        "portfolio_id": portfolio_id,
        "cash": cash,
        "open_positions": open_positions,
        "positions_value": positions_value,
        "total_value": portfolio.total_value(cash, positions_value),
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "total_pnl": realized + unrealized,
    }


def record_equity_snapshot(
    conn,
    portfolio_id: int,
    captured_at: str,
    prices: dict[str, float] | None = None,
    benchmark_value: float | None = None,
) -> dict:
    """Valuate now and persist the equity-curve point; returns the snapshot."""
    snap = snapshot(conn, portfolio_id, prices)
    db.insert_portfolio_value_snapshot(
        conn, portfolio_id, captured_at,
        total_value=snap["total_value"],
        cash=snap["cash"],
        positions_value=snap["positions_value"],
        benchmark_value=benchmark_value,
    )
    return snap


def equity_curve(conn, portfolio_id: int, limit: int = 500) -> list:
    """A portfolio's recorded equity curve, newest first."""
    return db.get_portfolio_value_snapshots(conn, portfolio_id, limit=limit)


def benchmark_return(values: list[float]) -> float | None:
    """Total % return of a buy-and-hold series (first -> last)."""
    if len(values) < 2:
        return None
    first, last = values[0], values[-1]
    if not first:
        return None
    return (last - first) / first


def portfolio_return(cash_in: float, current_value: float) -> float | None:
    """% total return of the paper portfolio since seeding."""
    if not cash_in:
        return None
    return (current_value - cash_in) / cash_in


def store_benchmark_for_day(
    conn, ticker: str, trade_date: str, close: float
) -> None:
    db.set_benchmark_price(conn, ticker, trade_date, close)


def benchmark_since_inception(
    conn, ticker: str, since_date: str | None = None
) -> float | None:
    """Buy-and-hold % return of the benchmark for its stored series.

    When ``since_date`` is given the series starts at the newest stored close
    on or before that date (portfolio inception); otherwise it spans whatever
    history exists.
    """
    rows = conn.execute(
        "SELECT close, trade_date FROM benchmark_prices WHERE ticker = ? "
        "ORDER BY trade_date ASC",
        (ticker.upper(),),
    ).fetchall()
    closes = [float(r["close"]) for r in rows if r["close"] is not None]
    if since_date and rows:
        start_idx = 0
        for i, r in enumerate(rows):
            if r["trade_date"] <= since_date:
                start_idx = i
        closes = closes[start_idx:]
    return benchmark_return(closes)