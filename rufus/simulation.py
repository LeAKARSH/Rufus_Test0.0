"""Paper-trading order fill engine (spec Section 7).

Translates a decision cycle's stored recommendations into simulated trades
against a virtual portfolio:

* ``BUY``  → open a position sized by the configured strategy (unless we
  already hold the ticker or lack the cash).
* ``SELL`` → close any open position at the current price and realize P&L.
* ``HOLD`` / ``AVOID`` → no trade; AVOID is logged as "considered and passed".

Every recommendation is marked in ``portfolio_actions`` the first time it is
processed, so re-running a decision day can never double-trade (the same
``recommendation_id`` maps to one daily recommendation row). Fills are priced
from the price snapshots the caller supplies — never from fresh network calls.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from rufus import db
from rufus.portfolio import realized_pnl
from rufus.sizing import SizingContext

log = logging.getLogger(__name__)

Sizer = Callable[[SizingContext], float]


def _cash_for(conn, portfolio_id: int) -> float:
    row = db.get_portfolio(conn, portfolio_id)
    if row is None:
        raise ValueError(f"portfolio {portfolio_id} not found")
    return float(row["current_cash"])


def _holding(conn, portfolio_id: int, ticker: str) -> bool:
    return len(db.list_open_positions(conn, portfolio_id, ticker=ticker)) > 0


def run_simulation_cycle(
    conn,
    portfolio_id: int,
    recs: Iterable,
    prices: dict[str, float],
    sizer: Sizer,
    portfolio_value: float | None = None,
    max_allocation_pct: float = 100.0,
    starting_cash: float = 0.0,
) -> list[dict]:
    """Process one decision cycle's recommendations into paper trades.

    ``recs``: the cycle's recommendation rows (e.g. from
    ``get_latest_recommendations``). ``prices``: ticker -> fill price.
    ``portfolio_value``: total value used for the allocation cap (falls back
    to cash when the caller has not valuated yet).

    Returns one result dict per recommendation, in stable ticker order.
    """
    recs = list(recs)
    buy_targets = [
        r for r in recs if (r["recommendation"] or "").upper() == "BUY"
    ]
    # Equal-weight budgeting counts distinct BUY-rated tickers we are allowed
    # to buy into (not already holding).
    buyable = [r["ticker"] for r in buy_targets if not _holding(conn, portfolio_id, r["ticker"])]
    buy_rated_count = max(1, len(buyable))
    initial_cash = _cash_for(conn, portfolio_id)
    portfolio_value = portfolio_value if portfolio_value not in (None, 0) else initial_cash

    results: list[dict] = []
    for rec in sorted(recs, key=lambda r: r["ticker"]):
        ticker = rec["ticker"]
        acted = db.get_portfolio_action(conn, rec["id"])
        if acted is not None:
            results.append({
                "ticker": ticker, "action": "SKIP",
                "reason": f"already_acted:{acted['action']}",
            })
            continue

        action = (rec["recommendation"] or "").upper()
        price = prices.get(ticker)

        if action in ("HOLD", "AVOID"):
            note = "considered and passed" if action == "AVOID" else "no action"
            db.record_portfolio_action(conn, rec["id"], "NONE", acted_at=db.utc_now_iso(), note=note)
            results.append({"ticker": ticker, "action": "NONE", "reason": note})
            continue

        if action == "BUY":
            if price is None or price <= 0:
                results.append({"ticker": ticker, "action": "SKIP", "reason": "no_price"})
                continue
            if _holding(conn, portfolio_id, ticker):
                results.append({"ticker": ticker, "action": "SKIP", "reason": "already_holding"})
                continue
            ctx = SizingContext(
                available_cash=initial_cash,
                price=price,
                confidence=rec["confidence"],
                buy_rated_count=buy_rated_count,
                portfolio_value=portfolio_value,
                starting_cash=starting_cash,
                max_allocation_pct=max_allocation_pct,
            )
            qty = sizer(ctx)
            cost = qty * price
            if qty <= 0 or cost > _cash_for(conn, portfolio_id) + 0.005:
                db.record_portfolio_action(
                    conn, rec["id"], "NONE", acted_at=db.utc_now_iso(),
                    note="insufficient cash",
                )
                results.append({"ticker": ticker, "action": "SKIP", "reason": "insufficient_cash"})
                continue
            position_id = db.open_position(
                conn, portfolio_id, ticker, price, qty, rec["id"]
            )
            db.record_portfolio_action(
                conn, rec["id"], "BUY", position_id=position_id,
                acted_at=db.utc_now_iso(),
            )
            results.append({
                "ticker": ticker, "action": "BUY", "position_id": position_id,
                "quantity": qty, "price": price, "cost": cost,
            })
            continue

        if action == "SELL":
            open_rows = db.list_open_positions(conn, portfolio_id, ticker=ticker)
            if price is None or price <= 0:
                results.append({"ticker": ticker, "action": "SKIP", "reason": "no_price"})
                continue
            if not open_rows:
                db.record_portfolio_action(
                    conn, rec["id"], "NONE", acted_at=db.utc_now_iso(),
                    note="no open position",
                )
                results.append({"ticker": ticker, "action": "SKIP", "reason": "not_holding"})
                continue
            pnl_total = 0.0
            for pos in open_rows:
                pnl = db.close_position(
                    conn, pos["id"], exit_price=price,
                    exit_recommendation_id=rec["id"],
                )
                pnl_total += pnl
            db.record_portfolio_action(
                conn, rec["id"], "SELL", acted_at=db.utc_now_iso(),
                note=f"closed {len(open_rows)} position(s)",
            )
            results.append({
                "ticker": ticker, "action": "SELL",
                "quantity": sum(p["quantity"] for p in open_rows),
                "price": price, "realized_pnl": pnl_total,
            })
            continue

        results.append({"ticker": ticker, "action": "SKIP", "reason": f"unknown_recommendation:{action}"})

    return results


def prices_from_snapshots(conn, recs: Iterable) -> dict[str, float]:
    """Latest stored close price per recommendation ticker (no network calls)."""
    prices: dict[str, float] = {}
    for rec in recs:
        rows = db.get_recent_price_snapshots(conn, rec["ticker"], limit=1)
        if rows and rows[0]["price"] is not None:
            prices[rec["ticker"]] = float(rows[0]["price"])
    return prices