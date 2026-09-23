"""Pure portfolio accounting helpers (spec Section 7).

This module deliberately contains **no I/O** (no sqlite, network, or file
access): it exists so the paper-trading math can be unit-tested in isolation
and so the architectural guard in the test suite can prove the simulation
core never touches the network or rate limiters.
"""

from __future__ import annotations

from typing import Any


def buy_cost(price: float, quantity: float) -> float:
    return price * quantity


def sell_proceeds(price: float, quantity: float) -> float:
    return price * quantity


def realized_pnl(entry_price: float, exit_price: float, quantity: float) -> float:
    return (exit_price - entry_price) * quantity


def unrealized_pnl(entry_price: float, current_price: float, quantity: float) -> float:
    return (current_price - entry_price) * quantity


def position_value(price: float, quantity: float) -> float:
    return price * quantity


def total_value(cash: float, positions_value: float) -> float:
    return cash + positions_value


def snapshot_report(portfolio: dict[str, Any]) -> dict[str, Any]:
    """Roll cash + open positions into a portfolio-style summary dict."""
    open = portfolio.get("open", [])
    positions_value = sum(
        p.get("current_price", 0.0) * p.get("quantity", 0.0) for p in open
    )
    realized = sum(p.get("realized_pnl", 0.0) for p in portfolio.get("closed", []))
    return {
        "cash": portfolio.get("cash", 0.0),
        "positions_value": positions_value,
        "total_value": total_value(portfolio.get("cash", 0.0), positions_value),
        "realized_pnl": realized,
        "unrealized_pnl": sum(
            unrealized_pnl(
                p.get("entry_price", 0.0),
                p.get("current_price", 0.0),
                p.get("quantity", 0.0),
            )
            for p in open
        ),
        "open_positions": len(open),
        "closed_positions": len(portfolio.get("closed", [])),
    }