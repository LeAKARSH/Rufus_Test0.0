"""Position sizing strategies (spec Section 7.2).

Each strategy maps a sizing context to a quantity of shares (fractional
allowed in the paper portfolio). Strategies are pure functions of their
context — no I/O here, so the network guard in the test suite stays clean.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from rufus.portfolio import buy_cost

log = logging.getLogger(__name__)

CONFIDENCE_WEIGHTS = {"HIGH": 0.40, "MEDIUM": 0.25, "LOW": 0.10}
DEFAULT_FIXED_AMOUNT = 5_000.0

SIZING_STRATEGIES = ("equal_weight", "fixed_amount", "confidence_weighted")


@dataclass(frozen=True)
class SizingContext:
    """Everything a strategy needs to pick a position size."""

    available_cash: float = 0.0
    price: float = 0.0
    confidence: str | None = None
    # Number of distinct tickers concurrently rated BUY but not yet held.
    buy_rated_count: int = 1
    # Total portfolio value (cash + positions) for the allocation cap.
    portfolio_value: float = 0.0
    starting_cash: float = 0.0
    max_allocation_pct: float = 100.0


def _cap_by_cash(budget: float, ctx: SizingContext) -> float:
    """Quantity affordable within ``budget``, floored to the paisa.

    Flooring (never rounding) guarantees the fill cost can never exceed the
    budget or the available cash.
    """
    if budget <= 0 or ctx.price <= 0:
        return 0.0
    if budget > ctx.available_cash:
        budget = ctx.available_cash
    raw = budget / ctx.price
    return math.floor(raw * 1000) / 1000


def _cap_by_allocation(budget: float, ctx: SizingContext) -> float:
    """Shrink the budget so the filled value stays under the allocation cap."""
    cap = ctx.max_allocation_pct / 100.0
    if cap <= 0 or ctx.portfolio_value <= 0:
        return budget
    max_budget = ctx.portfolio_value * cap
    return budget if budget <= max_budget else max_budget


def equal_weight(ctx: SizingContext) -> float:
    """Split available cash across concurrently-BUY-rated tickers.

    ``buy_rated_count`` counts tickers we are allowed to buy into (BUY rated
    and not already held), so a single call gets the whole slice.
    """
    share = max(1, ctx.buy_rated_count)
    budget = ctx.available_cash / share
    budget = _cap_by_allocation(budget, ctx)
    return _cap_by_cash(budget, ctx)


def fixed_amount(ctx: SizingContext, amount: float = DEFAULT_FIXED_AMOUNT) -> float:
    """Always size the trade to a fixed rupee amount, capped by cash/alloc."""
    budget = _cap_by_allocation(amount, ctx)
    return _cap_by_cash(budget, ctx)


def confidence_weighted(
    ctx: SizingContext, weights: dict[str, float] | None = None
) -> float:
    """Size as a fraction of starting cash by confidence grade."""
    table = weights or CONFIDENCE_WEIGHTS
    weight = table.get((ctx.confidence or "").upper(), 0.10)
    budget = _cap_by_allocation(ctx.starting_cash * weight, ctx)
    return _cap_by_cash(budget, ctx)


def make_sizer(
    strategy: str,
    fixed_amount: float = DEFAULT_FIXED_AMOUNT,
    confidence_weights: dict[str, float] | None = None,
):
    """Factory returning a ``sizer(ctx) -> quantity`` for a strategy name.

    Unknown names fall back to ``equal_weight`` so a bad config value never
    crashes the simulation.
    """
    key = (strategy or "").strip().lower()
    if key == "fixed_amount":
        amount = fixed_amount
        return lambda ctx: _fixed_with_amount(ctx, amount)
    if key == "confidence_weighted":
        weights = confidence_weights
        return lambda ctx: confidence_weighted(ctx, weights)
    if key != "equal_weight":
        log.warning(
            "unknown position sizing strategy %r; falling back to equal_weight",
            strategy,
        )
    return equal_weight


def _fixed_with_amount(ctx: SizingContext, amount: float) -> float:
    return fixed_amount(ctx, amount)


def _always_paid(budget: float, qty: float, price: float) -> bool:
    """True when a fill never overspends budget (used by tests)."""
    return buy_cost(price, qty) <= budget + 0.005