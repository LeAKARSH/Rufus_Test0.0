import pytest

from rufus import sizing
from rufus.sizing import (
    SizingContext,
    confidence_weighted,
    equal_weight,
    fixed_amount,
    make_sizer,
)


def _ctx(**kw):
    base = dict(
        available_cash=10_000.0, price=100.0, confidence="MEDIUM",
        buy_rated_count=1, portfolio_value=10_000.0,
        starting_cash=10_000.0, max_allocation_pct=100.0,
    )
    base.update(kw)
    return SizingContext(**base)


def test_equal_weight_splits_across_buy_rated():
    qty = equal_weight(_ctx(available_cash=10_000.0, price=100.0, buy_rated_count=2))
    # Budget 5000 -> 50 shares.
    assert qty == 50.0
    assert sizing._always_paid(5_000.0, qty, 100.0)


def test_equal_weight_single_gets_full_slice():
    assert equal_weight(_ctx(price=200.0)) == 50.0


def test_equal_weight_respects_cash_budget():
    qty = equal_weight(_ctx(available_cash=1_000.0, price=300.0))
    assert 300.0 * qty <= 1_000.0 + 0.005
    assert qty > 0


def test_allocation_cap_shrinks_position():
    qty = equal_weight(_ctx(available_cash=10_000.0, price=100.0, max_allocation_pct=25.0))
    # Cap budget = 25% of 10000 = 2500 -> 25 shares.
    assert qty == 25.0


def test_fixed_amount_uses_configured_value():
    qty = fixed_amount(_ctx(), amount=3_000.0)
    assert qty == 30.0


def test_fixed_amount_capped_by_cash():
    qty = fixed_amount(_ctx(available_cash=500.0), amount=3_000.0)
    assert 500.0 >= 100.0 * qty


def test_confidence_weighted_uses_starting_cash_fraction():
    assert confidence_weighted(_ctx(confidence="HIGH")) == 4_000.0 / 100.0
    assert confidence_weighted(_ctx(confidence="MEDIUM")) == 2_500.0 / 100.0
    assert confidence_weighted(_ctx(confidence="LOW")) == 1_000.0 / 100.0
    assert confidence_weighted(_ctx(confidence="LOW")) == 10.0


def test_confidence_unknown_falls_back_to_lowest():
    assert confidence_weighted(_ctx(confidence="WHAT")) == 10.0
    assert confidence_weighted(_ctx(confidence=None)) == 10.0


def test_zero_price_or_cash_yields_zero():
    assert equal_weight(_ctx(price=0.0)) == 0.0
    assert equal_weight(_ctx(available_cash=0.0)) == 0.0


def test_floor_never_overspends_budget():
    # 10000 / 2105 does not divide evenly; the fill must not cost more than cash.
    qty = equal_weight(_ctx(available_cash=10_000.0, price=2105.0))
    assert 2105.0 * qty <= 10_000.0 + 0.005
    assert qty == 4.75


def test_make_sizer_factory_selects_strategy():
    sizer = make_sizer("confidence_weighted")
    assert sizer(_ctx(confidence="HIGH")) == 40.0
    fixed = make_sizer("fixed_amount", fixed_amount=2_000.0)
    assert fixed(_ctx()) == 20.0


def test_make_sizer_unknown_name_falls_back(caplog):
    sizer = make_sizer("momentum_vibes")
    assert sizer(_ctx()) == equal_weight(_ctx())
    assert "falling back to equal_weight" in caplog.text


def test_strategy_names_are_exported():
    assert sizing.SIZING_STRATEGIES == (
        "equal_weight", "fixed_amount", "confidence_weighted"
    )