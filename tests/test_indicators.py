import datetime

import numpy as np
import pandas as pd
import pytest

from rufus.indicators import (
    _macd,
    _position_vs_52w,
    _rsi,
    _sma,
    _trend_signal,
    _trends_pct,
    compute_indicators,
)

rng = np.random.default_rng(42)


def make_history(closes):
    idx = pd.DatetimeIndex(
        [datetime.datetime(2025, 1, 1) + datetime.timedelta(days=i) for i in range(len(closes))],
        tz="UTC",
    )
    return pd.DataFrame({"Close": closes, "Volume": [1_000_000] * len(closes)}, index=idx)


def days_up(start=100.0, n=300, step=0.5):
    return [start + i * step for i in range(n)]


def days_down(start=300.0, n=300, step=0.5):
    return [start - i * step for i in range(n)]


def walk(n=300):
    prices = np.cumprod(1 + rng.normal(0.0005, 0.01, n)) * 100
    return [float(p) for p in prices]


def test_empty_history_returns_empty():
    assert compute_indicators(pd.DataFrame()) == {}
    assert compute_indicators(pd.DataFrame({"Close": []})) == {}


def test_insufficient_data_is_graceful():
    ind = compute_indicators(make_history([100.0, 101.0]))
    assert ind["price"] == 101.0
    assert ind["sma_200"] is None
    assert ind["sma_50"] is None
    assert ind["trend_signal"] is None
    assert ind["rsi_14"] is None
    assert "macd" in ind["indicators"]


def test_uptrend_series():
    ind = compute_indicators(make_history(days_up()))
    assert ind["trend_signal"] == "uptrend"
    assert ind["sma_50"] > ind["sma_200"]  # rising series: fast MA above slow
    assert ind["price"] > ind["sma_50"]
    assert ind["position_vs_52w_range_pct"] == 100.0  # at 52-week high


def test_downtrend_series():
    ind = compute_indicators(make_history(days_down()))
    assert ind["trend_signal"] == "downtrend"
    assert ind["price"] < ind["sma_50"]
    assert ind["rsi_14"] is not None and 0 <= ind["rsi_14"] <= 100


def test_golden_cross_recent():
    # Flat for >200 sessions, then a sharp 4-session pop pushes SMA50 above
    # SMA200 inside the recent-cross window.
    closes = [100.0] * 250 + [120.0] * 4
    ind = compute_indicators(make_history(closes))
    assert ind["trend_signal"] == "golden_cross_recent"


def test_death_cross_recent():
    # Flat for >200 sessions, then a sharp 4-session drop pushes SMA50 below
    # SMA200 inside the recent-cross window.
    closes = [400.0] * 250 + [320.0] * 4
    ind = compute_indicators(make_history(closes))
    assert ind["trend_signal"] == "death_cross_recent"


def test_rsi_bounds_and_extremes():
    h = make_history(days_up())
    assert 50 <= _rsi(h["Close"]) <= 100
    h2 = make_history(days_down())
    assert 0 <= _rsi(h2["Close"]) <= 50
    monotonic_up = make_history([100.0 + i for i in range(100)])
    assert _rsi(monotonic_up["Close"]) == pytest.approx(100, abs=0.001)


def test_volatility_and_returns():
    cl = walk()
    ind = compute_indicators(make_history(cl))
    assert ind["volatility_90d"] is not None and ind["volatility_90d"] > 0
    trends = ind["indicators"]["trends_pct"]
    assert set(trends) == {"3m", "6m", "12m"}
    assert all(v is not None for v in trends.values())
    for label, days in (("3m", 63), ("6m", 126), ("12m", 252)):
        expected = (cl[-1] / cl[-1 - days] - 1) * 100
        assert trends[label] == pytest.approx(expected, abs=0.01)


def test_macd_shapes():
    macd = _macd(pd.Series(days_up()))
    assert set(macd) == {"line", "signal", "histogram"}
    assert macd["line"] is not None and macd["histogram"] is not None


def test_trend_signal_none_with_short_history():
    assert _trend_signal(make_history([100.0] * 10)["Close"]) is None


def test_sma_short_window():
    assert _sma(pd.Series([1.0, 2.0, 3.0]), 2) == 2.5


def test_position_vs_52w_edge_cases():
    assert _position_vs_52w(150, 200, 100) == 50.0
    assert _position_vs_52w(150, 150, 150) is None
    assert _position_vs_52w(None, 200, 100) is None
    assert _trends_pct(pd.Series([1.0])) == {"3m": None, "6m": None, "12m": None}