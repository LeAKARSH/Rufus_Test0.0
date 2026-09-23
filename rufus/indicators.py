"""Technical indicators tuned for long-horizon holdings (spec Section 4).

Indicators favour medium/long timeframes over day-trading noise: 50/200-day
SMA cross regime, RSI-14 (to flag extremes), MACD (secondary confirmation),
90-day volatility, 52-week range positioning and 3/6/12-month return trends.

``compute_indicators`` returns a flat dict whose keys map directly onto the
``price_snapshots`` columns, plus a nested ``indicators`` section (MACD and
multi-horizon trends) that is persisted inside ``data_json``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_CROSS_WINDOW = 5  # sessions within which a recent cross is "recent"
_TREND_WINDOWS = {"3m": 63, "6m": 126, "12m": 252}  # ~trading days


def _sma(series: pd.Series, window: int) -> float | None:
    value = series.rolling(window).mean().iloc[-1]
    return _maybe(value)


def _rsi(close: pd.Series, window: int = 14) -> float | None:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
    # Loss-less window -> RSI 100; gain-less window -> RSI 0.
    rsi = rsi.mask(avg_loss == 0, 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    return _maybe(rsi.iloc[-1])


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    line = _ema(close, fast) - _ema(close, slow)
    signal_line = line.ewm(span=signal, adjust=False).mean()
    hist = line - signal_line
    return {
        "line": _maybe(line.iloc[-1]),
        "signal": _maybe(signal_line.iloc[-1]),
        "histogram": _maybe(hist.iloc[-1]),
    }


def _trend_signal(close: pd.Series, fast: int = 50, slow: int = 200) -> str | None:
    """Categorise the 50/200 SMA regime, flagging recent crosses."""
    diff = close.rolling(fast).mean() - close.rolling(slow).mean()
    valid = diff.dropna()
    if valid.empty:
        return None
    last = float(valid.iloc[-1])
    recent = valid.tail(_CROSS_WINDOW + 1)
    signs = np.sign(recent.to_numpy(dtype=float))
    crosses = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0

    if last > 0:
        return "golden_cross_recent" if crosses else "uptrend"
    if last < 0:
        return "death_cross_recent" if crosses else "downtrend"
    return None


def _volatility_90d(close: pd.Series) -> float | None:
    returns = close.pct_change().dropna().tail(90)
    if returns.empty:
        return None
    return _maybe(returns.std())


def _position_vs_52w(price: float, high: float, low: float) -> float | None:
    if price is None or high is None or low is None or high <= low:
        return None
    return round(float((price - low) / (high - low) * 100), 2)


def _trends_pct(close: pd.Series) -> dict[str, float | None]:
    if close.empty:
        return {k: None for k in _TREND_WINDOWS}
    price_now = float(close.iloc[-1])
    out: dict[str, float | None] = {}
    for label, days in _TREND_WINDOWS.items():
        if len(close) <= days:
            out[label] = None
            continue
        price_then = float(close.iloc[-1 - days])
        out[label] = (
            round((price_now / price_then - 1) * 100, 2) if price_then else None
        )
    return out


def _maybe(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(value) or np.isinf(value)) else round(value, 4)


def compute_indicators(history: pd.DataFrame) -> dict[str, Any]:
    """Compute the long-horizon indicator set from a daily OHLCV frame."""
    if history.empty or "Close" not in history.columns:
        return {}
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if close.empty:
        return {}

    price = _maybe(close.iloc[-1])
    sma_50 = _sma(close, 50)
    sma_200 = _sma(close, 200)

    high_52w = float(close.max()) if not close.empty else None
    low_52w = float(close.min()) if not close.empty else None

    return {
        "price": price,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "trend_signal": _trend_signal(close),
        "rsi_14": _rsi(close),
        "volatility_90d": _volatility_90d(close),
        "high_52w": high_52w,
        "low_52w": low_52w,
        "position_vs_52w_range_pct": _position_vs_52w(price, high_52w, low_52w),
        "indicators": {
            "macd": _macd(close),
            "trends_pct": _trends_pct(close),
        },
    }