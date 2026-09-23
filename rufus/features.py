"""Feature builder (spec Sections 4 and 5: data digest -> LLM input).

Turns the latest persisted rows (price snapshot + news/sentiment snapshot)
into the compact per-ticker block that the Decision Engine embeds in its
prompt, so the LLM reasons only over data the app actually collected.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rufus.db import _PRICE_SNAPSHOT_COLUMNS

log = logging.getLogger(__name__)

# Fundamentals surfaced from the snapshot's ``data_json.info`` block.
_FUNDAMENTAL_KEYS = (
    "sector",
    "industry",
    "marketCap",
    "forwardPE",
    "earningsGrowth",
    "revenueGrowth",
)


def _parse_json(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        loaded = json.loads(value)
    except (ValueError, TypeError):
        log.warning("failed to parse stored JSON data")
        return None
    return loaded if isinstance(loaded, dict) else None


def build_ticker_feature(price_row, news_row=None) -> dict[str, Any]:
    """Compact digest for one ticker from its latest stored snapshots.

    ``price_row`` is a ``price_snapshots`` row; ``news_row`` an optional
    ``news_snapshots`` row. Values are ``None``-tolerant so a ticker with no
    scoring yet still produces a valid feature block.
    """
    data = _parse_json(price_row["data_json"]) or {}
    info = data.get("info") or {}

    technical = {
        key: price_row[key]
        for key in _PRICE_SNAPSHOT_COLUMNS
        if key in price_row.keys()
    }
    technical.pop("captured_at", None)
    technical.pop("id", None)

    company: dict[str, Any] = {}
    for key in _FUNDAMENTAL_KEYS:
        if info.get(key) is not None:
            company[key] = info.get(key)
    if info.get("fiftyTwoWeekHigh") is not None:
        company["fifty_two_week_high"] = info["fiftyTwoWeekHigh"]
    if info.get("fiftyTwoWeekLow") is not None:
        company["fifty_two_week_low"] = info["fiftyTwoWeekLow"]

    horizon_trends = data.get("indicators")
    if isinstance(horizon_trends, dict):
        company["multi_horizon_trends"] = {
            k: v for k, v in horizon_trends.items() if v is not None
        }

    sentiment = None
    if news_row is not None:
        headlines: list[Any] = []
        if news_row["top_headlines_json"]:
            try:
                parsed = json.loads(news_row["top_headlines_json"])
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, list):
                headlines = parsed
        sentiment = {
            "run_date": news_row["run_date"],
            "sentiment_score_avg": news_row["sentiment_score_avg"],
            "sentiment_trend_7d": news_row["sentiment_trend_7d"],
            "articles_considered": news_row["articles_considered"],
            "top_headlines": [
                {
                    "title": h.get("title"),
                    "published": h.get("published"),
                    "url": h.get("url"),
                }
                for h in (headlines or [])
                if isinstance(h, dict)
            ],
        }

    return {
        "ticker": price_row["ticker"],
        "as_of_date": (price_row["captured_at"] or "")[:10],
        "technical": technical,
        "company": company,
        "sentiment": sentiment,
    }