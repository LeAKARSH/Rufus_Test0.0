"""Feature-builder tests: latest snapshots -> compact LLM input block."""

import json

import pytest

from rufus import db
from rufus.features import build_ticker_feature


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    db.initialize_database(c)
    yield c
    c.close()


def _price_row(conn, ticker="RELIANCE.NS", **extra):
    db.upsert_ticker(conn, ticker)
    db.insert_price_snapshot(
        conn,
        ticker,
        captured_at="2026-09-22T05:30:00+00:00",
        price=1240.4,
        sma_50=1300.0,
        sma_200=1250.0,
        trend_signal="downtrend",
        rsi_14=38.6,
        pe_ratio=22.5,
        dividend_yield=0.3,
        high_52w=1608.85,
        low_52w=1235.0,
        volatility_90d=0.0116,
        position_vs_52w_range_pct=3.9,
        data_json=json.dumps(
            {
                "info": {
                    "longName": "Reliance Industries Limited",
                    "sector": "Energy",
                    "industry": "Oil & Gas Refining & Marketing",
                    "marketCap": 20600000000000,
                    "forwardPE": 19.2,
                    "earningsGrowth": 0.14,
                    "revenueGrowth": 0.05,
                    "fiftyTwoWeekHigh": 1608.85,
                    "fiftyTwoWeekLow": 1235.0,
                },
                "indicators": {"trend_3m": "falling", "trend_6m": "falling", "trend_12m": "mixed"},
            }
        ),
        **extra,
    )
    return db.get_recent_price_snapshots(conn, ticker, limit=1)[0]


def _news_row(conn, ticker="RELIANCE.NS"):
    db.insert_news_snapshot(
        conn,
        ticker,
        run_date="2026-09-22",
        query_keyword='"Reliance Industries" OR RELIANCE',
        sentiment_score_avg=0.3467,
        sentiment_trend_7d=None,
        articles_considered=15,
        top_headlines_json=json.dumps(
            [
                {"title": "Jio, T-Mobile Achieve World's First 5G Standalone Roaming",
                 "published": "2026-09-22T05:00:00+00:00", "url": "http://x/1"},
                {"title": "Start of Day Message",
                 "published": None, "url": "http://x/2"},
            ]
        ),
    )
    return db.get_news_snapshots(conn, ticker, limit=1)[0]


def test_feature_has_technical_summary(conn):
    feature = build_ticker_feature(_price_row(conn))
    assert feature["ticker"] == "RELIANCE.NS"
    assert feature["as_of_date"] == "2026-09-22"
    tech = feature["technical"]
    assert tech["price"] == 1240.4
    assert tech["sma_50"] == 1300.0
    assert tech["trend_signal"] == "downtrend"
    assert tech["rsi_14"] == 38.6
    assert tech["position_vs_52w_range_pct"] == 3.9


def test_feature_company_derived_from_data_json(conn):
    company = build_ticker_feature(_price_row(conn))["company"]
    assert company["sector"] == "Energy"
    assert company["marketCap"] == 20600000000000
    assert company["forwardPE"] == 19.2
    assert company["fifty_two_week_high"] == 1608.85
    assert company["multi_horizon_trends"]["trend_3m"] == "falling"


def test_feature_sentiment_optional_when_no_news(conn):
    feature = build_ticker_feature(_price_row(conn))
    assert feature["sentiment"] is None


def test_feature_sentiment_includes_headline_text(conn):
    sentiment = build_ticker_feature(_price_row(conn), _news_row(conn))["sentiment"]
    assert sentiment["sentiment_score_avg"] == pytest.approx(0.3467)
    assert sentiment["articles_considered"] == 15
    assert sentiment["top_headlines"][0]["title"].startswith("Jio")
    assert sentiment["top_headlines"][1]["title"] == "Start of Day Message"


def test_feature_tolerates_missing_optional_columns(conn):
    _price_row(conn)
    # Pretend stored fundamentals are absent (bare older snapshot, day before).
    db.insert_price_snapshot(
        conn, "RELIANCE.NS", captured_at="2026-09-21T05:30:00+00:00", price=1200.0
    )
    rows = db.get_recent_price_snapshots(conn, "RELIANCE.NS", limit=2)
    bare = rows[1]  # the 09-21 row lacks data_json and most columns
    feature = build_ticker_feature(bare)
    assert feature["technical"]["price"] == 1200.0
    assert feature["company"] == {}