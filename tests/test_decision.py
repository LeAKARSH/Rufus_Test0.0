"""Decision Engine tests: prompt/validation/normalization + the stored row."""

import json

import pytest

from rufus import db
from rufus.decision import (
    _valid_recommendation,
    build_messages,
    get_recommendation,
    normalize_recommendation,
    run_decision_cycle,
)
from rufus.features import build_ticker_feature
from rufus.ollama import OllamaClient, OllamaResponseError

FEATURE = {
    "ticker": "RELIANCE.NS",
    "as_of_date": "2026-09-22",
    "technical": {"price": 1240.4, "trend_signal": "downtrend", "rsi_14": 38.6},
    "company": {"sector": "Energy", "marketCap": 20600000000000},
    "sentiment": {
        "run_date": "2026-09-22",
        "sentiment_score_avg": 0.3467,
        "top_headlines": [{"title": "Jio and T-Mobile announce 5G roaming"}],
    },
}

GOOD = {
    "recommendation": "buy",
    "confidence": "medium",
    "suggested_horizon": "6 months",
    "reasoning": "Lows near 52-week supports with improving sentiment.",
    "key_catalysts": ["5G roaming deal"],
    "key_risks": ["sector slowdown"],
    "revisit_after": "2026-12-22",
}


class StubClient:
    model = "stub-model"

    def __init__(self, reply=None, replies=None):
        self.replies = list(replies) if replies is not None else ([reply] if reply is not None else [])
        self.calls = []

    def chat_json(self, messages, validate=None, temperature=0.1):
        self.calls.append(messages)
        if not self.replies:
            raise OllamaResponseError("no stub replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        parsed = reply
        if validate is not None and not validate(parsed):
            raise OllamaResponseError("failed validation")
        return parsed


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    db.initialize_database(c)
    yield c
    c.close()


def test_valid_recommendation_gate():
    assert _valid_recommendation(GOOD)
    bad = dict(GOOD)
    bad["recommendation"] = "YOLO"
    assert not _valid_recommendation(bad)
    assert not _valid_recommendation([])
    for key, val in (("confidence", "MEH"), ("reasoning", ""), ("key_catalysts", "nope")):
        assert not _valid_recommendation({**GOOD, key: val})


def test_normalize_recommendation():
    rec = normalize_recommendation(GOOD)
    assert rec["recommendation"] == "BUY"
    assert rec["confidence"] == "MEDIUM"
    assert rec["key_catalysts"] == ["5G roaming deal"]


def test_build_messages_includes_feature_block():
    messages = build_messages(FEATURE)
    assert {m["role"] for m in messages} == {"system", "user"}
    assert "RELIANCE.NS" in messages[-1]["content"]
    assert "1240.4" in messages[-1]["content"]
    assert "Jio and T-Mobile" in messages[-1]["content"]


def test_get_recommendation_returns_normalized(conn):
    client = StubClient(GOOD)
    rec = get_recommendation(client, FEATURE, temperature=0.2)
    assert rec["recommendation"] == "BUY"
    assert rec["confidence"] == "MEDIUM"
    assert rec["revisit_after"] == "2026-12-22"


def test_run_decision_cycle_stores_row(conn):
    db.upsert_ticker(conn, "RELIANCE.NS")
    db.insert_price_snapshot(conn, "RELIANCE.NS", captured_at="2026-09-22T05:30:00+00:00",
                             price=1240.4, trend_signal="downtrend", rsi_14=38.6)
    client = StubClient(GOOD)
    stats = run_decision_cycle(
        conn, settings=None, client=client, today=__import__("datetime").date(2026, 9, 22)
    )
    assert stats == [{"ticker": "RELIANCE.NS", "recommendation": "BUY"}]
    row = db.get_latest_recommendation(conn, "RELIANCE.NS")
    assert row["run_date"] == "2026-09-22"
    assert row["recommendation"] == "BUY"
    assert row["confidence"] == "MEDIUM"
    assert json.loads(row["key_catalysts_json"]) == ["5G roaming deal"]
    feature = json.loads(row["input_data_json"])
    assert feature["technical"]["price"] == 1240.4


def test_run_decision_cycle_skips_ticker_without_price(conn):
    db.upsert_ticker(conn, "RELIANCE.NS")
    client = StubClient(GOOD)
    stats = run_decision_cycle(
        conn, settings=None, client=client, today=__import__("datetime").date(2026, 9, 22)
    )
    assert stats == []
    assert db.get_latest_recommendation(conn, "RELIANCE.NS") is None


def test_run_decision_cycle_llm_failure_is_skipped(conn):
    db.upsert_ticker(conn, "RELIANCE.NS")
    db.insert_price_snapshot(conn, "RELIANCE.NS", captured_at="2026-09-22T05:30:00+00:00",
                             price=1240.4)
    client = StubClient(OllamaResponseError("model down"))
    stats = run_decision_cycle(
        conn, settings=None, client=client, today=__import__("datetime").date(2026, 9, 22)
    )
    assert len(stats) == 1
    assert stats[0]["recommendation"] is None
    assert "error" in stats[0]
    assert db.get_latest_recommendation(conn, "RELIANCE.NS") is None


def test_run_decision_cycle_upserts_same_day(conn):
    db.upsert_ticker(conn, "RELIANCE.NS")
    db.insert_price_snapshot(conn, "RELIANCE.NS", captured_at="2026-09-22T05:30:00+00:00",
                             price=1240.4)
    client = StubClient(replies=[GOOD, {**GOOD, "recommendation": "hold"}])
    for _ in range(2):
        run_decision_cycle(conn, settings=None, client=client,
                           today=__import__("datetime").date(2026, 9, 22))
    rows = db.get_recommendations(conn, "RELIANCE.NS")
    assert len(rows) == 1
    assert rows[0]["recommendation"] == "HOLD"


def test_feature_builder_integration_used_by_cycle(conn):
    """The cycle builds its feature from live-style rows (happy path end-to-end)."""
    from datetime import date

    db.upsert_ticker(conn, "RELIANCE.NS")
    db.insert_price_snapshot(
        conn, "RELIANCE.NS", captured_at="2026-09-22T05:30:00+00:00",
        price=1240.4, sma_50=1300.0, trend_signal="downtrend",
        data_json=json.dumps({"info": {"sector": "Energy"}, "indicators": {}}),
    )
    db.insert_news_snapshot(conn, "RELIANCE.NS", run_date="2026-09-22",
                            sentiment_score_avg=0.3, articles_considered=5,
                            top_headlines_json=json.dumps([
                                {"title": "Heard on the Street"}
                            ]))

    captured = {}

    class CapturingClient:
        model = "capture-model"

        def chat_json(self, messages, validate=None, temperature=0.1):
            from rufus.ollama import extract_json

            captured["face"] = extract_json(messages[-1]["content"])
            return GOOD

    run_decision_cycle(conn, settings=None, client=CapturingClient(), today=date(2026, 9, 22))
    face = captured["face"]
    assert face["technical"]["price"] == 1240.4
    assert face["sentiment"]["sentiment_score_avg"] == pytest.approx(0.3)
    assert face["sentiment"]["top_headlines"][0]["title"] == "Heard on the Street"