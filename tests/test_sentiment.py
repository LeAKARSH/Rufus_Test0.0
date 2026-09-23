import json

import pytest

import rufus.db as db
import rufus.sentiment as sentiment
from rufus.sentiment import (
    SentimentScoreError,
    aggregate_sentiment,
    score_batch,
    score_ticker_news,
    sentiment_trend,
)


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    db.initialize_database(c)
    yield c
    c.close()


def article(title, score=None):
    a = {
        "id": title,
        "title": title,
        "description": "the value chain pivoted on a bold refi and split",
        "url": f"https://example.com/{title}",
        "source": "example.com",
        "published": "2026-09-07T14:22:08+00:00",
        "category": ["finance"],
    }
    return a if score is None else dict(a, score=score)


class StubClient:
    """Minimal stand-in for the configurable Ollama sentiment client."""

    def __init__(self, reply=None):
        self.reply = reply or {"articles": []}
        self.calls = []

    def chat_json(self, messages, validate=None, temperature=0.1):
        self.calls.append(messages)
        return self.reply


def scored_reply(*titles, base=0.3):
    return {
        "articles": [
            {"title": t, "score": base + 0.1 * i, "note": "looks ok"}
            for i, t in enumerate(titles)
        ]
    }


def test_build_messages_trims_descriptions():
    msgs = sentiment.build_messages(
        [{"title": "x", "description": "A" * 500}], trim_chars=10
    )
    user = json.loads(msgs[1]["content"])
    assert len(user["articles"][0]["description"]) == 10
    assert msgs[0]["role"] == "system"


def test_score_batch_keeps_input_order():
    articles = [article("a"), article("b"), article("c")]
    client = StubClient(scored_reply("a", "b", "c"))
    out = score_batch(client, articles, max_articles=10)
    assert [s["title"] for s in out] == ["a", "b", "c"]
    assert out[1]["score"] == pytest.approx(0.4)


def test_score_batch_drops_unscored_articles():
    articles = [article("a"), article("b"), article("c")]
    reply = {"articles": [{"title": "a", "score": 0.5, "note": ""},
                          {"title": "c", "score": -0.5, "note": ""}]}
    out = score_batch(StubClient(reply), articles, max_articles=10)
    assert [s["title"] for s in out] == ["a", "c"]


def test_score_batch_respects_max_articles():
    articles = [article(f"t{i}") for i in range(5)]
    client = StubClient(scored_reply("t0", "t1"))
    out = score_batch(client, articles, max_articles=2)
    feed = json.loads(client.calls[0][1]["content"])["articles"]
    assert [f["title"] for f in feed] == ["t0", "t1"]  # only 2 sent to the model
    assert len(out) == 2


def test_score_batch_raises_when_nothing_scored():
    articles = [article("a")]
    with pytest.raises(SentimentScoreError):
        score_batch(StubClient(scored_reply("other")), articles, max_articles=5)


def test_valid_scores_checks_range_and_types():
    good = {"articles": [{"title": "a", "score": 0.5, "note": "x"}]}
    assert sentiment._valid_scores(good, {"a"}) is True
    assert sentiment._valid_scores({"articles": [{"title": "a", "score": 1.5}]}, {"a"}) is False
    assert sentiment._valid_scores({"articles": [{"title": "a", "score": True}]}, {"a"}) is False
    assert sentiment._valid_scores({"articles": [{"title": "zzz", "score": 0.5}]}, {"a"}) is False
    assert sentiment._valid_scores({"nope": 1}, {"a"}) is False


def test_aggregate_averages_and_top_headlines():
    articles = [
        article("mild", 0.1),
        article("spicy", 0.9),
        article("gloomy", -0.8),
        article("meh", 0.0),
    ]
    scored = [{"title": a["title"], "score": a["score"]} for a in articles]
    agg = aggregate_sentiment(articles, scored)
    assert agg["articles_considered"] == 4
    assert agg["sentiment_score_avg"] == pytest.approx(0.05)
    assert [h["title"] for h in agg["top_headlines"]] == ["spicy", "gloomy", "mild", "meh"]
    assert set(agg["top_headlines"][0]) == {"title", "score", "published", "url", "source"}


def test_sentiment_trend_labels():
    assert sentiment_trend([0.5, 0.6], 0.9) == "improving"
    assert sentiment_trend([0.5, 0.5], 0.1) == "worsening"
    assert sentiment_trend([0.5, 0.5], 0.55) == "stable"
    assert sentiment_trend([], None) is None
    assert sentiment_trend([0.5], None) is None


def test_score_ticker_news_persists_row(conn):
    db.upsert_ticker(conn, "AAPL")
    client = StubClient(scored_reply("t1", base=0.4))
    agg = score_ticker_news(
        client, conn, "AAPL", '"Apple Inc." OR AAPL', "2026-09-20",
        [article("t1")], max_articles=10,
    )
    assert agg["articles_considered"] == 1
    row = db.get_news_snapshots(conn, "AAPL")[0]
    assert row["run_date"] == "2026-09-20"
    assert row["query_keyword"] == '"Apple Inc." OR AAPL'
    assert row["sentiment_score_avg"] == pytest.approx(0.4)
    assert row["sentiment_trend_7d"] is None  # no history yet
    assert json.loads(row["top_headlines_json"])[0]["title"] == "t1"


def test_score_ticker_news_same_day_overwrites(conn):
    db.upsert_ticker(conn, "AAPL")
    client = StubClient(scored_reply("t1", base=0.4))
    score_ticker_news(client, conn, "AAPL", "kw", "2026-09-20", [article("t1")], 10)
    score_ticker_news(client, conn, "AAPL", "kw", "2026-09-20", [article("t1")], 10)
    assert len(db.get_news_snapshots(conn, "AAPL")) == 1
    assert len(db.get_news_snapshots(conn, "AAPL", limit=30)) == 1


def test_score_ticker_news_empty_day_records_pull(conn):
    db.upsert_ticker(conn, "AAPL")
    agg = score_ticker_news(StubClient(), conn, "AAPL", "kw", "2026-09-20", [], 10)
    assert agg is None
    row = db.get_news_snapshots(conn, "AAPL")[0]
    assert row["articles_considered"] == 0
    assert row["sentiment_score_avg"] is None


def test_score_ticker_news_computes_trend_from_history(conn):
    db.upsert_ticker(conn, "AAPL")
    db.insert_news_snapshot(
        conn, "AAPL", run_date="2026-09-10", query_keyword="kw",
        sentiment_score_avg=0.5, articles_considered=2,
    )
    db.insert_news_snapshot(
        conn, "AAPL", run_date="2026-09-12", query_keyword="kw",
        sentiment_score_avg=0.7, articles_considered=2,
    )
    client = StubClient({"articles": [{"title": "t1", "score": 0.9, "note": ""}]})
    agg = score_ticker_news(client, conn, "AAPL", "kw", "2026-09-20", [article("t1")], 10)
    row = db.get_news_snapshots(conn, "AAPL")[0]
    assert row["sentiment_trend_7d"] == "improving"  # 0.9 vs prior avg 0.6


def test_scoring_failure_still_records_pull(conn):
    db.upsert_ticker(conn, "AAPL")
    db.insert_news_snapshot(conn, "AAPL", run_date="2026-09-12", query_keyword="kw")
    client = StubClient(scored_reply("other"))
    agg = score_ticker_news(
        client, conn, "AAPL", "kw", "2026-09-20",
        [article("t1")], max_articles=10,
    )
    assert agg is None
    rows = db.get_news_snapshots(conn, "AAPL")
    assert len(rows) == 2
    assert rows[0]["articles_considered"] == 1
    assert rows[0]["sentiment_score_avg"] is None


def test_ollama_failure_still_records_pull(conn):
    from rufus.ollama import OllamaConnectionError

    db.upsert_ticker(conn, "AAPL")

    class BrokenClient:
        def chat_json(self, *a, **k):
            raise OllamaConnectionError("refused")

    agg = score_ticker_news(
        BrokenClient(), conn, "AAPL", "kw", "2026-09-20",
        [article("t1")], max_articles=10,
    )
    assert agg is None
    row = db.get_news_snapshots(conn, "AAPL")[0]
    assert row["articles_considered"] == 1
    assert row["sentiment_score_avg"] is None