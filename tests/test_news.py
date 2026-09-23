from datetime import date, datetime

import pytest

import rufus.currents as currents_mod
import rufus.db as db
import rufus.news as news
import rufus.yahoo as yahoo_mod
from rufus.cache import TTLCache
from rufus.config import Settings
from rufus.currents import CurrentsClient
from rufus.ollama import OllamaClient
from rufus.rate_limit import RateLimiter
from rufus.yahoo import YahooClient


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    db.initialize_database(c)
    yield c
    c.close()


@pytest.fixture()
def settings():
    return Settings(
        _env_file=None,
        currentsapi_key="testkey",
        currents_max_req_per_day=100,
        yahoo_max_req_per_hour=1000,
        news_sentiment_max_articles=10,
        market_timezone="America/New_York",
    )


def article(title, published="2026-09-07 14:22:08 +0000"):
    return {
        "id": title,
        "title": title,
        "description": "the company delivered mixed quarterly results",
        "url": f"https://example.com/{title}",
        "author": "example.com",
        "language": "en",
        "category": ["finance"],
        "published": published,
    }


def ok_payload(*articles):
    return 200, {"status": "ok", "news": list(articles), "page": 1}, {}


def make_currents(conn, settings):
    return CurrentsClient(
        api_key=settings.currentsapi_key,
        limiter=RateLimiter("currents", settings.currents_max_req_per_day, "daily", conn),
    )


def make_yahoo(conn, settings):
    return YahooClient(
        limiter=RateLimiter("yahoo", settings.yahoo_max_req_per_hour, "hourly", conn),
        cache=TTLCache(),
    )


class StubOllama:
    def __init__(self, reply):
        self.reply = reply

    def chat_json(self, messages, validate=None, temperature=0.1):
        return self.reply


def seed_watchlist(conn):
    for t in ("AAPL", "MSFT"):
        db.upsert_ticker(conn, t)
        db.set_ticker_keywords(conn, t, f'"{t}" OR {t}')


def test_run_news_cycle_pulls_scores_stores(conn, settings, monkeypatch):
    seed_watchlist(conn)
    monkeypatch.setattr(
        currents_mod, "_get",
        lambda url, params, headers: ok_payload(article("headline one"), article("headline two")),
    )
    monkeypatch.setattr(
        yahoo_mod, "_fetch_calendar",
        lambda t: {"Earnings Date": [datetime(2026, 9, 25, 0, 0)]},
    )
    ollama = StubOllama(
        {"articles": [
            {"title": "headline one", "score": 0.7, "note": ""},
            {"title": "headline two", "score": -0.3, "note": ""},
        ]}
    )

    stats = news.run_news_cycle(
        conn, settings, make_currents(conn, settings), make_yahoo(conn, settings),
        ollama, today=date(2026, 9, 20),
    )
    assert len(stats) == 2
    for row in db.get_news_snapshots(conn, "AAPL"):
        assert row["sentiment_score_avg"] == pytest.approx(0.2)  # mean of 0.7, -0.3
        assert row["articles_considered"] == 2
    # Budget recorded two CurrentsAPI requests (one per ticker).
    limiter = make_currents(conn, settings).limiter
    assert limiter.used() == 2


def test_run_news_cycle_stops_at_budget(conn, settings, monkeypatch):
    seed_watchlist(conn)
    limited = Settings(
        _env_file=None,
        currentsapi_key="k", currents_max_req_per_day=1, yahoo_max_req_per_hour=1000,
        news_sentiment_max_articles=10, market_timezone="America/New_York",
    )
    monkeypatch.setattr(
        currents_mod, "_get",
        lambda url, params, headers: ok_payload(article("a")),
    )
    monkeypatch.setattr(yahoo_mod, "_fetch_calendar", lambda t: {})
    ollama = StubOllama({"articles": []})

    stats = news.run_news_cycle(
        conn, limited, make_currents(conn, limited), make_yahoo(conn, limited),
        ollama, today=date(2026, 9, 20),
    )
    # Both tickers are overdue (never pulled) and both are Tier A; AAPL
    # consumes the single available request, MSFT is never searched.
    assert len(stats) == 1
    assert len(db.get_news_snapshots(conn, "AAPL", 10)) == 1
    assert db.get_news_snapshots(conn, "MSFT", 10) == []


def test_run_news_cycle_empty_results_records_pull(conn, settings, monkeypatch):
    seed_watchlist(conn)
    monkeypatch.setattr(
        currents_mod, "_get",
        lambda url, params, headers: ok_payload(),
    )
    monkeypatch.setattr(yahoo_mod, "_fetch_calendar", lambda t: {})
    news.run_news_cycle(
        conn, settings, make_currents(conn, settings), make_yahoo(conn, settings),
        StubOllama({"articles": []}), today=date(2026, 9, 20),
    )
    rows = db.get_news_snapshots(conn, "AAPL")
    assert rows[0]["articles_considered"] == 0
    assert rows[0]["sentiment_score_avg"] is None


def test_run_news_cycle_logs_quota_retry_after(conn, settings, monkeypatch, caplog):
    seed_watchlist(conn)
    monkeypatch.setattr(
        currents_mod, "_get",
        lambda url, params, headers: (429, {"msg": "quota"}, {"Retry-After": "120"}),
    )
    monkeypatch.setattr(yahoo_mod, "_fetch_calendar", lambda t: {})
    with caplog.at_level("ERROR", logger="rufus.news"):
        news.run_news_cycle(
            conn, settings, make_currents(conn, settings), make_yahoo(conn, settings),
            StubOllama({"articles": []}), today=date(2026, 9, 20),
        )
    assert "quota-exceeded" in caplog.text
    assert "retry after 120s" in caplog.text


def test_run_news_cycle_skips_without_key(conn, settings, monkeypatch):
    seed_watchlist(conn)
    no_key = Settings(
        _env_file=None, currentsapi_key="", yahoo_max_req_per_hour=1000,
        news_sentiment_max_articles=10, market_timezone="America/New_York",
    )
    called = []
    monkeypatch.setattr(currents_mod, "_get", lambda *a, **k: called.append(1))
    out = news.run_news_cycle(
        conn, no_key, make_currents(conn, no_key), make_yahoo(conn, no_key),
        StubOllama({"articles": []}), today=date(2026, 9, 20),
    )
    assert out == [] and called == []
    assert db.get_news_snapshots(conn, "AAPL", 10) == []


def test_create_news_poll_fn_wires_full_cycle(conn, settings, monkeypatch):
    seed_watchlist(conn)
    monkeypatch.setattr(
        currents_mod, "_get",
        lambda url, params, headers: ok_payload(article("x"), article("y")),
    )
    monkeypatch.setattr(yahoo_mod, "_fetch_calendar", lambda t: {})
    monkeypatch.setattr(
        news, "OllamaClient",
        lambda **kw: StubOllama({"articles": [
            {"title": "x", "score": 0.1, "note": ""},
            {"title": "y", "score": 0.2, "note": ""},
        ]}),
    )
    poll = news.create_news_poll_fn(settings, conn)
    poll(conn)
    rows = db.get_news_snapshots(conn, "AAPL")
    assert len(rows) == 1
    assert rows[0]["articles_considered"] == 2