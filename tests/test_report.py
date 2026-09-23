import pytest

from rufus import db, scorecard
from rufus.config import Settings
from rufus.paper import run_paper_cycle
from rufus.report import (
    build_report,
    portfolio_view,
    ticker_detail,
    trade_log,
    watchlist_table,
)


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
        portfolio_name="default",
        starting_cash=10_000.0,
        position_sizing_strategy="equal_weight",
        benchmark_ticker="^NSEI",
        max_allocation_pct=100.0,
    )


def _seed_rec(conn, ticker, recommendation, run_date="2026-09-22", confidence="MEDIUM",
              reasoning="test reasoning", price=None):
    db.upsert_ticker(conn, ticker)
    db.insert_recommendation(
        conn, ticker, run_date=run_date, recommendation=recommendation,
        confidence=confidence, reasoning=reasoning,
        model="fake-model", key_catalysts=["catalyst-a"], key_risks=["risk-a"],
    )
    if price is not None:
        db.insert_price_snapshot(
            conn, ticker, captured_at=f"{run_date}T10:00:00+00:00", price=price,
            sma_50=price * 0.9, sma_200=price * 0.8,
        )


def _seed_news(conn, ticker, run_date, score, trend="stable", headlines=None):
    db.upsert_ticker(conn, ticker)
    db.insert_news_snapshot(
        conn, ticker, run_date=run_date, sentiment_score_avg=score,
        sentiment_trend_7d=trend, articles_considered=2,
        top_headlines_json=__import__("json").dumps(headlines or [
            {"title": "headline-a", "published": "2026-09-20T05:00:00Z",
             "url": "https://example.com/a", "source": "X"}
        ]),
    )


# --------------------------------------------------------------------------
# watchlist_table
# --------------------------------------------------------------------------

def test_watchlist_table_reports_latest_state(conn, settings):
    _seed_rec(conn, "RELIANCE.NS", "BUY", price=100.0)
    _seed_news(conn, "RELIANCE.NS", "2026-09-22", score=0.35, trend="improving")
    rows = watchlist_table(conn)
    assert len(rows) == 1
    r = rows[0]
    assert r["ticker"] == "RELIANCE.NS"
    assert r["recommendation"] == "BUY"
    assert r["confidence"] == "MEDIUM"
    assert r["decision_date"] == "2026-09-22"
    assert r["model"] == "fake-model"
    assert r["sentiment_score"] == 0.35
    assert r["sentiment_trend"] == "improving"
    assert r["price"] == 100.0
    assert r["price_date"] == "2026-09-22"
    assert r["held"] is False


def test_watchlist_table_marks_held_position(conn, settings):
    _seed_rec(conn, "TCS.NS", "BUY", price=200.0)
    run_paper_cycle(conn, settings)
    pid = db.ensure_default_portfolio(conn, settings.portfolio_name)["id"]
    rows = watchlist_table(conn, portfolio_id=pid)
    assert rows[0]["held"] is True


def test_watchlist_table_empty_db(conn):
    assert watchlist_table(conn) == []


# --------------------------------------------------------------------------
# ticker_detail
# --------------------------------------------------------------------------

def test_ticker_detail_series_and_latest(conn):
    _seed_rec(conn, "TCS.NS", "BUY", run_date="2026-09-20", price=100.0)
    _seed_rec(conn, "TCS.NS", "HOLD", run_date="2026-09-21", price=110.0, reasoning="second-day")
    _seed_news(conn, "TCS.NS", "2026-09-21", score=0.2)
    _seed_news(conn, "TCS.NS", "2026-09-20", score=0.1)

    detail = ticker_detail(conn, "TCS.NS")

    assert detail["ticker"] == "TCS.NS"
    assert len(detail["price_series"]) == 2
    assert detail["price_series"][0]["captured_at"] < detail["price_series"][1]["captured_at"]
    assert detail["latest_price"]["price"] == 110.0
    assert detail["latest_price"]["sma_50"] == 99.0

    assert len(detail["sentiment_series"]) == 2
    assert detail["sentiment_series"][-1]["score"] == 0.2

    lr = detail["latest_recommendation"]
    assert lr["recommendation"] == "HOLD"
    assert lr["run_date"] == "2026-09-21"
    assert lr["reasoning"] == "second-day"
    assert lr["key_catalysts"] == ["catalyst-a"]
    assert lr["key_risks"] == ["risk-a"]
    assert lr["model"] == "fake-model"

    assert len(detail["recommendation_history"]) == 2
    assert detail["recommendation_history"][0]["run_date"] == "2026-09-21"
    assert len(detail["headlines"]) == 1
    assert detail["headlines"][0]["title"] == "headline-a"


def test_ticker_detail_handles_bad_json(conn):
    db.upsert_ticker(conn, "T.NS")
    db.insert_news_snapshot(
        conn, "T.NS", run_date="2026-09-22",
        top_headlines_json="not-json",
    )
    detail = ticker_detail(conn, "T.NS")
    assert detail["headlines"] == []
    assert detail["latest_recommendation"] is None


def test_ticker_detail_empty_db(conn):
    detail = ticker_detail(conn, "NOPE.NS")
    assert detail["price_series"] == []
    assert detail["latest_price"] is None
    assert detail["recommendation_history"] == []
    assert detail["headlines"] == []


# --------------------------------------------------------------------------
# portfolio_view
# --------------------------------------------------------------------------

def test_portfolio_view_reflects_position_and_curve(conn, settings):
    _seed_rec(conn, "TCS.NS", "BUY", price=200.0)
    out = run_paper_cycle(conn, settings, captured_at="2026-09-22T10:00:00+00:00")
    db.insert_price_snapshot(
        conn, "TCS.NS", captured_at="2026-09-22T12:00:00+00:00", price=220.0,
    )
    view = portfolio_view(conn, settings)

    assert view["portfolio"] == "default"
    assert view["starting_cash"] == 10_000.0
    assert view["total_value"] == pytest.approx(11_000.0)
    assert view["positions_value"] == pytest.approx(11_000.0)
    assert view["portfolio_return"] == pytest.approx(0.10)
    assert view["benchmark_return"] is None
    assert view["benchmark_ticker"] == "^NSEI"
    assert len(view["open_positions"]) == 1
    assert view["open_positions"][0]["ticker"] == "TCS.NS"
    assert len(view["equity_curve"]) == 1
    assert view["equity_curve"][0]["total_value"] == pytest.approx(10_000.0)


def test_portfolio_view_equity_curve_ascending(conn, settings):
    _seed_rec(conn, "RELIANCE.NS", "BUY", price=100.0)
    run_paper_cycle(conn, settings, captured_at="2026-09-22T10:00:00+00:00")
    db.insert_price_snapshot(conn, "RELIANCE.NS", captured_at="2026-09-22T12:00:00+00:00", price=110.0)
    view = portfolio_view(conn, settings)
    times = [p["captured_at"] for p in view["equity_curve"]]
    assert times == sorted(times)


# --------------------------------------------------------------------------
# trade_log + scorecard
# --------------------------------------------------------------------------

def _buy_sell_cycle(conn, settings, ticker="TCS.NS", entry="2026-09-20", exit_="2026-09-23", exit_price=130.0):
    _seed_rec(conn, ticker, "BUY", run_date=entry, price=100.0)
    run_paper_cycle(conn, settings, captured_at=f"{entry}T10:00:00+00:00")
    _seed_rec(conn, ticker, "SELL", run_date=exit_, price=0.0, reasoning="exit reasoning")
    db.insert_price_snapshot(conn, ticker, captured_at=f"{exit_}T10:00:00+00:00", price=exit_price)
    run_paper_cycle(conn, settings, captured_at=f"{exit_}T10:00:00+00:00")


def test_trade_log_links_reasoning(conn, settings):
    _buy_sell_cycle(conn, settings)
    trades = trade_log(conn, settings)
    assert len(trades) == 1
    t = trades[0]
    assert t["ticker"] == "TCS.NS"
    assert t["status"] == "closed"
    assert t["realized_pnl"] == pytest.approx(30.0 * 100.0)
    assert t["entry_recommendation"]["recommendation"] == "BUY"
    assert t["entry_recommendation"]["reasoning"] == "test reasoning"
    assert t["exit_recommendation"]["recommendation"] == "SELL"
    assert t["exit_recommendation"]["reasoning"] == "exit reasoning"


def test_scorecard_direction_hit_rate(conn, settings):
    # Two simultaneous BUYs split the 10k budget (5k each -> 50 shares each).
    _seed_rec(conn, "TCS.NS", "BUY", run_date="2026-09-20", price=100.0)
    _seed_rec(conn, "RELIANCE.NS", "BUY", run_date="2026-09-20", price=100.0)
    run_paper_cycle(conn, settings, captured_at="2026-09-20T10:00:00+00:00")
    _seed_rec(conn, "TCS.NS", "SELL", run_date="2026-09-23", price=0.0, reasoning="exit reasoning")
    _seed_rec(conn, "RELIANCE.NS", "SELL", run_date="2026-09-23", price=0.0, reasoning="exit reasoning")
    db.insert_price_snapshot(conn, "TCS.NS", captured_at="2026-09-23T10:00:00+00:00", price=130.0)
    db.insert_price_snapshot(conn, "RELIANCE.NS", captured_at="2026-09-23T10:00:00+00:00", price=90.0)
    run_paper_cycle(conn, settings, captured_at="2026-09-23T10:00:00+00:00")
    pid = db.ensure_default_portfolio(conn, settings.portfolio_name)["id"]
    sc = scorecard.compute(conn, pid)
    assert sc["decided_closed"] == 2
    assert sc["hits"] == 1
    assert sc["hit_rate"] == 0.5
    assert sc["total_realized_pnl"] == pytest.approx(50 * 30 + 50 * (-10))
    assert sc["avg_realized_pnl"] == pytest.approx(1000.0 / 2)
    bye = sc["by_direction"]["BUY"]
    assert bye["hits"] == 1
    assert bye["hit_rate"] == 0.5


def test_scorecard_with_only_open_position(conn, settings):
    _seed_rec(conn, "TCS.NS", "BUY", price=200.0)
    run_paper_cycle(conn, settings)
    pid = db.ensure_default_portfolio(conn, settings.portfolio_name)["id"]
    sc = scorecard.compute(conn, pid)
    assert sc["open_positions"] == 1
    assert sc["decided_closed"] == 0
    assert sc["hit_rate"] is None
    assert sc["avg_realized_pnl"] is None


def test_scorecard_empty(conn):
    assert scorecard.compute(conn, 0)["decided_closed"] == 0


# --------------------------------------------------------------------------
# build_report
# --------------------------------------------------------------------------

def test_build_report_aggregates_all_views(conn, settings):
    _seed_rec(conn, "TCS.NS", "BUY", price=100.0)
    _seed_news(conn, "TCS.NS", "2026-09-22", score=0.1)
    run_paper_cycle(conn, settings)
    report = build_report(conn, settings)
    assert set(report) >= {
        "generated_at", "portfolio_name", "benchmark_ticker",
        "watchlist", "portfolio", "trades", "scorecard",
    }
    assert report["benchmark_ticker"] == "^NSEI"
    assert len(report["watchlist"]) == 1
    assert report["watchlist"][0]["held"] is True
    assert report["portfolio"]["open_positions"][0]["ticker"] == "TCS.NS"
    assert len(report["trades"]) == 1
    assert report["trades"][0]["status"] == "open"
    assert report["scorecard"]["open_positions"] == 1