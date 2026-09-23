import sqlite3

import pytest

from rufus import db


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    db.initialize_database(c)
    yield c
    c.close()


def test_connect_creates_parent_dirs_and_db_file(tmp_path):
    path = tmp_path / "nested" / "dir" / "rufus.db"
    c = db.connect(path)
    try:
        assert path.exists()
    finally:
        c.close()


def test_initialize_database_idempotent(conn):
    db.initialize_database(conn)
    db.initialize_database(conn)
    assert db._schema_version(conn) == db.SCHEMA_VERSION


def test_ticker_lifecycle(conn):
    db.upsert_ticker(conn, "aapl")
    assert db.get_active_tickers(conn) == ["AAPL"]

    db.upsert_ticker(conn, "MSFT", notes="first")
    db.upsert_ticker(conn, "msft", notes="second")
    assert db.get_active_tickers(conn) == ["AAPL", "MSFT"]
    row = conn.execute(
        "SELECT notes FROM tickers WHERE ticker = 'MSFT'"
    ).fetchone()
    assert row["notes"] == "second"

    db.set_ticker_active(conn, "AAPL", False)
    assert db.get_active_tickers(conn) == ["MSFT"]

    db.remove_ticker(conn, "MSFT")
    assert db.get_active_tickers(conn) == []


def test_upsert_ticker_preserves_added_date(conn):
    db.upsert_ticker(conn, "AAPL")
    added = conn.execute(
        "SELECT added_date FROM tickers WHERE ticker = 'AAPL'"
    ).fetchone()["added_date"]
    db.upsert_ticker(conn, "AAPL")
    added_again = conn.execute(
        "SELECT added_date FROM tickers WHERE ticker = 'AAPL'"
    ).fetchone()["added_date"]
    assert added == added_again


def test_price_snapshot_insert_and_retrieve(conn):
    db.upsert_ticker(conn, "AAPL")
    db.insert_price_snapshot(
        conn,
        "AAPL",
        captured_at="2026-01-01T15:00:00+00:00",
        price=227.5,
        sma_50=220.1,
        sma_200=205.3,
        trend_signal="golden_cross_recent",
        rsi_14=58.2,
        pe_ratio=34.1,
        data_json='{"ticker": "AAPL"}',
    )
    db.insert_price_snapshot(
        conn,
        "AAPL",
        captured_at="2026-01-02T15:00:00+00:00",
        price=230.0,
    )
    snaps = db.get_recent_price_snapshots(conn, "AAPL")
    assert [s["price"] for s in snaps] == [230.0, 227.5]
    assert snaps[1]["sma_50"] == 220.1


def test_price_snapshot_same_timestamp_overwrites(conn):
    db.upsert_ticker(conn, "AAPL")
    db.insert_price_snapshot(
        conn, "AAPL", captured_at="2026-01-01T15:00:00+00:00", price=100.0
    )
    db.insert_price_snapshot(
        conn, "AAPL", captured_at="2026-01-01T15:00:00+00:00", price=110.0
    )
    snaps = db.get_recent_price_snapshots(conn, "AAPL")
    assert len(snaps) == 1
    assert snaps[0]["price"] == 110.0


def test_price_snapshot_ignores_unknown_fields(conn):
    db.upsert_ticker(conn, "AAPL")
    db.insert_price_snapshot(
        conn,
        "AAPL",
        captured_at="2026-01-01T15:00:00+00:00",
        price=100.0,
        not_a_column="boom",
    )
    row = conn.execute(
        "SELECT * FROM price_snapshots WHERE ticker = 'AAPL'"
    ).fetchone()
    assert row["price"] == 100.0


def test_price_snapshot_requires_ticker(conn):
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_price_snapshot(
            conn,
            "NOSUCH",
            captured_at="2026-01-01T15:00:00+00:00",
            price=1.0,
        )


def test_api_usage_increment_and_query(conn):
    assert db.get_api_usage(conn, "yahoo", "2026-01-01T15") == 0
    total = db.increment_api_usage(conn, "yahoo", "2026-01-01T15")
    assert total == 1
    total = db.increment_api_usage(conn, "yahoo", "2026-01-01T15", delta=3)
    assert total == 4
    assert db.get_api_usage(conn, "yahoo", "2026-01-01T15") == 4
    assert db.get_api_usage(conn, "currents", "2026-01-01T15") == 0


def test_schema_has_all_planned_tables(conn):
    tables = {
        r["name"]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    expected = {
        "tickers",
        "price_snapshots",
        "news_snapshots",
        "recommendations",
        "api_usage_log",
        "portfolios",
        "portfolio_value_snapshots",
        "simulated_positions",
    }
    assert expected <= tables


def test_utc_now_iso_format():
    value = db.utc_now_iso()
    assert "T" in value and "+00:00" in value


def test_keywords_column_added_by_migration_v2(conn):
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(tickers)").fetchall()}
    assert "search_keywords" in cols


def test_keywords_set_get_and_clear(conn):
    db.upsert_ticker(conn, "AAPL")
    assert db.get_ticker_keywords(conn, "AAPL") is None

    db.set_ticker_keywords(conn, "aapl", '"Apple Inc." OR AAPL')
    assert db.get_ticker_keywords(conn, "AAPL") == '"Apple Inc." OR AAPL'

    db.set_ticker_keywords(conn, "AAPL", None)
    assert db.get_ticker_keywords(conn, "AAPL") is None


def test_active_tickers_with_keywords_filters(conn):
    db.upsert_ticker(conn, "AAPL")
    db.upsert_ticker(conn, "MSFT")
    db.upsert_ticker(conn, "TSLA")

    db.set_ticker_keywords(conn, "AAPL", '"Apple Inc." OR AAPL')
    db.set_ticker_keywords(conn, "MSFT", '"Microsoft Corp" OR MSFT')
    db.set_ticker_keywords(conn, "TSLA", "   ")

    rows = db.get_active_tickers_with_keywords(conn)
    assert [r["ticker"] for r in rows] == ["AAPL", "MSFT"]

    db.set_ticker_active(conn, "AAPL", False)
    rows = db.get_active_tickers_with_keywords(conn)
    assert [r["ticker"] for r in rows] == ["MSFT"]


def test_job_state_set_and_get(conn):
    assert db.get_job_last_run(conn, "news") is None
    db.set_job_last_run(conn, "news", "2026-09-20T06:00:00+00:00")
    assert db.get_job_last_run(conn, "news") == "2026-09-20T06:00:00+00:00"
    db.set_job_last_run(conn, "news", "2026-09-21T06:00:00+00:00")
    assert db.get_job_last_run(conn, "news") == "2026-09-21T06:00:00+00:00"


def test_news_snapshot_upsert_same_day(conn):
    db.upsert_ticker(conn, "AAPL")
    db.insert_news_snapshot(
        conn, "AAPL", run_date="2026-09-20", query_keyword="kw",
        sentiment_score_avg=0.3, articles_considered=2,
        top_headlines_json="[]", articles_json="[]",
    )
    db.insert_news_snapshot(
        conn, "AAPL", run_date="2026-09-20", query_keyword="kw",
        sentiment_score_avg=0.8, articles_considered=5,
    )
    rows = db.get_news_snapshots(conn, "AAPL")
    assert len(rows) == 1
    assert rows[0]["sentiment_score_avg"] == 0.8
    assert rows[0]["articles_considered"] == 5


def test_news_snapshot_lists_newest_first(conn):
    db.upsert_ticker(conn, "AAPL")
    db.insert_news_snapshot(conn, "AAPL", run_date="2026-09-19")
    db.insert_news_snapshot(conn, "AAPL", run_date="2026-09-20", sentiment_score_avg=0.4)
    rows = db.get_news_snapshots(conn, "AAPL")
    assert [r["run_date"] for r in rows] == ["2026-09-20", "2026-09-19"]


def test_prior_news_scores_exclude_current_and_nulls(conn):
    db.upsert_ticker(conn, "AAPL")
    db.insert_news_snapshot(conn, "AAPL", run_date="2026-09-10", sentiment_score_avg=0.5)
    db.insert_news_snapshot(conn, "AAPL", run_date="2026-09-12")  # null score
    db.insert_news_snapshot(conn, "AAPL", run_date="2026-09-13", sentiment_score_avg=0.7)
    db.insert_news_snapshot(conn, "AAPL", run_date="2026-09-20", sentiment_score_avg=0.9)
    prior = db.get_prior_news_scores(conn, "AAPL", before_run_date="2026-09-20")
    assert prior == [0.7, 0.5]


def test_recommendation_insert_and_latest(conn):
    db.upsert_ticker(conn, "AAPL")
    db.insert_recommendation(
        conn, "AAPL", run_date="2026-09-20", recommendation="BUY",
        confidence="MEDIUM", key_catalysts=["earnings"], key_risks=["rates"],
        model="qwen3:32b",
    )
    db.insert_recommendation(
        conn, "AAPL", run_date="2026-09-22", recommendation="HOLD", confidence="LOW",
    )
    latest = db.get_latest_recommendation(conn, "AAPL")
    assert latest["recommendation"] == "HOLD"
    assert latest["run_date"] == "2026-09-22"
    history = db.get_recommendations(conn, "AAPL")
    assert [r["run_date"] for r in history] == ["2026-09-22", "2026-09-20"]


def test_recommendation_upsert_same_day_overwrites(conn):
    db.upsert_ticker(conn, "AAPL")
    db.insert_recommendation(conn, "AAPL", run_date="2026-09-20", recommendation="BUY")
    db.insert_recommendation(conn, "AAPL", run_date="2026-09-20", recommendation="SELL")
    rows = db.get_recommendations(conn, "AAPL")
    assert len(rows) == 1
    assert rows[0]["recommendation"] == "SELL"


def test_latest_recommendations_per_ticker(conn):
    db.upsert_ticker(conn, "AAPL")
    db.upsert_ticker(conn, "MSFT")
    db.insert_recommendation(conn, "AAPL", run_date="2026-09-20", recommendation="BUY")
    db.insert_recommendation(conn, "AAPL", run_date="2026-09-22", recommendation="HOLD")
    db.insert_recommendation(conn, "MSFT", run_date="2026-09-21", recommendation="AVOID")
    rows = db.get_latest_recommendations(conn)
    assert {r["ticker"]: r["recommendation"] for r in rows} == {"AAPL": "HOLD", "MSFT": "AVOID"}
    filtered = db.get_latest_recommendations(conn, ["MSFT"])
    assert [r["ticker"] for r in filtered] == ["MSFT"]


def test_ensure_default_portfolio_seeds_cash(conn):
    p = db.ensure_default_portfolio(conn, name="default", starting_cash=10_000.0)
    assert p["starting_cash"] == 10_000.0
    assert p["current_cash"] == 10_000.0
    # Idempotent: calling again returns the same row, does not re-seed.
    again = db.ensure_default_portfolio(conn, name="default", starting_cash=10_000.0)
    assert again["id"] == p["id"]
    assert again["current_cash"] == 10_000.0


def test_open_position_deducts_cash(conn):
    p = db.ensure_default_portfolio(conn, starting_cash=10_000.0)
    pid = db.open_position(
        conn, p["id"], "RELIANCE.NS", entry_price=100.0, quantity=20,
        entry_recommendation_id=None, entry_date="2026-09-22",
    )
    assert pid > 0
    assert db.get_portfolio(conn, p["id"])["current_cash"] == 8_000.0
    open_rows = db.list_open_positions(conn, p["id"])
    assert [r["ticker"] for r in open_rows] == ["RELIANCE.NS"]
    assert open_rows[0]["quantity"] == 20
    assert open_rows[0]["entry_price"] == 100.0
    assert open_rows[0]["status"] == "open"


def test_close_position_credits_cash_and_pnl(conn):
    p = db.ensure_default_portfolio(conn, starting_cash=10_000.0)
    pid = db.open_position(
        conn, p["id"], "TCS.NS", entry_price=100.0, quantity=10,
        entry_recommendation_id=None, entry_date="2026-09-22",
    )
    realized = db.close_position(
        conn, pid, exit_price=120.0, exit_date="2026-09-23"
    )
    assert realized == 200.0
    pos = db.get_position(conn, pid)
    assert pos["status"] == "closed"
    assert pos["realized_pnl"] == 200.0
    assert pos["exit_price"] == 120.0
    # Cash: 10000 - (100*10) + (120*10) = 10200
    assert db.get_portfolio(conn, p["id"])["current_cash"] == 10_200.0
    assert db.list_open_positions(conn, p["id"]) == []
    with pytest.raises(ValueError):
        db.close_position(conn, pid, exit_price=130.0)


def test_list_open_positions_filters_by_ticker(conn):
    p = db.ensure_default_portfolio(conn, starting_cash=10_000.0)
    db.open_position(conn, p["id"], "RELIANCE.NS", 100.0, 5, None, "2026-09-22")
    db.open_position(conn, p["id"], "TCS.NS", 200.0, 5, None, "2026-09-22")
    rows = db.list_open_positions(conn, p["id"], ticker="TCS.NS")
    assert [r["ticker"] for r in rows] == ["TCS.NS"]
    assert len(db.list_open_positions(conn, p["id"])) == 2


def test_position_links_to_recommendations(conn):
    db.upsert_ticker(conn, "RELIANCE.NS")
    db.insert_recommendation(
        conn, "RELIANCE.NS", run_date="2026-09-22", recommendation="BUY"
    )
    rec = db.get_latest_recommendation(conn, "RELIANCE.NS")
    p = db.ensure_default_portfolio(conn, starting_cash=10_000.0)
    pid = db.open_position(
        conn, p["id"], "RELIANCE.NS", 100.0, 5, rec["id"], "2026-09-22"
    )
    pos = db.get_position(conn, pid)
    assert pos["entry_recommendation_id"] == rec["id"]
    db.close_position(conn, pid, exit_price=110.0,
                      exit_recommendation_id=rec["id"], exit_date="2026-09-23")
    pos = db.get_position(conn, pid)
    assert pos["exit_recommendation_id"] == rec["id"]


def test_portfolio_action_idempotent_marker(conn):
    db.upsert_ticker(conn, "RELIANCE.NS")
    db.insert_recommendation(conn, "RELIANCE.NS", run_date="2026-09-22", recommendation="BUY")
    rec = db.get_latest_recommendation(conn, "RELIANCE.NS")
    assert db.get_portfolio_action(conn, rec["id"]) is None
    db.record_portfolio_action(conn, rec["id"], "BUY", position_id=None, acted_at="2026-09-22T10:00:00+00:00")
    act = db.get_portfolio_action(conn, rec["id"])
    assert act is not None
    assert act["action"] == "BUY"
    # Upsert on the same recommendation keeps one marker row (no double-trade).
    db.record_portfolio_action(conn, rec["id"], "BUY", position_id=None, acted_at="2026-09-22T10:05:00+00:00")
    rows = conn.execute(
        "SELECT COUNT(*) FROM portfolio_actions WHERE recommendation_id = ?",
        (rec["id"],),
    ).fetchone()[0]
    assert rows == 1


def test_portfolio_value_snapshot_upsert_and_ordering(conn):
    p = db.ensure_default_portfolio(conn, starting_cash=10_000.0)
    db.insert_portfolio_value_snapshot(
        conn, p["id"], "2026-09-22T10:00:00+00:00", total_value=10_500.0,
        cash=8_000.0, positions_value=2_500.0,
    )
    db.insert_portfolio_value_snapshot(
        conn, p["id"], "2026-09-22T16:00:00+00:00", total_value=11_000.0,
        cash=8_000.0, positions_value=3_000.0,
    )
    # Same timestamp overwrites (intraday refresh restart-safe).
    db.insert_portfolio_value_snapshot(
        conn, p["id"], "2026-09-22T16:00:00+00:00", total_value=11_100.0,
        cash=8_000.0, positions_value=3_100.0,
    )
    rows = db.get_portfolio_value_snapshots(conn, p["id"])
    assert [r["captured_at"] for r in rows] == [
        "2026-09-22T16:00:00+00:00", "2026-09-22T10:00:00+00:00"
    ]
    assert rows[0]["total_value"] == 11_100.0


def test_migration_v6_adds_benchmark_columns_and_table(conn):
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(portfolio_value_snapshots)").fetchall()}
    assert "benchmark_value" in cols
    tables = {
        r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert "benchmark_prices" in tables


def test_benchmark_price_daos(conn):
    db.set_benchmark_price(conn, "^NSEI", "2026-09-20", 25000.0)
    db.set_benchmark_price(conn, "^NSEI", "2026-09-21", 25500.0)
    assert db.get_benchmark_price(conn, "^NSEI", "2026-09-21")["close"] == 25500.0
    on_or_before = db.get_benchmark_price_on_or_before(conn, "^NSEI", "2026-09-20")
    assert on_or_before["trade_date"] == "2026-09-20"
    assert db.get_latest_benchmark_price(conn, "^NSEI")["trade_date"] == "2026-09-21"


def test_benchmark_price_upsert_same_day_overwrites(conn):
    db.set_benchmark_price(conn, "^NSEI", "2026-09-21", 25500.0)
    db.set_benchmark_price(conn, "^NSEI", "2026-09-21", 25600.0)
    rows = conn.execute(
        "SELECT COUNT(*) FROM benchmark_prices WHERE ticker = '^NSEI' AND trade_date = '2026-09-21'"
    ).fetchone()[0]
    assert rows == 1
    assert db.get_benchmark_price(conn, "^NSEI", "2026-09-21")["close"] == 25600.0