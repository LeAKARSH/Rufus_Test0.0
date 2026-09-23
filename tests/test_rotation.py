from datetime import date, timedelta

import pytest

import rufus.db as db
from rufus.config import Settings
from rufus.rotation import plan_daily_news_rotation, queries_per_ticker


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
        news_rotation_earnings_window_days=14,
        news_rotation_overdue_days=7,
        news_rotation_volatility_spike_pct=50.0,
    )


def add_ticker(conn, ticker, keywords='"Acme Corp" OR ACME'):
    db.upsert_ticker(conn, ticker)
    db.set_ticker_keywords(conn, ticker, keywords)


def add_price_pairs(conn, ticker, vols):
    """Two snapshots (older then newest) with the given volatility_90d values."""
    db.insert_price_snapshot(
        conn, ticker,
        captured_at="2026-09-19T15:00:00+00:00", price=100.0, volatility_90d=vols[1],
    )
    db.insert_price_snapshot(
        conn, ticker,
        captured_at="2026-09-20T15:00:00+00:00", price=102.0, volatility_90d=vols[0],
    )


def news_run(conn, ticker, run_date):
    conn.execute(
        "INSERT INTO news_snapshots (ticker, run_date, captured_at) VALUES (?, ?, ?)",
        (ticker.upper(), str(run_date), "2026-09-01T00:00:00+00:00"),
    )
    conn.commit()


def test_queries_per_ticker_math():
    assert queries_per_ticker(100, 7) == 14
    assert queries_per_ticker(budget=0, watchlist_size=3) == 0
    assert queries_per_ticker(budget=100, watchlist_size=200) == 0
    assert queries_per_ticker(100, 0) == 0


def test_empty_watchlist_returns_empty(conn, settings):
    assert plan_daily_news_rotation(conn, budget=100, settings=settings) == []


def test_tickers_without_keywords_are_skipped(conn, settings):
    db.upsert_ticker(conn, "AAPL")  # no keywords -> excluded
    assert plan_daily_news_rotation(conn, budget=100, settings=settings) == []


def test_all_tickers_pulled_when_budget_covers_watchlist(conn, settings):
    today = date(2026, 9, 20)
    for t in ("AAPL", "MSFT"):
        add_ticker(conn, t)
        news_run(conn, t, today)  # fresh -> Tier B, quiet
    plan = plan_daily_news_rotation(conn, budget=100, today=today, settings=settings)
    assert len(plan) == 2
    assert plan[0].queries == 50 and plan[0].tier == "B"
    assert {i.ticker for i in plan} == {"AAPL", "MSFT"}


def test_never_pulled_ticker_is_tier_a(conn, settings):
    today = date(2026, 9, 20)
    add_ticker(conn, "AAPL")  # no news run -> overdue
    add_ticker(conn, "MSFT")
    news_run(conn, "MSFT", today - timedelta(days=1))  # quiet Tier B
    plan = plan_daily_news_rotation(conn, budget=100, today=today, settings=settings)
    aapl = next(i for i in plan if i.ticker == "AAPL")
    msft = next(i for i in plan if i.ticker == "MSFT")
    assert aapl.tier == "A" and msft.tier == "B"
    assert plan.index(aapl) < plan.index(msft)


def test_overdue_after_gap_is_tier_a(conn, settings):
    today = date(2026, 9, 20)
    add_ticker(conn, "AAPL")
    news_run(conn, "AAPL", today - timedelta(days=8))  # older than 7 days
    add_ticker(conn, "MSFT")
    news_run(conn, "MSFT", today - timedelta(days=6))  # still fresh
    plan = plan_daily_news_rotation(conn, budget=100, today=today, settings=settings)
    assert next(i for i in plan if i.ticker == "AAPL").tier == "A"
    assert next(i for i in plan if i.ticker == "MSFT").tier == "B"


def test_earnings_date_within_window_is_tier_a(conn, settings):
    today = date(2026, 9, 20)
    add_ticker(conn, "AAPL")
    news_run(conn, "AAPL", today)  # fresh, quiet without earnings
    plan = plan_daily_news_rotation(
        conn, budget=100, today=today,
        earnings_dates={"AAPL": date(2026, 9, 25)}, settings=settings,
    )
    assert next(i for i in plan if i.ticker == "AAPL").tier == "A"


def test_earnings_outside_window_is_not_catalyst(conn, settings):
    today = date(2026, 9, 20)
    add_ticker(conn, "AAPL")
    news_run(conn, "AAPL", today)
    plan = plan_daily_news_rotation(
        conn, budget=100, today=today,
        earnings_dates={"AAPL": date(2027, 1, 15)}, settings=settings,
    )
    assert next(i for i in plan if i.ticker == "AAPL").tier == "B"


def test_volatility_spike_is_tier_a(conn, settings):
    today = date(2026, 9, 20)
    add_ticker(conn, "AAPL")
    news_run(conn, "AAPL", today)
    add_price_pairs(conn, "AAPL", vols=[0.06, 0.02])  # +200%, above 50% threshold
    plan = plan_daily_news_rotation(conn, budget=100, today=today, settings=settings)
    assert next(i for i in plan if i.ticker == "AAPL").tier == "A"


def test_volatility_no_spike_is_quiet(conn, settings):
    today = date(2026, 9, 20)
    add_ticker(conn, "AAPL")
    news_run(conn, "AAPL", today)
    add_price_pairs(conn, "AAPL", vols=[0.021, 0.02])  # +5%, below threshold
    plan = plan_daily_news_rotation(conn, budget=100, today=today, settings=settings)
    assert next(i for i in plan if i.ticker == "AAPL").tier == "B"


def test_tier_a_guaranteed_when_quota_is_zero(conn, settings):
    today = date(2026, 9, 20)
    add_ticker(conn, "AAPL")  # overdue -> Tier A
    for t in ("MSFT", "TSLA", "NVDA"):
        add_ticker(conn, t)
        news_run(conn, t, today)  # quiet Tier B
    plan = plan_daily_news_rotation(conn, budget=1, today=today, settings=settings)
    assert len(plan) == 1
    assert plan[0].ticker == "AAPL"
    assert plan[0].queries == 1 and plan[0].tier == "A"


def test_plan_never_exceeds_budget(conn, settings):
    today = date(2026, 9, 20)
    for t in ("AAPL", "MSFT", "TSLA", "NVDA", "IBM"):
        add_ticker(conn, t)  # all overdue Tier A
    plan = plan_daily_news_rotation(conn, budget=13, today=today, settings=settings)
    assert sum(i.queries for i in plan) <= 13
    assert all(i.queries >= 1 for i in plan)


def test_tier_b_rotation_cycles_across_days(conn, settings):
    today = date(2026, 9, 20)
    add_ticker(conn, "AAPL")  # overdue -> Tier A
    quiet = ["MSFT", "TSLA", "NVDA", "IBM"]
    for t in quiet:
        add_ticker(conn, t)
        news_run(conn, t, today)
    plan_d0 = plan_daily_news_rotation(conn, budget=5, today=today, settings=settings)
    plan_d1 = plan_daily_news_rotation(
        conn, budget=5, today=today + timedelta(days=1), settings=settings
    )
    b_d0 = [i.ticker for i in plan_d0 if i.tier == "B"]
    b_d1 = [i.ticker for i in plan_d1 if i.tier == "B"]
    assert b_d0 != b_d1  # different start offset
    assert set(b_d0) == set(quiet) and set(b_d1) == set(quiet)


def test_budget_consumed_uses_earliest_overdue_first(conn, settings):
    today = date(2026, 9, 20)
    add_ticker(conn, "AAPL")
    news_run(conn, "AAPL", today - timedelta(days=30))  # longest overdue
    add_ticker(conn, "MSFT")
    news_run(conn, "MSFT", today - timedelta(days=8))  # overdue but less
    plan = plan_daily_news_rotation(conn, budget=1, today=today, settings=settings)
    assert plan[0].ticker == "AAPL"