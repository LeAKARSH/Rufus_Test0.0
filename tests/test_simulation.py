import pytest

from rufus import db
from rufus.simulation import prices_from_snapshots, run_simulation_cycle
from rufus.sizing import make_sizer


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    db.initialize_database(c)
    yield c
    c.close()


def _seed_rec(conn, ticker, recommendation, run_date="2026-09-22", confidence="MEDIUM", price=None):
    db.upsert_ticker(conn, ticker)
    db.insert_recommendation(
        conn, ticker, run_date=run_date, recommendation=recommendation,
        confidence=confidence,
    )
    if price is not None:
        db.insert_price_snapshot(conn, ticker, captured_at=f"{run_date}T10:00:00+00:00", price=price)
    return db.get_latest_recommendation(conn, ticker)


def _portfolio(conn, cash=10_000.0):
    return db.ensure_default_portfolio(conn, name="default", starting_cash=cash)


def _prices(conn, tickers, price=100.0):
    return {t: price for t in tickers}


def test_buy_opens_position_and_marks_action(conn):
    p = _portfolio(conn)
    rec = _seed_rec(conn, "TCS.NS", "BUY", price=100.0)
    results = run_simulation_cycle(conn, p["id"], [rec], _prices(conn, ["TCS.NS"]), make_sizer("equal_weight"))
    assert results[0]["action"] == "BUY"
    assert results[0]["quantity"] == 100.0
    open_rows = db.list_open_positions(conn, p["id"])
    assert len(open_rows) == 1
    assert open_rows[0]["entry_recommendation_id"] == rec["id"]
    assert db.get_portfolio(conn, p["id"])["current_cash"] < 10_000.0
    assert db.get_portfolio_action(conn, rec["id"])["action"] == "BUY"


def test_no_double_trade_on_rerun(conn):
    p = _portfolio(conn)
    rec = _seed_rec(conn, "TCS.NS", "BUY", price=100.0)
    first = run_simulation_cycle(conn, p["id"], [rec], _prices(conn, ["TCS.NS"]), make_sizer("equal_weight"))
    second = run_simulation_cycle(conn, p["id"], [rec], _prices(conn, ["TCS.NS"]), make_sizer("equal_weight"))
    assert first[0]["action"] == "BUY"
    assert second[0]["action"] == "SKIP"
    assert second[0]["reason"] == "already_acted:BUY"
    assert len(db.list_open_positions(conn, p["id"])) == 1


def test_buy_while_holding_is_skipped(conn):
    p = _portfolio(conn)
    db.open_position(conn, p["id"], "TCS.NS", 50.0, 10, None, "2026-09-22")
    rec = _seed_rec(conn, "TCS.NS", "BUY", price=100.0)
    results = run_simulation_cycle(conn, p["id"], [rec], _prices(conn, ["TCS.NS"]), make_sizer("equal_weight"))
    assert results[0]["action"] == "SKIP"
    assert results[0]["reason"] == "already_holding"
    assert len(db.list_open_positions(conn, p["id"])) == 1


def test_sell_closes_position_with_pnl(conn):
    p = _portfolio(conn)
    rec_buy = _seed_rec(conn, "RELIANCE.NS", "BUY", run_date="2026-09-20", price=100.0)
    db.insert_price_snapshot(conn, "RELIANCE.NS", captured_at="2026-09-20T10:00:00+00:00", price=100.0)
    run_simulation_cycle(conn, p["id"], [rec_buy], _prices(conn, ["RELIANCE.NS"]), make_sizer("equal_weight"))
    rec_sell = _seed_rec(conn, "RELIANCE.NS", "SELL", run_date="2026-09-23", price=0.0)
    results = run_simulation_cycle(
        conn, p["id"], [rec_sell], {"RELIANCE.NS": 120.0}, make_sizer("equal_weight")
    )
    assert results[0]["action"] == "SELL"
    assert results[0]["realized_pnl"] == pytest.approx(20.0 * 100.0)
    assert db.list_open_positions(conn, p["id"]) == []
    assert db.get_portfolio(conn, p["id"])["current_cash"] == pytest.approx(12_000.0)


def test_sell_without_position_is_skipped(conn):
    p = _portfolio(conn)
    rec = _seed_rec(conn, "TCS.NS", "SELL", price=100.0)
    results = run_simulation_cycle(conn, p["id"], [rec], _prices(conn, ["TCS.NS"]), make_sizer("equal_weight"))
    assert results[0]["action"] == "SKIP"
    assert results[0]["reason"] == "not_holding"


def test_buy_without_price_is_skipped(conn):
    p = _portfolio(conn)
    rec = _seed_rec(conn, "TCS.NS", "BUY")
    results = run_simulation_cycle(conn, p["id"], [rec], {}, make_sizer("equal_weight"))
    assert results[0]["action"] == "SKIP"
    assert results[0]["reason"] == "no_price"
    assert db.list_open_positions(conn, p["id"]) == []


def test_buy_insufficient_cash_is_skipped(conn):
    p = _portfolio(conn, cash=100.0)
    rec = _seed_rec(conn, "TCS.NS", "BUY", price=1_000.0)
    # A strategy that returns absurd quantity must never overspend the ledger.
    results = run_simulation_cycle(
        conn, p["id"], [rec], {"TCS.NS": 1_000.0}, lambda ctx: 1e9
    )
    assert results[0]["action"] == "SKIP"
    assert results[0]["reason"] == "insufficient_cash"
    assert db.list_open_positions(conn, p["id"]) == []


def test_avoid_is_considered_and_passed(conn):
    p = _portfolio(conn)
    rec = _seed_rec(conn, "TSLA", "AVOID", price=100.0)
    results = run_simulation_cycle(conn, p["id"], [rec], _prices(conn, ["TSLA"]), make_sizer("equal_weight"))
    assert results[0]["action"] == "NONE"
    assert results[0]["reason"] == "considered and passed"
    marker = db.get_portfolio_action(conn, rec["id"])
    assert marker["action"] == "NONE"
    assert db.list_open_positions(conn, p["id"]) == []


def test_equal_weight_splits_budget_across_two_buys(conn):
    p = _portfolio(conn)
    rec_a = _seed_rec(conn, "RELIANCE.NS", "BUY", price=100.0)
    rec_b = _seed_rec(conn, "TCS.NS", "BUY", price=100.0)
    results = run_simulation_cycle(conn, p["id"], [rec_a, rec_b], _prices(conn, ["RELIANCE.NS", "TCS.NS"]), make_sizer("equal_weight"))
    buys = {r["ticker"]: r["quantity"] for r in results if r["action"] == "BUY"}
    assert buys == {"RELIANCE.NS": 50.0, "TCS.NS": 50.0}
    assert db.get_portfolio(conn, p["id"])["current_cash"] == pytest.approx(0.0)


def test_unknown_recommendation_skipped(conn):
    p = _portfolio(conn)
    db.upsert_ticker(conn, "TCS.NS")
    db.insert_recommendation(conn, "TCS.NS", run_date="2026-09-22", recommendation="STONKS")
    rec = db.get_latest_recommendation(conn, "TCS.NS")
    results = run_simulation_cycle(conn, p["id"], [rec], _prices(conn, ["TCS.NS"]), make_sizer("equal_weight"))
    assert results[0]["action"] == "SKIP"
    assert results[0]["reason"] == "unknown_recommendation:STONKS"


def test_prices_from_snapshots_uses_latest_stored(conn):
    _portfolio(conn)
    db.upsert_ticker(conn, "TCS.NS")
    db.insert_price_snapshot(conn, "TCS.NS", captured_at="2026-09-20T10:00:00+00:00", price=90.0)
    db.insert_price_snapshot(conn, "TCS.NS", captured_at="2026-09-22T10:00:00+00:00", price=110.0)
    rec = db.insert_recommendation(conn, "TCS.NS", run_date="2026-09-22", recommendation="BUY")  # noqa: F841
    row = db.get_latest_recommendation(conn, "TCS.NS")
    assert prices_from_snapshots(conn, [row]) == {"TCS.NS": 110.0}


def test_closed_position_traces_back_to_entry_and_exit_recommendations(conn):
    """Spec §7.4 traceability: a fill links to the reasoning that triggered it."""
    p = _portfolio(conn)
    db.upsert_ticker(conn, "TCS.NS")
    db.insert_recommendation(
        conn, "TCS.NS", run_date="2026-09-20", recommendation="BUY",
        reasoning="strong fundamentals", model="qwen3:32b",
    )
    rec_buy = db.get_latest_recommendation(conn, "TCS.NS")
    pid = db.open_position(
        conn, p["id"], "TCS.NS", 100.0, 10, entry_recommendation_id=rec_buy["id"],
        entry_date="2026-09-20",
    )
    db.insert_recommendation(
        conn, "TCS.NS", run_date="2026-09-28", recommendation="SELL",
        reasoning="risk materialized", model="qwen3:32b",
    )
    rec_sell = db.get_latest_recommendation(conn, "TCS.NS")
    db.close_position(conn, pid, exit_price=110.0, exit_recommendation_id=rec_sell["id"],
                      exit_date="2026-09-28")

    # Closed position carries both the entry and exit recommendation ids, so
    # analysis can walk back to each call's reasoning.
    closed = [pos for pos in db.list_positions(conn, p["id"])][0]
    assert closed["status"] == "closed"
    assert closed["entry_recommendation_id"] == rec_buy["id"]
    assert closed["exit_recommendation_id"] == rec_sell["id"]
    entry = db.get_recommendations(conn, "TCS.NS")
    assert entry[0]["reasoning"] == "risk materialized"
    assert entry[1]["reasoning"] == "strong fundamentals"