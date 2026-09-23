import pytest

from rufus import db, valuation


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    db.initialize_database(c)
    yield c
    c.close()


@pytest.fixture()
def portfolio(conn):
    return db.ensure_default_portfolio(conn, name="default", starting_cash=10_000.0)


def test_snapshot_with_open_positions(conn, portfolio):
    db.open_position(conn, portfolio["id"], "TCS.NS", 100.0, 50, None, "2026-09-22")
    snap = valuation.snapshot(conn, portfolio["id"], prices={"TCS.NS": 120.0})
    assert snap["cash"] == 5_000.0
    assert snap["positions_value"] == 6_000.0
    assert snap["total_value"] == 11_000.0
    assert snap["unrealized_pnl"] == 1_000.0
    assert len(snap["open_positions"]) == 1
    assert snap["open_positions"][0]["current_price"] == 120.0


def test_snapshot_missing_price_marks_at_entry(conn, portfolio):
    db.open_position(conn, portfolio["id"], "TCS.NS", 100.0, 10, None, "2026-09-22")
    snap = valuation.snapshot(conn, portfolio["id"], prices={})
    assert snap["open_positions"][0]["current_price"] == 100.0
    assert snap["unrealized_pnl"] == 0.0


def test_snapshot_accumulates_realized_pnl(conn, portfolio):
    pid = db.open_position(conn, portfolio["id"], "TCS.NS", 100.0, 10, None, "2026-09-22")
    db.close_position(conn, pid, exit_price=120.0, exit_date="2026-09-23")
    snap = valuation.snapshot(conn, portfolio["id"])
    assert snap["realized_pnl"] == 200.0
    assert snap["total_pnl"] == 200.0


def test_record_equity_snapshot_persists(conn, portfolio):
    val = valuation.record_equity_snapshot(
        conn, portfolio["id"], "2026-09-22T10:00:00+00:00", benchmark_value=25000.0
    )
    assert val["total_value"] == 10_000.0
    rows = valuation.equity_curve(conn, portfolio["id"])
    assert len(rows) == 1
    assert rows[0]["benchmark_value"] == 25000.0
    assert rows[0]["total_value"] == 10_000.0


def test_benchmark_return_math():
    assert valuation.benchmark_return([100.0, 110.0]) == pytest.approx(0.10)
    assert valuation.benchmark_return([100.0, 90.0]) == pytest.approx(-0.10)
    assert valuation.benchmark_return([100.0]) is None
    assert valuation.benchmark_return([]) is None
    assert valuation.benchmark_return([0.0, 10.0]) is None


def test_portfolio_return_math():
    assert valuation.portfolio_return(10_000.0, 11_000.0) == pytest.approx(0.10)
    assert valuation.portfolio_return(0.0, 1.0) is None


def test_benchmark_since_inception_uses_series(conn, portfolio):
    for day, close in (("2026-09-20", 25000.0), ("2026-09-21", 25500.0), ("2026-09-22", 26000.0)):
        valuation.store_benchmark_for_day(conn, "^NSEI", day, close)
    assert valuation.benchmark_since_inception(conn, "^NSEI") == pytest.approx(0.04)
    # From inception date onward: series starts at 25500 -> 26000 = +1.96%
    assert valuation.benchmark_since_inception(conn, "^NSEI", since_date="2026-09-21") == pytest.approx(0.019607843, rel=1e-3)
    assert valuation.benchmark_since_inception(conn, "^NSEI", since_date="2026-09-22") is None