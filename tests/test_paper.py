import pytest

from rufus import db
from rufus.config import Settings
from rufus.paper import paper_summary, refresh_equity, run_paper_cycle
from rufus.paper_cli import _print


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


def _seed(conn, ticker, recommendation, run_date="2026-09-22", confidence="MEDIUM", price=None):
    db.upsert_ticker(conn, ticker)
    db.insert_recommendation(conn, ticker, run_date=run_date, recommendation=recommendation, confidence=confidence)
    if price is not None:
        db.insert_price_snapshot(conn, ticker, captured_at=f"{run_date}T10:00:00+00:00", price=price)
    return db.get_latest_recommendation(conn, ticker)


class FakeBenchmark:
    def __init__(self, pair=None):
        self._pair = pair
        self.calls = 0

    def fetch_benchmark_close(self, ticker):
        self.calls += 1
        return self._pair


def test_run_paper_cycle_opens_positions_and_snapshots(conn, settings):
    _seed(conn, "RELIANCE.NS", "BUY", price=100.0)
    _seed(conn, "TCS.NS", "BUY", price=200.0)
    bench = FakeBenchmark(pair=("2026-09-22", 26000.0))
    out = run_paper_cycle(conn, settings, benchmark_client=bench)
    acts = {r["ticker"]: r["action"] for r in out["results"]}
    assert acts == {"RELIANCE.NS": "BUY", "TCS.NS": "BUY"}
    pid = out["portfolio_id"]
    assert len(db.list_open_positions(conn, pid)) == 2
    # Equal weight over two buys: 5000 each -> 50 x 100 and 25 x 200.
    snap = out["snapshot"]
    assert snap["total_value"] == pytest.approx(10_000.0)
    assert snap["cash"] == pytest.approx(0.0)
    rows = db.get_portfolio_value_snapshots(conn, pid)
    assert len(rows) == 1
    assert rows[0]["benchmark_value"] == 26000.0
    assert bench.calls == 1
    assert db.get_benchmark_price(conn, "^NSEI", "2026-09-22")["close"] == 26000.0


def test_paper_cycle_is_idempotent_across_runs(conn, settings):
    _seed(conn, "TCS.NS", "BUY", price=100.0)
    run_paper_cycle(conn, settings)
    out = run_paper_cycle(conn, settings)
    pid = out["portfolio_id"]
    acts = {r["ticker"]: r["action"] for r in out["results"]}
    assert acts == {"TCS.NS": "SKIP"}
    assert len(db.list_open_positions(conn, pid)) == 1


def test_paper_cycle_executes_seed_then_sell(conn, settings):
    _seed(conn, "TCS.NS", "BUY", run_date="2026-09-20", price=100.0)
    run_paper_cycle(conn, settings, captured_at="2026-09-20T10:00:00+00:00")
    _seed(conn, "TCS.NS", "SELL", run_date="2026-09-23", price=0.0)
    db.insert_price_snapshot(conn, "TCS.NS", captured_at="2026-09-23T10:00:00+00:00", price=130.0)
    out = run_paper_cycle(conn, settings, captured_at="2026-09-23T10:00:00+00:00")
    pid = out["portfolio_id"]
    sell = [r for r in out["results"] if r["ticker"] == "TCS.NS"][0]
    assert sell["action"] == "SELL"
    assert sell["realized_pnl"] == pytest.approx(30.0 * 100.0)
    assert db.list_open_positions(conn, pid) == []
    # Cash: 10000 - 10000(buy) + 13000(sell) = 13000
    assert db.get_portfolio(conn, pid)["current_cash"] == pytest.approx(13_000.0)


def test_refresh_equity_records_additional_curve_point(conn, settings):
    _seed(conn, "TCS.NS", "BUY", price=100.0)
    run_paper_cycle(conn, settings, captured_at="2026-09-22T10:00:00+00:00")
    db.insert_price_snapshot(conn, "TCS.NS", captured_at="2026-09-22T12:00:00+00:00", price=105.0)
    refresh_equity(conn, settings, captured_at="2026-09-22T12:30:00+00:00")
    snapshots = db.get_portfolio_value_snapshots(conn, db.get_portfolio(conn, name=settings.portfolio_name)["id"])
    assert len(snapshots) == 2
    assert snapshots[0]["total_value"] == pytest.approx(10_500.0)


def test_paper_summary_reports_benchmark_and_equity(conn, settings):
    _seed(conn, "TCS.NS", "BUY", price=100.0)
    bench = FakeBenchmark(pair=("2026-09-21", 25000.0))
    run_paper_cycle(conn, settings, benchmark_client=bench, captured_at="2026-09-21T10:00:00+00:00")
    bench2 = FakeBenchmark(pair=("2026-09-22", 26000.0))
    run_paper_cycle(conn, settings, benchmark_client=bench2, captured_at="2026-09-22T10:00:00+00:00")
    summary = paper_summary(conn, settings)
    assert summary["total_value"] == pytest.approx(10_000.0)
    assert len(summary["open_positions"]) == 1
    assert summary["equity_points"] == 2
    assert summary["benchmark_ticker"] == "^NSEI"
    assert summary["benchmark_return"] == pytest.approx(0.04)


def test_paper_cli_print_renders_summary(capsys):
    summary = {
        "portfolio": "default",
        "cash": 5000.0,
        "positions_value": 6000.0,
        "total_value": 11000.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 1000.0,
        "open_positions": [
            {
                "ticker": "TCS.NS", "quantity": 50.0, "entry_price": 100.0,
                "current_price": 120.0, "unrealized_pnl": 1000.0,
            }
        ],
        "benchmark_ticker": "^NSEI",
        "benchmark_return": 0.04,
        "equity_points": 3,
    }
    _print(summary)
    out = capsys.readouterr().out
    assert "Total value   : 11,000.00" in out
    assert "Benchmark (^NSEI) buy-hold return: +4.00%" in out
    assert "TCS.NS" in out