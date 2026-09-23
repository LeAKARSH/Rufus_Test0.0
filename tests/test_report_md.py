import pytest

from rufus import db
from rufus.config import Settings
from rufus.paper import run_paper_cycle
from rufus.report import build_report
from rufus.report_md import render


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


def _seed(conn, ticker, rec, run_date="2026-09-22", price=None, reasoning=None):
    db.upsert_ticker(conn, ticker)
    db.insert_recommendation(conn, ticker, run_date=run_date, recommendation=rec,
                             confidence="MEDIUM", reasoning=reasoning or "reasoning text")
    if price is not None:
        db.insert_price_snapshot(conn, ticker, captured_at=f"{run_date}T10:00:00+00:00", price=price)


def test_render_seeded_report(conn, settings):
    _seed(conn, "TCS.NS", "BUY", price=200.0, reasoning="cheap")
    run_paper_cycle(conn, settings)
    text = render(build_report(conn, settings))

    assert "Paper" not in text  # only used in heading
    assert "## Watchlist" in text
    assert "## Portfolio" in text
    assert "## Holdings" in text
    assert "## Equity curve" in text
    assert "## Scorecard (measured so far)" in text
    assert "## Trade log" in text
    assert "## Recommendation history" in text
    assert "| TCS.NS | BUY | MEDIUM | 2026-09-22 |" in text
    assert "No open positions." not in text
    assert "9,998.75" in text or "10,000.00" in text
    assert "Advisory only, not financial advice." in text


def test_render_empty_data_renders_placeholders():
    text = render({
        "generated_at": "2026-09-22T12:00:00+00:00",
        "portfolio_name": "default",
        "benchmark_ticker": "^NSEI",
        "watchlist": [],
        "portfolio": {
            "portfolio": "default", "cash": 0.0, "positions_value": 0.0,
            "total_value": 0.0, "realized_pnl": 0.0, "unrealized_pnl": 0.0,
            "total_pnl": 0.0, "open_positions": [], "closed_positions": 0,
            "equity_curve": [], "benchmark_ticker": "^NSEI",
            "benchmark_return": None, "portfolio_return": None,
        },
        "trades": [],
        "scorecard": {
            "decided_closed": 0, "hits": 0, "hit_rate": None,
            "avg_realized_pnl": None, "total_realized_pnl": 0.0,
            "closed_positions": 0, "open_positions": 0, "by_direction": {},
        },
        "recommendation_history": [],
    })
    assert "_No tickers on the watchlist._" in text
    assert "_No open positions._" in text
    assert "_No equity-curve points recorded yet._" in text
    assert "_No simulated trades yet._" in text
    assert "_No recommendations recorded yet._" in text


def test_render_trade_log_lists_reasoning(conn, settings):
    _seed(conn, "TCS.NS", "BUY", run_date="2026-09-20", price=100.0, reasoning="buy reason")
    run_paper_cycle(conn, settings, captured_at="2026-09-20T10:00:00+00:00")
    _seed(conn, "TCS.NS", "SELL", run_date="2026-09-23", reasoning="sell reason")
    db.insert_price_snapshot(conn, "TCS.NS", captured_at="2026-09-23T10:00:00+00:00", price=130.0)
    run_paper_cycle(conn, settings, captured_at="2026-09-23T10:00:00+00:00")
    text = render(build_report(conn, settings))
    assert "| TCS.NS | closed |" in text
    assert "entry 2026-09-20 BUY: buy reason" in text
    assert "exit 2026-09-23 SELL: sell reason" in text


def test_render_scorecard_breakdown(conn, settings):
    _seed(conn, "TCS.NS", "BUY", run_date="2026-09-20", price=100.0)
    _seed(conn, "RELIANCE.NS", "BUY", run_date="2026-09-20", price=100.0)
    run_paper_cycle(conn, settings, captured_at="2026-09-20T10:00:00+00:00")
    _seed(conn, "TCS.NS", "SELL", run_date="2026-09-23")
    _seed(conn, "RELIANCE.NS", "SELL", run_date="2026-09-23")
    db.insert_price_snapshot(conn, "TCS.NS", captured_at="2026-09-23T10:00:00+00:00", price=130.0)
    db.insert_price_snapshot(conn, "RELIANCE.NS", captured_at="2026-09-23T10:00:00+00:00", price=90.0)
    run_paper_cycle(conn, settings, captured_at="2026-09-23T10:00:00+00:00")

    text = render(build_report(conn, settings))
    assert "| BUY | 2 | 1 | +50.00% |" in text
    assert "| Direction hit rate | +50.00% |" in text