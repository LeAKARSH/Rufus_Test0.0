import json

import pytest

import rufus.db as db
from rufus.config import Settings
from rufus.paper import run_paper_cycle
from rufus.report import build_report, generate_daily_report
from rufus.report_html import _payload, render


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    db.initialize_database(c)
    yield c
    c.close()


@pytest.fixture()
def settings(tmp_path):
    return Settings(
        _env_file=None,
        portfolio_name="default",
        starting_cash=10_000.0,
        position_sizing_strategy="equal_weight",
        benchmark_ticker="^NSEI",
        max_allocation_pct=100.0,
        report_dir=str(tmp_path / "reports"),
    )


def _seed(conn, ticker="TCS.NS", rec="BUY", run_date="2026-09-22", price=200.0, reasoning=None):
    db.upsert_ticker(conn, ticker)
    db.insert_recommendation(
        conn, ticker, run_date=run_date, recommendation=rec,
        confidence="MEDIUM", reasoning=reasoning or "reasoning text",
        key_catalysts=["cheap"], key_risks=["volatile"],
    )
    db.insert_price_snapshot(conn, ticker, captured_at=f"{run_date}T10:00:00+00:00", price=price,
                             sma_50=price * 0.9, sma_200=price * 0.8)


def test_payload_roundtrip():
    data = {"a": [1, 2.5, None, "₹ text"], "b": {"x": True}}
    parsed = json.loads(_payload(data))
    assert parsed == data


def test_payload_escapes_script_end():
    data = {"reasoning": "close </script><!-- comment here"}
    dumped = _payload(data)
    assert "</script>" not in dumped
    assert json.loads(dumped)["reasoning"] == data["reasoning"]


def _extract_payload(text):
    marker = "const RUFUS_DATA = "
    start = text.index(marker) + len(marker)
    end = text.index("\n</script>", start)
    return json.loads(text[start:end].rstrip(";").strip())


def test_render_structure_and_payload(conn, settings):
    _seed(conn)
    run_paper_cycle(conn, settings)
    text = render(build_report(conn, settings))
    assert text.startswith("<!DOCTYPE html>")
    assert text.rstrip().endswith("</html>")
    assert "<style>" in text
    assert "lineChart" in text  # SVG chart renderer ships in the page
    assert "const RUFUS_DATA = {" in text
    payload = _extract_payload(text)
    assert payload["benchmark_ticker"] == "^NSEI"
    assert payload["portfolio"]["portfolio"] == "default"
    assert payload["watchlist"][0]["ticker"] == "TCS.NS"
    assert payload["tickers"][0]["price_series"][0]["price"] == 200.0
    assert len(payload["portfolio"]["equity_curve"]) == 1


def test_render_inline_safety_bad_json_never_matters(conn, settings):
    _seed(conn, reasoning="<script>alert('x')</script><img src=x onerror=1>")
    text = render(build_report(conn, settings))
    payload = _extract_payload(text)
    assert payload["tickers"][0]["latest_recommendation"]["reasoning"].startswith("<script>")


def test_render_empty_data_has_no_payload_breakout():
    text = render({
        "generated_at": "2026-09-22T12:00:00+00:00",
        "portfolio_name": "default",
        "benchmark_ticker": "^NSEI",
        "watchlist": [], "portfolio": {}, "trades": [], "scorecard": {},
        "recommendation_history": [], "tickers": [],
    })
    assert "RUFUS_DATA" in text
    assert "</script>" in text


def test_generate_html_writes_file(conn, settings):
    _seed(conn)
    run_paper_cycle(conn, settings)
    path = generate_daily_report(conn, settings, fmt="html", today="2026-09-22")
    assert path.name == "report-2026-09-22.html"
    text = path.read_text(encoding="utf-8")
    assert "lineChart" in text  # SVG charts are client-rendered from the payload
    assert "Advisory only, not financial advice." in text