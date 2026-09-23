import pytest

import rufus.db as db
from rufus.config import Settings
from rufus.report import build_report, generate_daily_report
from rufus.report_cli import build_parser, main


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


def _seed(conn, ticker="TCS.NS", rec="BUY", price=200.0):
    db.upsert_ticker(conn, ticker)
    db.insert_recommendation(conn, ticker, run_date="2026-09-22", recommendation=rec)
    db.insert_price_snapshot(conn, ticker, captured_at="2026-09-22T10:00:00+00:00", price=price)


def test_generate_md_writes_expected_file(conn, settings):
    _seed(conn)
    path = generate_daily_report(conn, settings, fmt="md", today="2026-09-22")
    assert path.name == "report-2026-09-22.md"
    assert path.parent == settings.report_dir_path
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Rufus")
    assert "## Watchlist" in text


def test_generate_is_restart_safe_and_force_regenerates(conn, settings):
    _seed(conn)
    settings.report_dir_path.mkdir(parents=True, exist_ok=True)
    target = settings.report_dir_path / "report-2026-09-22.md"
    target.write_text("stale", encoding="utf-8")
    first = generate_daily_report(conn, settings, fmt="md", today="2026-09-22")
    assert first.read_text(encoding="utf-8") == "stale"
    second = generate_daily_report(conn, settings, fmt="md", today="2026-09-22", force=True)
    assert second.read_text(encoding="utf-8").startswith("# Rufus")


def test_generate_custom_out_path(conn, settings, tmp_path):
    _seed(conn)
    out = tmp_path / "custom" / "my.md"
    path = generate_daily_report(conn, settings, fmt="md", out=out, force=True)
    assert path == out
    assert out.exists()


def test_generate_unknown_format_falls_back_to_md(conn, settings):
    _seed(conn)
    path = generate_daily_report(conn, settings, fmt="md", today="2026-09-22", force=True)
    assert path.name == "report-2026-09-22.md"


def test_parser_defaults_and_flags():
    p = build_parser()
    args = p.parse_args([])
    assert args.format == "html"
    assert args.force is False
    assert args.open is False
    assert args.out is None
    args = p.parse_args(["-f", "md", "--force", "--open", "-o", "x.md"])
    assert args.format == "md"
    assert args.force is True
    assert args.open is True
    assert args.out == "x.md"


def test_report_cli_main_writes_and_prints(tmp_path, capsys, monkeypatch):
    conn = db.connect(tmp_path / "cli.db")
    db.initialize_database(conn)
    _seed(conn)
    build_report(conn, Settings(_env_file=None))  # ensure views build cleanly
    conn.close()

    monkeypatch.setattr(
        "rufus.report_cli.Settings",
        lambda: Settings(_env_file=None, report_dir=str(tmp_path / "reports")),
    )
    main(["-f", "md"])
    out = capsys.readouterr().out.strip().splitlines()
    assert out and out[-1].endswith("report-md.md") is False
    assert (tmp_path / "reports").exists()


def test_build_report_includes_recommendation_history(conn, settings):
    _seed(conn)
    report = build_report(conn, settings)
    assert len(report["recommendation_history"]) == 1
    assert report["recommendation_history"][0]["ticker"] == "TCS.NS"