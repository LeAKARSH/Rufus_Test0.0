"""Tests for the interactive Rufus shell (slash commands, routing, /status)."""

import pytest

import rufus.db as db
import rufus.shell as shell
from rufus.config import Settings


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "shell.db")
    db.initialize_database(c)
    yield c
    c.close()


@pytest.fixture()
def settings():
    return Settings()


# --- command resolution ----------------------------------------------------


def test_resolve_command_names_and_rest():
    assert shell.resolve_command("/report -f md -o out.md") == ("report", "-f md -o out.md")
    assert shell.resolve_command("/status") == ("status", "")
    assert shell.resolve_command("/ticker   add   TCS.NS  ") == ("ticker", "add   TCS.NS")
    assert shell.resolve_command("/TICKER list") == ("ticker", "list")  # case-insensitive


def test_resolve_command_aliases():
    assert shell.resolve_command("/quit") == ("quit", "")
    assert shell.resolve_command("/exit") == ("quit", "")
    assert shell.resolve_command("/q") == ("quit", "")
    assert shell.resolve_command("/??") == ("help", "")
    assert shell.resolve_command("/ls") == ("ticker", "")


def test_resolve_command_none_for_text_and_unknown():
    assert shell.resolve_command("hello there") is None
    assert shell.resolve_command("/") is None
    assert shell.resolve_command("/bogus") is None
    assert shell.resolve_command("/reportx") is None  # prefix must match exactly


def test_is_exit():
    for word in ("quit", "exit", "q", "/quit", "/exit", "/q", "  quit  "):
        assert shell.is_exit(word)
    for word in ("/report", "report", "status"):
        assert not shell.is_exit(word)


def test_registry_is_consistent():
    assert set(shell.CMD_HELP) == set(shell.CMD_HANDLERS)
    for name in shell.CMD_HELP:
        assert shell.resolve_command(f"/{name}") == (name, "")


# --- shell object ----------------------------------------------------------


def test_prompt_shows_active_count(conn, settings):
    s = shell.Shell(settings, conn)
    assert "rufus" in s.prompt()
    assert "(0)" in s.prompt()
    db.upsert_ticker(conn, "TCS.NS")
    db.upsert_ticker(conn, "MSFT", active=False)
    assert "(1)" in s.prompt()


# --- /status rendering -----------------------------------------------------


def test_render_status_empty(conn, settings):
    lines = shell.render_status(conn, settings)
    joined = "\n".join(lines)
    assert "0 active / 0 total" in joined
    assert "api budgets" in joined
    assert "yahoo hourly" in joined and "currents daily" in joined
    assert "none yet" in joined  # no portfolio created yet
    assert settings.decision_cycle_time in joined


def test_render_status_with_watchlist_and_portfolio(conn, settings):
    db.upsert_ticker(conn, "RELIANCE.NS")
    db.upsert_ticker(conn, "TCS.NS")
    db.upsert_ticker(conn, "OLD.NS", active=False)
    portfolio = db.ensure_default_portfolio(
        conn, name=settings.portfolio_name, starting_cash=settings.starting_cash
    )
    conn.execute(
        "INSERT INTO portfolio_value_snapshots (portfolio_id, captured_at, total_value, cash) "
        "VALUES (?, ?, ?, ?)",
        (portfolio["id"], "2026-09-23T15:30:00Z", 12345.0, 7000.0),
    )
    conn.commit()

    lines = shell.render_status(conn, settings)
    joined = "\n".join(lines)
    assert "2 active / 3 total" in joined
    assert "RELIANCE.NS" in joined and "no price data yet" in joined
    assert "equity=12345" in joined and "net +2345" in joined
    assert "currents daily 0/" in joined