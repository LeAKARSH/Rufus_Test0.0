"""Tests for the ``python -m rufus ticker`` watchlist CLI."""

import pytest

import rufus.db as db
import rufus.ticker_cli as ticker_cli


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    db.initialize_database(c)
    yield c
    c.close()


def run_cli(conn, *argv):
    args = ticker_cli.build_parser().parse_args(list(argv))
    return ticker_cli.run(args, conn)


def test_add_and_list(conn):
    out = run_cli(conn, "add", "Infy.NS", "--keywords", '"Infosys" OR INFY', "--notes", "infosys")
    assert out == ["added INFY.NS (active)"]
    listed = run_cli(conn, "list")
    assert any("INFY.NS" in line and "active" in line for line in listed)
    assert any('keywords="Infosys" OR INFY' in line for line in listed)
    assert db.get_ticker_keywords(conn, "INFY.NS") == '"Infosys" OR INFY'


def test_add_is_idempotent(conn):
    run_cli(conn, "add", "TCS.NS")
    run_cli(conn, "add", "TCS.NS", "--keywords", "new kw")
    assert db.get_active_tickers(conn) == ["TCS.NS"]
    # Second add does not resurrect/duplicate and overwrites keywords when given.
    assert db.get_ticker_keywords(conn, "TCS.NS") == "new kw"


def test_active_toggle(conn):
    run_cli(conn, "add", "MSFT")
    run_cli(conn, "active", "MSFT", "false")
    assert "MSFT" not in db.get_active_tickers(conn)
    listed = run_cli(conn, "list")
    assert any("MSFT" in line and "inactive" in line for line in listed)
    run_cli(conn, "active", "MSFT")
    assert db.get_active_tickers(conn) == ["MSFT"]


def test_remove(conn):
    run_cli(conn, "add", "AAPL")
    out = run_cli(conn, "remove", "aapl")
    assert out == ["removed AAPL"]
    assert db.get_active_tickers(conn) == []
    assert run_cli(conn, "list") == ["(empty watchlist)"]


def test_seed_tickers_inserts_missing_only(conn):
    db.upsert_ticker(conn, "RELIANCE.NS")  # already known
    db.upsert_ticker(conn, "OLD.NS", active=False)  # deactivated stays untouched

    added = db.seed_tickers(conn, ["RELIANCE.NS", "TCS.NS", "OLD.NS", " NEW.NS "])
    assert added == 2  # TCS.NS + NEW.NS; RELIANCE/OLD already exist
    assert db.get_active_tickers(conn) == ["NEW.NS", "RELIANCE.NS", "TCS.NS"]
    assert "OLD.NS" not in db.get_active_tickers(conn)


def test_seed_tickers_recreates_hard_removed_when_still_configured(conn):
    # remove_ticker hard-deletes; a ticker still in WATCHLIST returns on the
    # next daemon start. Use `active false` to pause without resurrecting.
    db.upsert_ticker(conn, "GONE.NS")
    db.remove_ticker(conn, "GONE.NS")
    added = db.seed_tickers(conn, ["GONE.NS"])
    assert added == 1
    assert db.get_active_tickers(conn) == ["GONE.NS"]