"""``python -m rufus ticker`` — manage the watchlist.

Add/remove/activate tickers directly in the persistence layer, so the
watchlist can change without touching configuration. The ``WATCHLIST`` env
var seeds missing tickers when the daemon starts; this CLI is the steady-state
way to manage the live watchlist.

Examples::

    python -m rufus ticker list
    python -m rufus ticker add INFY.NS --notes "Infosys"
    python -m rufus ticker add TCS.NS --keywords '"Tata Consultancy Services" OR TCS'
    python -m rufus ticker active RELIANCE.NS false
    python -m rufus ticker remove MSFT
"""

from __future__ import annotations

import argparse
import sys

import rufus.db as db
from rufus.config import Settings
from rufus.logging_config import setup_logging

_TRUE_FALSE = ("true", "false")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rufus ticker",
        description="Manage the watchlist in the persistence layer.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="add a ticker (idempotent upsert)")
    add.add_argument("ticker", help="e.g. RELIANCE.NS or AAPL")
    add.add_argument("--keywords", default=None,
                     help="CurrentsAPI search string, e.g. '\"Tata Consultancy Services\" OR TCS'")
    add.add_argument("--notes", default=None, help="optional human note")

    remove = sub.add_parser("remove", help="remove a ticker entirely")
    remove.add_argument("ticker")

    active = sub.add_parser("active", help="set a ticker active or inactive")
    active.add_argument("ticker")
    active.add_argument(
        "flag", nargs="?", default="true", choices=_TRUE_FALSE,
        help="true (default) or false",
    )

    sub.add_parser("list", help="list all tickers with their status/keywords")

    return p


def run(args: argparse.Namespace, conn) -> list[str]:
    """Execute a parsed command against ``conn``; returns output lines."""
    command = args.command
    lines: list[str] = []

    if command == "add":
        db.upsert_ticker(conn, args.ticker, notes=args.notes)
        if args.keywords is not None:
            db.set_ticker_keywords(conn, args.ticker, args.keywords)
        lines.append(f"added {args.ticker.upper()} (active)")
    elif command == "remove":
        db.remove_ticker(conn, args.ticker)
        lines.append(f"removed {args.ticker.upper()}")
    elif command == "active":
        active = args.flag == "true"
        db.set_ticker_active(conn, args.ticker, active)
        lines.append(
            f"{args.ticker.upper()} is now {'active' if active else 'inactive'}"
        )
    elif command == "list":
        rows = conn.execute(
            "SELECT ticker, is_active, search_keywords, notes "
            "FROM tickers ORDER BY ticker"
        ).fetchall()
        if not rows:
            lines.append("(empty watchlist)")
        for row in rows:
            state = "active" if row["is_active"] else "inactive"
            keywords = row["search_keywords"] or "-"
            notes = row["notes"] or "-"
            lines.append(f"{row['ticker']:<14} {state:<9} keywords={keywords} notes={notes}")
    else:  # pragma: no cover - argparse enforces the choices
        raise SystemExit(f"unknown command: {command}")

    return lines


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = Settings()
    setup_logging(settings)
    conn = db.connect(settings.db_path_abs)
    try:
        db.initialize_database(conn)
        lines = run(args, conn)
    finally:
        conn.close()
    for line in lines:
        print(line)


if __name__ == "__main__":
    sys.exit(main())