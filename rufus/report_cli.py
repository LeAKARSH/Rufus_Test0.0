"""``python -m rufus report`` — render the daily report (Markdown or HTML).

Writes to ``REPORT_DIR/report-YYYY-MM-DD.<ext>`` by default; the scheduler's
auto-generation uses the same :func:`rufus.report.generate_daily_report`
entry point, so the on-demand CLI and the daily hook are always in sync.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import rufus.db as db
from rufus.config import Settings
from rufus.logging_config import setup_logging
from rufus.report import generate_daily_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rufus report",
        description="Generate the daily Rufus report (Markdown or offline HTML).",
    )
    p.add_argument(
        "-f", "--format", choices=("html", "md"), default="html",
        help="output format (default: html)",
    )
    p.add_argument(
        "-o", "--out", default=None,
        help="output file path; defaults to REPORT_DIR/report-YYYY-MM-DD.<ext>",
    )
    p.add_argument(
        "--date", default=None,
        help="report date (YYYY-MM-DD); defaults to today",
    )
    p.add_argument(
        "--force", action="store_true",
        help="regenerate even if today's report already exists",
    )
    p.add_argument(
        "--open", action="store_true",
        help="open the report in the default browser",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = Settings()
    setup_logging(settings)
    conn = db.connect(settings.db_path_abs)
    try:
        db.initialize_database(conn)
        path = generate_daily_report(
            conn, settings,
            fmt=args.format, force=args.force, today=args.date, out=args.out,
        )
    finally:
        conn.close()
    print(path)
    if args.open:
        webbrowser.open(Path(path).as_uri())


if __name__ == "__main__":
    sys.exit(main())