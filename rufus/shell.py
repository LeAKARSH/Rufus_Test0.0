"""Interactive "claude-code-like" shell for Rufus.

``python -m rufus`` (no subcommand) opens a REPL with slash commands:

    rufus> /status
    rufus> /report -f md -o today.md
    rufus> /ticker add INFY.NS
    rufus> /paper
    rufus> /help

Commands run inline (blocking); ``/quit`` (or bare ``quit``/``exit``/``q``, or
Ctrl-D) leaves. Ctrl-C clears the current line and stays in the shell. Logs
continue to the rotating file; the console stays clean for the REPL.
"""

from __future__ import annotations

import os
import shlex
import sqlite3
import sys
from typing import Callable

import rufus.db as db
from rufus.config import Settings
from rufus.logging_config import setup_logging

# ANSI palette; all empty when stdout is not a terminal (pipes, tests).
if sys.stdout.isatty():
    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    CYAN = "\x1b[36m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
else:
    RESET = BOLD = DIM = CYAN = GREEN = YELLOW = RED = ""

CWD = os.path.basename(os.getcwd()) or os.getcwd()

BANNER = (
    f"{BOLD}rufus{RESET} — stock advisory + paper trading. Type {CYAN}/help{RESET}"
    " for commands, /quit to leave."
)


def resolve_command(line: str) -> tuple[str, str] | None:
    """Map ``/name rest`` to ``(name, rest)`` when ``name`` is a known command.

    Returns ``None`` for bare text, bare ``/``, and unknown slash commands, so
    the shell can print a hint. Aliases like ``/exit`` resolve to ``quit``.
    """
    if not line.startswith("/"):
        return None
    head, _, rest = line.partition(" ")
    name = head[1:].lower()
    name = _ALIASES.get(name, name)
    if name in CMD_HELP:
        return name, rest.strip()
    return None


def is_exit(line: str) -> bool:
    """True for exit words (bare or slashed) before resolve_command runs."""
    return line.strip().lower() in {"quit", "exit", "q", "/quit", "/exit", "/q"}


class Shell:
    """One REPL session. Owns a long-lived database connection (for /status)
    while slash commands run the same pipelines the one-shot CLIs run."""

    def __init__(self, settings: Settings, conn: sqlite3.Connection) -> None:
        self.settings = settings
        self.conn = conn

    # ------------------------------------------------------------------ #
    # Plumbing

    def prompt(self) -> str:
        count = len(db.get_active_tickers(self.conn))
        return f"{CYAN}rufus{RESET} {DIM}({count}){RESET} {DIM}cwd:{CWD}{RESET}> "

    def run(self) -> None:
        print(BANNER)
        print(f"{DIM}hint: logs go to {self.settings.log_dir_path / 'rufus.log'}{RESET}")
        while True:
            try:
                line = input(self.prompt())
            except EOFError:
                print()
                return
            except KeyboardInterrupt:
                print("^C")
                continue
            line = line.strip()
            if not line:
                continue
            if is_exit(line):
                return
            if resolve_command(line) is None and line.startswith("/"):
                print(f"{YELLOW}unknown command{RESET}: {line}  ({DIM}/help{RESET})")
                print()
                continue
            self._run(resolve_command(line))

    def _run(self, resolved: tuple[str, str]) -> None:
        name, rest = resolved
        print(f"{DIM}· {name}{(' ' + rest) if rest else ''}{RESET}")
        try:
            CMD_HANDLERS[name](self, rest)
        except SystemExit:
            raise
        except Exception as exc:  # keep the session alive on pipeline errors
            print(f"{RED}error{RESET}: {type(exc).__name__}: {exc}")
        print()

    # ------------------------------------------------------------------ #
    # /status

    def cmd_status(self, _rest: str) -> None:
        for line in render_status(self.conn, self.settings):
            print(line)

    # ------------------------------------------------------------------ #
    # Pass-through commands (same code the one-shot CLIs use)

    def cmd_report(self, rest: str) -> None:
        from rufus.report_cli import main as report_main

        report_main(shlex.split(rest))

    def cmd_paper(self, _rest: str) -> None:
        from rufus.paper_cli import main as paper_main

        paper_main()

    def cmd_ticker(self, rest: str) -> None:
        from rufus.ticker_cli import main as ticker_main

        ticker_main(shlex.split(rest))

    # ------------------------------------------------------------------ #
    # Utility

    def cmd_clear(self, _rest: str) -> None:
        os.system("cls" if os.name == "nt" else "clear")

    def cmd_help(self, _rest: str) -> None:
        width = max(len(name) for name in CMD_HELP) + 2
        for name, doc in CMD_HELP.items():
            print(f"  {CYAN}/{name:<{width}}{RESET}{doc}")

    def cmd_quit(self, _rest: str) -> None:
        raise SystemExit(0)


# -------------------------------------------------------------------------- #
# Command registry (kept data-only so tests can check coverage without I/O)

CMD_HELP: dict[str, str] = {
    "help": "list available commands",
    "status": "watchlist, API budgets, portfolio, schedule",
    "report": "render the daily report (flags: -f md|html -o OUT --date DATE --open)",
    "paper": "paper-trading summary",
    "ticker": "manage watchlist: list | add T [--keywords K] [--notes N] | active T [true|false] | remove T",
    "clear": "clear the terminal",
    "quit": "exit the shell",
}

_ALIASES: dict[str, str] = {"exit": "quit", "q": "quit", "ls": "ticker", "??": "help"}

CMD_HANDLERS: dict[str, Callable[[Shell, str], None]] = {
    "help": Shell.cmd_help,
    "status": Shell.cmd_status,
    "report": Shell.cmd_report,
    "paper": Shell.cmd_paper,
    "ticker": Shell.cmd_ticker,
    "clear": Shell.cmd_clear,
    "quit": Shell.cmd_quit,
}


def render_status(conn: sqlite3.Connection, settings: Settings) -> list[str]:
    """Pure-renderable /status payload: watchlist recency, budgets, portfolio."""
    from rufus.rate_limit import RateLimiter

    lines: list[str] = []

    active = db.get_active_tickers(conn)
    total = conn.execute("SELECT COUNT(*) AS n FROM tickers").fetchone()["n"]
    lines.append(
        f"{BOLD}watchlist{RESET}  {len(active)} active / {total} total"
    )
    for ticker in active:
        row = conn.execute(
            "SELECT captured_at, price FROM price_snapshots "
            "WHERE ticker = ? ORDER BY captured_at DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        if row is None:
            lines.append(f"   {ticker:<14} {"-"*14} no price data yet")
        else:
            lines.append(f"   {ticker:<14} {row['captured_at']:<26} close={row['price']}")

    yahoo = RateLimiter("yahoo", settings.yahoo_max_req_per_hour, "hourly", conn)
    currents = RateLimiter("currents", settings.currents_max_req_per_day, "daily", conn)
    y_used, y_max = yahoo.used(), yahoo.max_requests
    c_used, c_max = currents.used(), currents.max_requests
    y_color = YELLOW if y_used >= y_max else GREEN
    c_color = YELLOW if c_used >= c_max else GREEN
    lines.append(f"{BOLD}api budgets{RESET}  yahoo hourly {y_color}{y_used}/{y_max}{RESET}"
                 f"   currents daily {c_color}{c_used}/{c_max}{RESET}")

    portfolio = db.get_portfolio(conn, name=settings.portfolio_name)
    if portfolio is None:
        lines.append(f"{BOLD}portfolio{RESET}  none yet (first decision cycle seeds it)")
    else:
        value = conn.execute(
            "SELECT * FROM portfolio_value_snapshots WHERE portfolio_id = ? "
            "ORDER BY captured_at DESC LIMIT 1",
            (portfolio["id"],),
        ).fetchone()
        equity = value["total_value"] if value is not None else portfolio["current_cash"]
        net = equity - portfolio["starting_cash"]
        delta = f"+{net:.0f}" if net >= 0 else f"{net:.0f}"
        color = GREEN if net >= 0 else RED
        lines.append(f"{BOLD}portfolio{RESET}  equity={equity:.0f} "
                     f"(as of {value['captured_at'] if value is not None else 'seed'}) "
                     f"cash={portfolio['current_cash']:.0f}  "
                     f"net {color}{delta}{RESET} vs starting {portfolio['starting_cash']:.0f}")

    lines.append(f"{BOLD}schedule{RESET}  decision cycle {settings.decision_cycle_time}"
                 f" {settings.market_timezone}; intraday poll every "
                 f"{settings.intraday_poll_minutes} min")

    return lines


def shell_main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m rufus`` with no subcommand."""
    settings = Settings()
    setup_logging(settings, console=False)
    conn = db.connect(settings.db_path_abs)
    try:
        db.initialize_database(conn)
        Shell(settings, conn).run()
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(shell_main())