"""``python -m rufus paper`` — print the virtual portfolio's current state."""

from __future__ import annotations

import sys

import rufus.db as db
from rufus.config import Settings
from rufus.logging_config import setup_logging
from rufus.paper import paper_summary


def main() -> None:
    settings = Settings()
    setup_logging(settings)
    conn = db.connect(settings.db_path_abs)
    try:
        db.initialize_database(conn)
        summary = paper_summary(conn, settings)
    finally:
        conn.close()
    _print(summary)


def _print(s: dict) -> None:
    bench = s.get("benchmark_return")
    bench_text = f"{bench * 100:+.2f}%" if bench is not None else "n/a"
    out = [
        f"Portfolio     : {s['portfolio']}",
        f"Cash          : {s['cash']:,.2f}",
        f"Positions     : {s['positions_value']:,.2f}",
        f"Total value   : {s['total_value']:,.2f}",
        f"Realized P&L  : {s['realized_pnl']:+,.2f}",
        f"Unrealized P&L: {s['unrealized_pnl']:+,.2f}",
        f"Open positions: {len(s['open_positions'])}",
        "",
        f"Benchmark ({s.get('benchmark_ticker') or 'n/a'}) buy-hold return: {bench_text}",
        f"Equity curve points: {s.get('equity_points', 0)}",
    ]
    for pos in s.get("open_positions", []):
        out.append(
            f"  - {pos['ticker']}: {pos['quantity']:,.3f} @ {pos['entry_price']:,.2f} "
            f"(now {pos['current_price']:,.2f}, unrealized {pos['unrealized_pnl']:+,.2f})"
        )
    print("\n".join(out))
    if s.get("benchmark_return") is not None:
        print("\n  (Advisory only, not financial advice.)")


if __name__ == "__main__":
    sys.exit(main())