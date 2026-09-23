"""Accuracy scorecard for simulated trades (spec Section 7.4).

A minimal feedback loop: among positions that were opened on a BUY
recommendation and later closed, how often did the direction pay off, and
what realized P&L resulted? Forward-looking accuracy (e.g. "was the call
profitable 6 months later") needs a filled benchmark history and is deferred
to a later phase; this module reports only what is currently measurable.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import rufus.db as db


def compute(conn: sqlite3.Connection, portfolio_id: int) -> dict[str, Any]:
    """Compute the current scorecard for a portfolio's simulated trades.

    Direction is taken from the entry recommendation's ``recommendation``
    value; only positions linked to a recommendation are scored.
    """
    rows = db.list_positions(conn, portfolio_id)
    rec_ids = set()
    for p in rows:
        if p["entry_recommendation_id"]:
            rec_ids.add(int(p["entry_recommendation_id"]))
        if p["exit_recommendation_id"]:
            rec_ids.add(int(p["exit_recommendation_id"]))

    recs: dict[int, sqlite3.Row] = {}
    if rec_ids:
        placeholders = ",".join("?" for _ in rec_ids)
        for r in conn.execute(
            f"SELECT * FROM recommendations WHERE id IN ({placeholders})",
            sorted(rec_ids),
        ).fetchall():
            recs[int(r["id"])] = r

    def direction(p) -> str | None:
        rec = recs.get(p["entry_recommendation_id"])
        return rec["recommendation"] if rec is not None else None

    closed = [p for p in rows if p["status"] == "closed"]
    open_rows = [p for p in rows if p["status"] == "open"]

    decided = [p for p in closed if direction(p) == "BUY"]
    hits = [p for p in decided if (p["realized_pnl"] or 0.0) > 0]
    avg = (
        sum(float(p["realized_pnl"]) for p in decided) / len(decided)
        if decided else None
    )

    by_direction: dict[str, dict[str, Any]] = {}
    for d in ("BUY", "SELL", "HOLD", "AVOID"):
        pool = [p for p in closed if direction(p) == d]
        if not pool:
            continue
        good = [p for p in pool if (p["realized_pnl"] or 0.0) > 0]
        by_direction[d] = {
            "closed": len(pool),
            "decided": len(pool),
            "hits": len(good),
            "hit_rate": len(good) / len(pool) if pool else None,
            "total_realized_pnl": sum(float(p["realized_pnl"]) for p in pool),
        }

    return {
        "decided_closed": len(decided),
        "hits": len(hits),
        "hit_rate": len(hits) / len(decided) if decided else None,
        "avg_realized_pnl": round(avg, 4) if avg is not None else None,
        "total_realized_pnl": sum(float(p["realized_pnl"]) for p in decided),
        "closed_positions": len(closed),
        "open_positions": len(open_rows),
        "by_direction": by_direction,
    }