"""Report query/assembly layer (spec Section 9).

Collects the data for every presentation view into plain JSON-safe dicts,
reused by the Markdown renderer, the offline HTML report, and the CLI. The
module is *read-only* over the database — it never mutates state or touches
the network — so every view is trivially testable in isolation.

Views
-----
- ``watchlist_table``: one row per active ticker: latest recommendation,
  sentiment and price.
- ``ticker_detail``: a single ticker's price/sentiment series, latest LLM
  reasoning and headline list, and recommendation timeline.
- ``portfolio_view``: the virtual portfolio (holdings, P&L, equity curve,
  benchmark comparison).
- ``trade_log``: every simulated trade joined to its triggering reasoning.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

import rufus.db as db
from rufus import scorecard
from rufus.config import Settings
from rufus.valuation import (
    benchmark_since_inception,
    portfolio_return,
    snapshot,
)

log = logging.getLogger(__name__)


def _parse_json(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return default


def _portfolio_id(conn: sqlite3.Connection, settings: Settings | None = None) -> int | None:
    """Portfolio id for read-only views; ``None`` when it does not exist yet."""
    if settings is not None:
        row = db.get_portfolio(conn, name=settings.portfolio_name)
    else:
        row = conn.execute("SELECT * FROM portfolios ORDER BY id LIMIT 1").fetchone()
    return row["id"] if row is not None else None


# --------------------------------------------------------------------------
# Watchlist
# --------------------------------------------------------------------------

def watchlist_table(
    conn: sqlite3.Connection,
    portfolio_id: int | None = None,
) -> list[dict[str, Any]]:
    """One row per active ticker: recommendation, sentiment, price, held flag."""
    rows = []
    for ticker in db.get_active_tickers(conn):
        rec = db.get_latest_recommendation(conn, ticker)
        news = db.get_news_snapshots(conn, ticker, limit=1)
        price = db.get_recent_price_snapshots(conn, ticker, limit=1)
        held = False
        if portfolio_id is not None:
            held = len(db.list_open_positions(conn, portfolio_id, ticker)) > 0
        rows.append({
            "ticker": ticker,
            "recommendation": rec["recommendation"] if rec is not None else None,
            "confidence": rec["confidence"] if rec is not None else None,
            "decision_date": rec["run_date"] if rec is not None else None,
            "model": rec["model"] if rec is not None else None,
            "sentiment_score": news[0]["sentiment_score_avg"] if news else None,
            "sentiment_trend": news[0]["sentiment_trend_7d"] if news else None,
            "sentiment_date": news[0]["run_date"] if news else None,
            "price": price[0]["price"] if price else None,
            "price_date": (price[0]["captured_at"] or "")[:10] if price else None,
            "held": held,
        })
    return rows


# --------------------------------------------------------------------------
# Ticker detail
# --------------------------------------------------------------------------

def ticker_detail(
    conn: sqlite3.Connection,
    ticker: str,
    price_limit: int = 120,
    rec_limit: int = 30,
) -> dict[str, Any]:
    """Everything the per-ticker view needs for one symbol."""
    ticker = ticker.upper()

    price_rows = db.get_recent_price_snapshots(conn, ticker, limit=price_limit)
    price_rows_desc = list(reversed(price_rows))
    price_series = [
        {
            "captured_at": r["captured_at"],
            "price": r["price"],
            "sma_50": r["sma_50"],
            "sma_200": r["sma_200"],
            "trend_signal": r["trend_signal"],
        }
        for r in price_rows_desc
    ]

    news_rows = db.get_news_snapshots(conn, ticker, limit=rec_limit)
    sentiment_series = [
        {
            "run_date": r["run_date"],
            "score": r["sentiment_score_avg"],
            "trend": r["sentiment_trend_7d"],
        }
        for r in reversed(news_rows)
    ]

    headlines: list[dict[str, Any]] = []
    if news_rows:
        parsed = _parse_json(news_rows[0]["top_headlines_json"], default=[])
        if isinstance(parsed, list):
            headlines = [
                {
                    "title": h.get("title"),
                    "published": h.get("published"),
                    "url": h.get("url"),
                    "source": h.get("source"),
                    "score": h.get("score"),
                }
                for h in parsed
                if isinstance(h, dict)
            ]

    rec = db.get_latest_recommendation(conn, ticker)
    latest_rec = None
    if rec is not None:
        latest_rec = {
            "run_date": rec["run_date"],
            "recommendation": rec["recommendation"],
            "confidence": rec["confidence"],
            "suggested_horizon": rec["suggested_horizon"],
            "reasoning": rec["reasoning"],
            "key_catalysts": _parse_json(rec["key_catalysts_json"], default=[]) or [],
            "key_risks": _parse_json(rec["key_risks_json"], default=[]) or [],
            "revisit_after": rec["revisit_after"],
            "model": rec["model"],
        }

    rec_history = [
        {
            "run_date": r["run_date"],
            "recommendation": r["recommendation"],
            "confidence": r["confidence"],
            "suggested_horizon": r["suggested_horizon"],
            "reasoning": r["reasoning"],
        }
        for r in db.get_recommendations(conn, ticker, limit=rec_limit)
    ]

    return {
        "ticker": ticker,
        "price_series": price_series,
        "latest_price": price_series[-1] if price_series else None,
        "sentiment_series": sentiment_series,
        "latest_sentiment": sentiment_series[-1] if sentiment_series else None,
        "headlines": headlines,
        "latest_recommendation": latest_rec,
        "recommendation_history": rec_history,
    }


# --------------------------------------------------------------------------
# Portfolio + benchmark
# --------------------------------------------------------------------------

def portfolio_view(conn: sqlite3.Connection, settings: Settings) -> dict[str, Any]:
    """Virtual portfolio summary, equity curve, and benchmark comparison."""
    portfolio_row = db.ensure_default_portfolio(
        conn, name=settings.portfolio_name, starting_cash=settings.starting_cash
    )
    pid = portfolio_row["id"]

    prices: dict[str, float] = {}
    for pos in db.list_open_positions(conn, pid):
        latest = db.get_recent_price_snapshots(conn, pos["ticker"], limit=1)
        if latest and latest[0]["price"] is not None:
            prices[pos["ticker"]] = float(latest[0]["price"])

    snap = snapshot(conn, pid, prices)
    curve_rows = db.get_portfolio_value_snapshots(conn, pid)
    equity_curve = [
        {
            "captured_at": r["captured_at"],
            "total_value": r["total_value"],
            "cash": r["cash"],
            "positions_value": r["positions_value"],
            "benchmark_value": r["benchmark_value"],
        }
        for r in reversed(curve_rows)
    ]

    bench = settings.benchmark_ticker if settings.benchmark_ticker else None
    return {
        "portfolio": portfolio_row["name"],
        "starting_cash": portfolio_row["starting_cash"],
        "cash": snap["cash"],
        "positions_value": snap["positions_value"],
        "total_value": snap["total_value"],
        "realized_pnl": snap["realized_pnl"],
        "unrealized_pnl": snap["unrealized_pnl"],
        "total_pnl": snap["total_pnl"],
        "open_positions": snap["open_positions"],
        "closed_positions": sum(
            1 for p in db.list_positions(conn, pid) if p["status"] == "closed"
        ),
        "equity_curve": equity_curve,
        "benchmark_ticker": bench,
        "benchmark_return": (
            benchmark_since_inception(conn, bench) if bench else None
        ),
        "portfolio_return": portfolio_return(
            float(portfolio_row["starting_cash"]), snap["total_value"]
        ),
    }


# --------------------------------------------------------------------------
# Trade log
# --------------------------------------------------------------------------

def trade_log(conn: sqlite3.Connection, settings: Settings) -> list[dict[str, Any]]:
    """Every simulated trade, linked to the reasoning that triggered it."""
    portfolio_row = db.ensure_default_portfolio(
        conn, name=settings.portfolio_name, starting_cash=settings.starting_cash
    )
    pid = portfolio_row["id"]

    rows = db.list_positions(conn, pid)
    rec_ids = set()
    for r in rows:
        if r["entry_recommendation_id"]:
            rec_ids.add(int(r["entry_recommendation_id"]))
        if r["exit_recommendation_id"]:
            rec_ids.add(int(r["exit_recommendation_id"]))

    recs: dict[int, sqlite3.Row] = {}
    if rec_ids:
        placeholders = ",".join("?" for _ in rec_ids)
        for r in conn.execute(
            f"SELECT * FROM recommendations WHERE id IN ({placeholders})",
            sorted(rec_ids),
        ).fetchall():
            recs[int(r["id"])] = r

    def rec_view(rid: int | None) -> dict[str, Any] | None:
        if rid is None or rid not in recs:
            return None
        r = recs[rid]
        return {
            "id": r["id"],
            "run_date": r["run_date"],
            "recommendation": r["recommendation"],
            "confidence": r["confidence"],
            "reasoning": r["reasoning"],
            "model": r["model"],
        }

    trades = []
    for r in rows:
        trades.append({
            "id": r["id"],
            "ticker": r["ticker"],
            "status": r["status"],
            "entry_date": r["entry_date"],
            "entry_price": r["entry_price"],
            "quantity": r["quantity"],
            "exit_date": r["exit_date"],
            "exit_price": r["exit_price"],
            "realized_pnl": r["realized_pnl"],
            "entry_recommendation": rec_view(r["entry_recommendation_id"]),
            "exit_recommendation": rec_view(r["exit_recommendation_id"]),
        })
    return trades


# --------------------------------------------------------------------------
# Combined report
# --------------------------------------------------------------------------

def recommendation_log(conn: sqlite3.Connection, limit: int = 200) -> list[dict[str, Any]]:
    """The historical recommendation log across tickers, newest first."""
    rows = conn.execute(
        "SELECT ticker, run_date, recommendation, confidence, model "
        "FROM recommendations ORDER BY run_date DESC, ticker LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def build_report(conn: sqlite3.Connection, settings: Settings) -> dict[str, Any]:
    """Assemble every view into one JSON-safe document for the renderers."""
    pid = _portfolio_id(conn, settings)
    return {
        "generated_at": db.utc_now_iso(),
        "portfolio_name": settings.portfolio_name,
        "benchmark_ticker": settings.benchmark_ticker if settings.benchmark_ticker else None,
        "watchlist": watchlist_table(conn, portfolio_id=pid),
        "portfolio": portfolio_view(conn, settings),
        "trades": trade_log(conn, settings),
        "scorecard": scorecard.compute(conn, pid) if pid is not None
        else scorecard.compute(conn, 0),
        "recommendation_history": recommendation_log(conn),
        "tickers": [
            ticker_detail(conn, t) for t in db.get_active_tickers(conn)
        ],
    }


def generate_daily_report(
    conn: sqlite3.Connection,
    settings: Settings,
    fmt: str = "html",
    force: bool = False,
    today: str | None = None,
    out: str | Path | None = None,
) -> Path:
    """Render the assembled report to a dated file; return its path.

    Restart-safe: when no ``out`` is given and today's file already exists,
    it is returned unchanged unless ``force`` is set (the scheduler generates
    at most one report per day).
    """
    ext = "html" if fmt == "html" else "md"
    if out is None:
        out_dir = settings.report_dir_path
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"report-{(today or str(date.today()))}.{ext}"
    else:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return path

    data = build_report(conn, settings)
    if fmt == "html":
        from rufus.report_html import render as render_html

        text = render_html(data)
    else:
        from rufus.report_md import render as render_md

        text = render_md(data)
    path.write_text(text, encoding="utf-8")
    log.info("report written to %s", path)
    return path