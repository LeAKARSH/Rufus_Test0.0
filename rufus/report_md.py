"""Markdown report renderer (spec Section 9, alternative/simpler MVP).

Pure function ``render(data) -> str`` over the JSON-safe document produced
by :func:`rufus.report.build_report`, so it is trivially testable and shares
nothing with the HTML renderer except the data shape.
"""

from __future__ import annotations

from typing import Any


def _num(v: Any, decimals: int = 2) -> str:
    if v is None:
        return "-"
    return f"{v:,.{decimals}f}"


def _pct(v: Any) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:+.2f}%"


def _tf(v: Any) -> str:
    if v is None:
        return "-"
    return (v[:10] if isinstance(v, str) else str(v)) or "-"


def _quant(v: Any) -> str:
    return "-" if v is None else f"{v:,.3f}"


def _table(header: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |"]
    out.append("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render(data: dict[str, Any]) -> str:
    portfolio = data.get("portfolio", {})
    lines: list[str] = [
        f"# Rufus — {data.get('portfolio_name', 'default')} report",
        "",
        "Generated "
        + _tf(data.get("generated_at"))
        + " · Benchmark: "
        + str(data.get("benchmark_ticker") or "n/a"),
        "",
    ]
    lines.append(_section_watchlist(data))
    lines.append(_section_portfolio(portfolio))
    lines.append(_section_holdings(portfolio))
    lines.append(_section_curve(portfolio))
    lines.append(_section_scorecard(data))
    lines.append(_section_trades(data))
    lines.append(_section_history(data))
    lines.append("")
    lines.append("_Advisory only, not financial advice._")
    lines.append("")
    return "\n".join(lines)


def _section_watchlist(data: dict[str, Any]) -> str:
    header = ["Ticker", "Recommendation", "Confidence", "Decision date",
              "Sentiment", "Trend", "Price", "Held"]
    rows = [
        [
            w["ticker"], w["recommendation"] or "-", w["confidence"] or "-",
            w["decision_date"] or "-", _num(w["sentiment_score"]),
            w["sentiment_trend"] or "-", _num(w["price"]),
            "yes" if w["held"] else "",
        ]
        for w in data.get("watchlist", [])
    ]
    head = "## Watchlist"
    if not rows:
        return head + "\n\n_No tickers on the watchlist._"
    return head + "\n\n" + _table(header, rows)


def _section_portfolio(portfolio: dict[str, Any]) -> str:
    head = "## Portfolio"
    metrics = [
        ("Cash", _num(portfolio.get("cash"))),
        ("Positions value", _num(portfolio.get("positions_value"))),
        ("Total value", _num(portfolio.get("total_value"))),
        ("Realized P&L", _num(portfolio.get("realized_pnl"))),
        ("Unrealized P&L", _num(portfolio.get("unrealized_pnl"))),
        ("Total P&L", _num(portfolio.get("total_pnl"))),
        ("Portfolio return", _pct(portfolio.get("portfolio_return"))),
        ("Benchmark return", _pct(portfolio.get("benchmark_return"))),
    ]
    return head + "\n\n" + _table(["Metric", "Value"], [list(r) for r in metrics])


def _section_holdings(portfolio: dict[str, Any]) -> str:
    head = "## Holdings"
    header = ["Ticker", "Quantity", "Entry price", "Current price", "Market value", "Unrealized"]
    rows = [
        [
            p["ticker"], _quant(p["quantity"]), _num(p["entry_price"]),
            _num(p["current_price"]), _num(p["market_value"]), _num(p["unrealized_pnl"]),
        ]
        for p in portfolio.get("open_positions", [])
    ]
    if not rows:
        return head + "\n\n_No open positions._"
    return head + "\n\n" + _table(header, rows)


def _section_curve(portfolio: dict[str, Any]) -> str:
    head = "## Equity curve"
    header = ["Captured at", "Total value", "Cash", "Positions", "Benchmark"]
    rows = [
        [
            _tf(v["captured_at"]), _num(v["total_value"]), _num(v["cash"]),
            _num(v["positions_value"]), _num(v["benchmark_value"]),
        ]
        for v in portfolio.get("equity_curve", [])
    ]
    if not rows:
        return head + "\n\n_No equity-curve points recorded yet._"
    return head + "\n\n" + _table(header, rows)


def _section_scorecard(data: dict[str, Any]) -> str:
    head = "## Scorecard (measured so far)"
    sc = data.get("scorecard", {})
    metrics = [
        ("Closed positions", str(sc.get("closed_positions", 0))),
        ("Open positions", str(sc.get("open_positions", 0))),
        ("Decided (closed with BUY entry)", str(sc.get("decided_closed", 0))),
        ("Profitable", str(sc.get("hits", 0))),
        ("Direction hit rate", _pct(sc.get("hit_rate"))),
        ("Average realized P&L", _num(sc.get("avg_realized_pnl"))),
        ("Total realized P&L", _num(sc.get("total_realized_pnl"))),
    ]
    body = head + "\n\n" + _table(["Metric", "Value"], [list(r) for r in metrics])
    by = sc.get("by_direction", {})
    if by:
        body += "\n\n" + _table(
            ["Entry direction", "Closed", "Hits", "Hit rate", "Realized P&L"],
            [
                [d, str(v["closed"]), str(v["hits"]), _pct(v["hit_rate"]),
                 _num(v["total_realized_pnl"])]
                for d, v in by.items()
            ],
        )
    if sc.get("decided_closed", 0) == 0:
        body += "\n\n_No closed trade to score yet — the scorecard fills in as positions resolve._"
    return body


def _section_trades(data: dict[str, Any]) -> str:
    head = "## Trade log"
    header = ["Ticker", "Status", "Entry date", "Entry price", "Quantity",
              "Exit date", "Exit price", "Realized P&L"]
    rows = []
    notes: list[str] = []
    for t in data.get("trades", []):
        rows.append([
            t["ticker"], t["status"], t["entry_date"] or "-", _num(t["entry_price"]),
            _quant(t["quantity"]), t["exit_date"] or "-", _num(t["exit_price"]),
            _num(t["realized_pnl"]),
        ])
        reasons = []
        ent = t.get("entry_recommendation") or {}
        ext = t.get("exit_recommendation") or {}
        if ent.get("reasoning"):
            reasons.append(f"entry {ent.get('run_date')} {ent.get('recommendation')}: {ent['reasoning']}")
        if ext.get("reasoning"):
            reasons.append(f"exit {ext.get('run_date')} {ext.get('recommendation')}: {ext['reasoning']}")
        if reasons:
            notes.append(f"- {t['ticker']} — " + "; ".join(reasons))
    body = head
    if not rows:
        body += "\n\n_No simulated trades yet._"
    else:
        body += "\n\n" + _table(header, rows)
    if notes:
        body += "\n\nReasoning:\n\n" + "\n".join(notes)
    return body


def _section_history(data: dict[str, Any]) -> str:
    head = "## Recommendation history"
    header = ["Ticker", "Recommendation", "Confidence", "Date", "Model"]
    rows = [
        [r["ticker"], r["recommendation"], r["confidence"] or "-",
         r["run_date"] or "-", r["model"] or "-"]
        for r in data.get("recommendation_history", [])
    ]
    if not rows:
        return head + "\n\n_No recommendations recorded yet._"
    return head + "\n\n" + _table(header, rows)