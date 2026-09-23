"""Self-contained offline HTML report renderer (spec Section 9).

``render(data) -> str`` returns a single HTML file that works from ``file://``
with **no server, CDN, or network**: all data is inlined as JSON in one
``<script>`` tag and every chart is drawn client-side as SVG by a small
hand-rolled JS renderer. User-supplied text is written to the DOM via
``textContent`` only (no ``innerHTML`` with user data), so there is no HTML
injection surface.
"""

from __future__ import annotations

import json
from typing import Any

# The JSON payload is embedded inside a <script> tag; escaping `<`/`>` as
# unicode literals guarantees no `</script>` (nor any other markup) can break
# out, while json.loads still round-trips the original text.
def _payload(data: dict[str, Any]) -> str:
    dumped = json.dumps(data, separators=(",", ":"), ensure_ascii=True)
    return dumped.replace("<", "\\u003c").replace(">", "\\u003e")


_CSS = """
:root { --ink:#1f2937; --muted:#6b7280; --line:#e5e7eb; --accent:#2563eb; }
* { box-sizing: border-box; }
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0;
       color: var(--ink); background: #f8fafc; }
header { padding: 24px 28px; background: #fff; border-bottom: 1px solid var(--line); }
header h1 { margin: 0 0 4px; font-size: 22px; }
header p { margin: 0; color: var(--muted); font-size: 13px; }
main { padding: 24px 28px; max-width: 1180px; margin: 0 auto; }
.cards { display: flex; flex-wrap: wrap; gap: 14px; margin: 4px 0 24px; }
.card { background: #fff; border: 1px solid var(--line); border-radius: 10px;
        padding: 14px 18px; min-width: 150px; }
.card .k { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.card .v { font-size: 20px; font-weight: 600; margin-top: 4px; }
.card .v.up { color: #047857; } .card .v.down { color: #b91c1c; }
section { background: #fff; border: 1px solid var(--line); border-radius: 10px;
          padding: 18px 20px; margin-bottom: 22px; }
section h2 { margin: 0 0 14px; font-size: 16px; }
section h3 { margin: 20px 0 10px; font-size: 14px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.chart { width: 100%; }
.chart svg { width: 100%; height: auto; display: block; }
.muted { color: var(--muted); }
.pos-badge { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; }
.pos-badge.BUY { background: #dcfce7; color: #047857; }
.pos-badge.SELL { background: #fee2e2; color: #b91c1c; }
.pos-badge.HOLD { background: #fef9c3; color: #a16207; }
.pos-badge.AVOID { background: #f3f4f6; color: #374151; }
details { border-top: 1px solid var(--line); padding: 10px 0; }
details summary { cursor: pointer; font-weight: 600; }
ul { margin: 6px 0; padding-left: 20px; }
footer { color: var(--muted); font-size: 12px; padding: 8px 28px 28px; }
"""

_JS = r"""
(function () {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";

  function num(v, d) {
    if (v === null || v === undefined || isNaN(v)) return "-";
    d = d === undefined ? 2 : d;
    return v.toLocaleString("en-IN", {
      minimumFractionDigits: d, maximumFractionDigits: d,
    });
  }
  function pct(v) {
    if (v === null || v === undefined || isNaN(v)) return "n/a";
    return (v * 100 >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%";
  }
  function iso(v) { return v ? String(v).slice(0, 10) : "-"; }
  function badge(r) {
    var span = document.createElement("span");
    span.className = "pos-badge " + (r || "HOLD");
    span.textContent = r || "-";
    return span;
  }

  function table(headers, rows) {
    var t = document.createElement("table");
    var thead = document.createElement("thead");
    var htr = document.createElement("tr");
    headers.forEach(function (h) {
      var th = document.createElement("th");
      th.textContent = h;
      htr.appendChild(th);
    });
    thead.appendChild(htr);
    t.appendChild(thead);
    var tb = document.createElement("tbody");
    rows.forEach(function (row) {
      var tr = document.createElement("tr");
      row.forEach(function (c) {
        var td = document.createElement("td");
        if (c && c.nodeType === 1) { td.appendChild(c); }
        else { td.textContent = (c === null || c === undefined) ? "-" : String(c); }
        if (typeof c === "number") td.className = "num";
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    return t;
  }

  function lineChart(container, pairs, opts) {
    opts = opts || {};
    var W = container.clientWidth || 820;
    var H = opts.height || 200;
    var pad = 8;
    var n = 0;
    pairs.forEach(function (p) { n = Math.max(n, p.values.length); });
    var flat = [];
    pairs.forEach(function (p) {
      p.values.forEach(function (v) { if (v !== null && v !== undefined) flat.push(v); });
    });
    var count = 0;
    pairs.forEach(function (p) {
      p.values.forEach(function (v) { if (v !== null && v !== undefined) count += 1; });
    });
    if (count === 0) { container.textContent = "No data"; return; }
    var lo = Math.min.apply(null, flat), hi = Math.max.apply(null, flat);
    var range = (hi - lo) || 1;
    function X(i) { return n <= 1 ? pad : pad + (i * (W - 2 * pad)) / (n - 1); }
    function Y(v) { return H - pad - ((Math.min(hi, v) - lo) / range) * (H - 2 * pad); }
    var svg = document.createElementNS(SVGNS, "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("height", String(H));
    pairs.forEach(function (p) {
      var d = "";
      for (var i = 0; i < p.values.length; i++) {
        var v = p.values[i];
        if (v === null || v === undefined) continue;
        var x = X(i).toFixed(2), y = Y(v).toFixed(2);
        d += (d.length ? "L" : "M") + x + " " + y + " ";
      }
      var path = document.createElementNS(SVGNS, "path");
      path.setAttribute("d", d);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", p.color || "#2563eb");
      path.setAttribute("stroke-width", "2");
      path.setAttribute("stroke-linejoin", "round");
      path.setAttribute("stroke-linecap", "round");
      svg.appendChild(path);
      var sel = document.createElementNS(SVGNS, "text");
      sel.setAttribute("x", "6");
      sel.setAttribute("y", "12");
      sel.setAttribute("font-size", "10");
      sel.setAttribute("fill", p.color || "#2563eb");
      sel.textContent = p.name || "";
      svg.appendChild(sel);
    });
    container.appendChild(svg);
  }

  function fillForward(values) {
    var last = null, out = [];
    values.forEach(function (v) {
      if (v !== null && v !== undefined) last = v;
      out.push(last);
    });
    return out;
  }

  function index100(values) {
    var first = null, i;
    for (i = 0; i < values.length; i++) {
      if (values[i] !== null && values[i] !== undefined) { first = values[i]; break; }
    }
    if (first === null || first === 0) return values.slice();
    return values.map(function (v) {
      return (v === null || v === undefined) ? null : (v / first) * 100;
    });
  }

  function appendBulletList(parent, cls, items, prefix) {
    if (!items || !items.length) return;
    var ul = document.createElement("ul");
    ul.className = cls;
    items.forEach(function (x) {
      var li = document.createElement("li");
      li.textContent = prefix + x;
      ul.appendChild(li);
    });
    parent.appendChild(ul);
  }

  function renderCards() {
    var p = RUFUS_DATA.portfolio || {};
    var els = {
      "Total value": { v: num(p.total_value), cls: "" },
      "Cash": { v: num(p.cash), cls: "" },
      "Positions": { v: num(p.positions_value), cls: "" },
      "Portfolio return": { v: pct(p.portfolio_return),
                            cls: (p.portfolio_return || 0) >= 0 ? "up" : "down" },
      ["Benchmark (" + (RUFUS_DATA.benchmark_ticker || "n/a") + ")"]: {
        v: pct(p.benchmark_return),
        cls: (p.benchmark_return || 0) >= 0 ? "up" : "down" },
      "Total P&L": { v: num(p.total_pnl),
                     cls: (p.total_pnl || 0) >= 0 ? "up" : "down" },
    };
    var host = document.getElementById("cards");
    Object.keys(els).forEach(function (k) {
      var card = document.createElement("div");
      card.className = "card";
      var kd = document.createElement("div");
      kd.className = "k"; kd.textContent = k;
      var vd = document.createElement("div");
      vd.className = "v " + els[k].cls; vd.textContent = els[k].v;
      card.appendChild(kd); card.appendChild(vd);
      host.appendChild(card);
    });
  }

  function renderWatchlist() {
    var host = document.getElementById("watchlist");
    var rows = RUFUS_DATA.watchlist || [];
    if (!rows.length) { host.innerHTML = ""; return; }
    var data = rows.map(function (w) {
      return [ w.ticker, badge(w.recommendation), w.confidence || "-",
               iso(w.decision_date), num(w.sentiment_score), w.sentiment_trend || "-",
               num(w.price), w.held ? "yes" : "" ];
    });
    host.appendChild(table([
      "Ticker", "Recommendation", "Confidence", "Decision date", "Sentiment",
      "Trend", "Price", "Held",
    ], data));
  }

  function renderEquity() {
    var host = document.getElementById("equity");
    var curve = (RUFUS_DATA.portfolio || {}).equity_curve || [];
    if (!curve.length) { host.textContent = "No equity-curve points yet."; return; }
    var totals = curve.map(function (c) { return c.total_value; });
    var benches = fillForward(curve.map(function (c) { return c.benchmark_value; }));
    var chart = document.createElement("div");
    chart.className = "chart";
    lineChart(chart, [
      { name: "Portfolio", values: index100(totals), color: "#2563eb" },
      { name: "Benchmark", values: index100(benches), color: "#d97706" },
    ], { height: 220 });
    host.appendChild(chart);
    host.appendChild(document.createElement("p")).className = "muted";
    host.lastChild.textContent =
      "Portfolio and benchmark, each indexed to 100 at its first recorded value.";
    host.appendChild(table(
      ["Captured at", "Total value", "Cash", "Positions", "Benchmark"],
      curve.map(function (c) {
        return [iso(c.captured_at), num(c.total_value), num(c.cash),
                num(c.positions_value), num(c.benchmark_value)];
      })
    ));
  }

  function renderHoldings() {
    var host = document.getElementById("holdings");
    var open = (RUFUS_DATA.portfolio || {}).open_positions || [];
    if (!open.length) { host.textContent = "No open positions."; return; }
    host.appendChild(table([
      "Ticker", "Quantity", "Entry price", "Current price", "Market value", "Unrealized",
    ], open.map(function (p) {
      return [p.ticker, num(p.quantity, 3), num(p.entry_price), num(p.current_price),
              num(p.market_value), num(p.unrealized_pnl)];
    })));
  }

  function renderTrades() {
    var host = document.getElementById("trades");
    var trades = RUFUS_DATA.trades || [];
    if (!trades.length) { host.textContent = "No simulated trades yet."; return; }
    host.appendChild(table([
      "Ticker", "Status", "Entry date", "Entry price", "Quantity",
      "Exit date", "Exit price", "Realized P&L",
    ], trades.map(function (t) {
      return [t.ticker, t.status, iso(t.entry_date), num(t.entry_price),
              num(t.quantity, 3), iso(t.exit_date), num(t.exit_price), num(t.realized_pnl)];
    })));
    trades.forEach(function (t) {
      var d = document.createElement("details");
      var s = document.createElement("summary");
      s.textContent = t.ticker + " reasoning";
      var body = document.createElement("div");
      var parts = [];
      if (t.entry_recommendation) {
        parts.push("entry " + iso(t.entry_recommendation.run_date) + " " +
          (t.entry_recommendation.recommendation || "") + ": " +
          (t.entry_recommendation.reasoning || "—"));
      }
      if (t.exit_recommendation) {
        parts.push("exit " + iso(t.exit_recommendation.run_date) + " " +
          (t.exit_recommendation.recommendation || "") + ": " +
          (t.exit_recommendation.reasoning || "—"));
      }
      body.textContent = parts.join("; ") || "No reasoning stored.";
      d.appendChild(s); d.appendChild(body);
      host.appendChild(d);
    });
  }

  function renderScorecard() {
    var host = document.getElementById("scorecard");
    var sc = RUFUS_DATA.scorecard || {};
    host.appendChild(table(["Metric", "Value"], [
      ["Closed positions", String(sc.closed_positions || 0)],
      ["Open positions", String(sc.open_positions || 0)],
      ["Decided (closed with BUY entry)", String(sc.decided_closed || 0)],
      ["Profitable", String(sc.hits || 0)],
      ["Direction hit rate", pct(sc.hit_rate)],
      ["Average realized P&L", num(sc.avg_realized_pnl)],
      ["Total realized P&L", num(sc.total_realized_pnl)],
    ]));
    var by = sc.by_direction || {};
    var keys = Object.keys(by);
    if (keys.length) {
      host.appendChild(document.createElement("h3")).textContent = "By entry direction";
      host.appendChild(table(
        ["Entry direction", "Closed", "Hits", "Hit rate", "Realized P&L"],
        keys.map(function (k) {
          return [k, String(by[k].closed), String(by[k].hits), pct(by[k].hit_rate),
                  num(by[k].total_realized_pnl)];
        })
      ));
    }
    if (!sc.decided_closed) {
      host.appendChild(document.createElement("p")).className = "muted";
      host.lastChild.textContent =
        "No closed trade to score yet — the scorecard fills in as positions resolve.";
    }
  }

  function renderDetail(w) {
    var sec = document.createElement("section");
    sec.id = "ticker-" + w.ticker;
    var h = document.createElement("h2");
    h.textContent = w.ticker;
    sec.appendChild(h);

    if (w.price_series && w.price_series.length) {
      var price = document.createElement("div");
      price.className = "chart";
      var pk = w.price_series.map(function (p) { return p.price; });
      var s50 = w.price_series.map(function (p) { return p.sma_50; });
      var s200 = w.price_series.map(function (p) { return p.sma_200; });
      var n = Math.max(pk.length, s50.length, s200.length);
      var pad = [];
      while (pad.length < n) pad.push(null);
      function align(s) {
        var out = [];
        for (var i = 0; i < n; i++) {
          out.push(i < s.length ? s[i] : null);
        }
        return out;
      }
      var pairs = [{ name: w.ticker, values: align(pk), color: "#2563eb" }];
      if (s50.some(function (v) { return v !== null; })) {
        pairs.push({ name: "SMA50", values: align(s50), color: "#d97706" });
      }
      if (s200.some(function (v) { return v !== null; })) {
        pairs.push({ name: "SMA200", values: align(s200), color: "#7c3aed" });
      }
      lineChart(price, pairs, { height: 200 });
      sec.appendChild(price);
    }

    if (w.sentiment_series && w.sentiment_series.length) {
      var sent = document.createElement("div");
      sent.className = "chart";
      lineChart(sent, [{
        name: "Sentiment", values: w.sentiment_series.map(function (s) { return s.score; }),
        color: "#059669",
      }], { height: 130 });
      sec.appendChild(document.createElement("h3")).textContent = "Sentiment trend";
      sec.appendChild(sent);
    }

    var lr = w.latest_recommendation;
    if (lr) {
      sec.appendChild(document.createElement("h3")).textContent = "Latest call";
      var blk = document.createElement("div");
      var head = document.createElement("p");
      head.appendChild(badge(lr.recommendation));
      head.appendChild(document.createTextNode(
        " " + (lr.confidence || "-") + " · " + iso(lr.run_date) +
        (lr.model ? " · " + lr.model : "")));
      blk.appendChild(head);
      if (lr.reasoning) {
        var rg = document.createElement("p");
        rg.textContent = lr.reasoning;
        blk.appendChild(rg);
      }
      if ((lr.key_catalysts || []).length || (lr.key_risks || []).length) {
        var lists = document.createElement("div");
        appendBulletList(lists, "cat", lr.key_catalysts, "+ ");
        appendBulletList(lists, "risk", lr.key_risks, "! ");
        blk.appendChild(lists);
      }
      if (lr.revisit_after) {
        blk.appendChild(document.createElement("p")).className = "muted";
        blk.lastChild.textContent = "Revisit after " + iso(lr.revisit_after);
      }
      sec.appendChild(blk);
    }

    if (w.headlines && w.headlines.length) {
      sec.appendChild(document.createElement("h3")).textContent = "Recent headlines";
      var ul = document.createElement("ul");
      w.headlines.slice(0, 10).forEach(function (h) {
        var li = document.createElement("li");
        var a = document.createElement("a");
        a.textContent = h.title || h.url || "";
        if (h.url) { a.href = h.url; a.target = "_blank"; a.rel = "noopener"; }
        li.appendChild(a);
        if (h.published) {
          var date = document.createElement("span");
          date.className = "muted";
          date.textContent = " · " + iso(h.published);
          li.appendChild(date);
        }
        ul.appendChild(li);
      });
      sec.appendChild(ul);
    }

    if (w.recommendation_history && w.recommendation_history.length) {
      sec.appendChild(document.createElement("h3")).textContent = "Decision history";
      sec.appendChild(table(
        ["Date", "Recommendation", "Confidence"],
        w.recommendation_history.map(function (r) {
          return [iso(r.run_date), badge(r.recommendation), r.confidence || "-"];
        })
      ));
    }
    return sec;
  }

  function renderTickerDetails() {
    var host = document.getElementById("tickers");
    (RUFUS_DATA.tickers || []).forEach(function (w) {
      host.appendChild(renderDetail(w));
    });
  }

  function renderHistory() {
    var host = document.getElementById("rec-history");
    var rows = RUFUS_DATA.recommendation_history || [];
    if (!rows.length) { host.textContent = "No recommendations recorded yet."; return; }
    host.appendChild(table(
      ["Ticker", "Recommendation", "Confidence", "Date", "Model"],
      rows.slice(0, 100).map(function (r) {
        return [r.ticker, badge(r.recommendation), r.confidence || "-", iso(r.run_date), r.model || "-"];
      })
    ));
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("gen").textContent =
      "Generated " + iso(RUFUS_DATA.generated_at) + " · Portfolio: " +
      RUFUS_DATA.portfolio_name;
    renderCards();
    renderWatchlist();
    renderEquity();
    renderHoldings();
    renderTrades();
    renderScorecard();
    renderTickerDetails();
    renderHistory();
  });
})();
"""


def render(data: dict[str, Any]) -> str:
    payload = _payload(data)
    return _TEMPLATE.replace("__CSS__", _CSS).replace("__DATA__", payload).replace("__JS__", _JS)


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rufus report</title>
<style>
__CSS__
</style>
</head>
<body>
<header>
  <h1>Rufus report</h1>
  <p id="gen"></p>
</header>
<main>
  <div class="cards" id="cards"></div>

  <section>
    <h2>Watchlist</h2>
    <div id="watchlist"></div>
  </section>

  <section>
    <h2>Equity curve vs benchmark</h2>
    <div id="equity"></div>
  </section>

  <section>
    <h2>Holdings</h2>
    <div id="holdings"></div>
  </section>

  <section>
    <h2>Trade log</h2>
    <div id="trades"></div>
  </section>

  <section>
    <h2>Scorecard (measured so far)</h2>
    <div id="scorecard"></div>
  </section>

  <section>
    <h2>Per-ticker detail</h2>
    <div id="tickers"></div>
  </section>

  <section>
    <h2>Recommendation history</h2>
    <div id="rec-history"></div>
  </section>
</main>
<footer>Advisory only, not financial advice.</footer>
<script>
const RUFUS_DATA = __DATA__;
</script>
<script>
__JS__
</script>
</body>
</html>
"""