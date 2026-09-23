# Rufus — Operations Guide

Day-to-day operation of the local stock-advisory + paper-trading daemon:
watchlist, models, retries/rate limits, backups, log reading, and the
architecture that guarantees no real trading ever happens.

## The interactive shell

`python -m rufus` (no subcommand) opens a claude-code-style REPL. Run the
24/7 scheduler separately with `python -m rufus daemon`. Slash commands run
the same pipelines as the one-shot CLIs, inline:

```powershell
rufus> /status     # watchlist recency, API budgets, portfolio, schedule
rufus> /report -f md -o today.md
rufus> /paper
rufus> /ticker list
rufus> /ticker add INFY.NS --keywords '"Infosys" OR INFY'
rufus> /help        # full list; /quit (or Ctrl-D) exits
```

Log lines go to `logs/rufus.log` while the shell is open; the console stays
clean. Ctrl-C clears the current line instead of killing the session.

---

## Adding / removing tickers

The daemon seeds the watchlist from the `.env` `WATCHLIST` variable **only
for tickers not already in the database** (existing rows, including inactive
ones, are never touched). Steady-state edits use the CLI, which writes
straight to SQLite — no restart needed:

```powershell
.\.venv\Scripts\python.exe -m rufus ticker add TCS.NS
.\.venv\Scripts\python.exe -m rufus ticker add RELIANCE.NS --keywords '"Reliance Industries" OR RELIANCE'
.\.venv\Scripts\python.exe -m rufus ticker list
.\.venv\Scripts\python.exe -m rufus ticker active MSFT false   # pause without removing
.\.venv\Scripts\python.exe -m rufus ticker remove AAPL          # hard delete
```

Gotchas:

- Keywords are auto-seeded from the company's long name on the first Yahoo
  snapshot if left unset (`"Tata Consultancy Services" OR TCS`).
- `ticker remove` hard-deletes the row. If the ticker is **still listed in
  `WATCHLIST`**, it is re-added on the next daemon start. To keep the row from
  ever trading again without resurrecting it later, use `ticker active TICKER
  false` instead.
- NSE listings use the `.NS` suffix (e.g. `RELIANCE.NS`, `TCS.NS`); pass the
  exact symbol the exchange uses.

## Changing the Ollama model / host

Two independent model configs (may share a host):

| Purpose | Env vars | Default |
| --- | --- | --- |
| Decision engine | `OLLAMA_HOST`, `OLLAMA_PORT`, `OLLAMA_MODEL` | `qwen3:32b` |
| Sentiment scoring | `SENTIMENT_OLLAMA_HOST`, `SENTIMENT_OLLAMA_PORT`, `SENTIMENT_OLLAMA_MODEL` | `qwen3:8b`, falls back to decision host/port |

Edit `.env`, then restart the daemon. No code changes. (A smaller sentiment
model is the norm: it scores many headlines per news cycle.)

## Retries, timeouts, and rate limits

- **Timeouts**: `OLLAMA_TIMEOUT_SECONDS` (default 90). CurrentsAPI wire calls
  use a fixed 15s `requests` timeout; yfinance (Yahoo) uses its own
  library/urllib internals — the retry layer treats resulting
  connection/timeout errors as transient and retries them.
- **Retries**: transient transport errors and HTTP 5xx are retried with
  exponential backoff + jitter (`RETRY_BASE_DELAY_S`, `RETRY_JITTER_S`).
  Per-provider attempt caps: `YAHOO_RETRY_ATTEMPTS` (2),
  `CURRENTS_RETRY_ATTEMPTS` (1), `OLLAMA_RETRY_ATTEMPTS` (2).
- **Never retried**: CurrentsAPI HTTP 429 (quota/burst — the daily window
  can't grant more) and 401; Ollama 4xx.
- **Budgets are charged once per logical request**, never per retry.
- **Hard stops**: the persistent `api_usage_log` bucket limiter refuses a
  request before the wire once the window's budget is spent, regardless of
  what the scheduler thinks it can spend. Caps: `YAHOO_MAX_REQ_PER_HOUR`
  (1000), `CURRENTS_MAX_REQ_PER_DAY` (100).

### Rate-limit tests against the real APIs

`tests/test_live_rate_limit.py` verifies the above against the live
providers and is **skipped by default** (`RUFUS_LIVE_TESTS` unset). To run:

```powershell
$env:RUFUS_LIVE_TESTS="1"
.\.venv\Scripts\python.exe -m pytest tests\test_live_rate_limit.py -q
```

What it costs: a handful of Yahoo requests, plus **one** Currents request with
a deliberately wrong key (HTTP 401 — no usable daily quota is granted). The
zero-budget tests never touch the wire.

## The no-real-trading boundary

Section 7.5 of the specification is enforced mechanically, not by convention:

- The simulation core modules (`paper`, `simulation`, `portfolio`, `sizing`,
  `valuation`) plus the read-only reporting layer (`report`, `scorecard`) are
  proven — in a **fresh interpreter**, so transitive imports count — to never
  import an HTTP transport (`requests`, `urllib.request`, `http.client`,
  `yfinance`).
- AST checks pin the boundary: only `rufus.paper` may import a provider
  client (the Yahoo client, for live fill prices). `paper` is the single
  "contact patch"; everything else is pure math over SQLite.

If you ever intend to add real execution, it must be a deliberate, separate
decision and a new module guarded out of the simulation core. Adding a random
import today will fail `tests/test_simulation_guard.py`.

## Backing up

The entire state lives in one SQLite file (default `data/rufus.db`, WAL
mode). To back up cleanly:

```powershell
.\.venv\Scripts\python.exe -c "import sqlite3; src=sqlite3.connect(r'data\rufus.db'); dst=sqlite3.connect(r'backup\rufus-YYYYMMDD.db'); src.backup(dst); dst.close(); src.close()"
```

This copies price snapshots, sentiment, recommendations, the paper portfolio,
the equity curve, and the API-usage counters that drive the rate limiter.

## Reading logs

- Rotating file: `logs/rufus.log` (5 MB per file, 3 backups), plus stdout.
- Level via `LOG_LEVEL` (`DEBUG` shows Ollama raw replies on parse failures —
  see §6.3 of the spec; `INFO` shows cycle summaries with durations).
- Every cycle logs: start/due time, per-ticker results, spent quota, and the
  step duration ("news/decision/paper/report completed in X.Xs"). Failures are
  non-fatal per-cycle and logged with full tracebacks; a single bad ticker or
  exhausted budget never takes the daemon down.