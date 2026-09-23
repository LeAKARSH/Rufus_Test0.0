# Project: Long-Term Stock Advisory Assistant (LLM + Technical + Sentiment)

## 0. One-line summary
A local/self-hosted application that monitors a watchlist of stocks, combines
technical market data (Yahoo Finance) with news sentiment (CurrentsAPI), sends
the synthesized data to a locally/externally hosted Ollama LLM for reasoning,
and outputs **buy / hold / sell advisory signals for long-term positions
(3–6 months, up to 1 year)**. The application NEVER executes trades — it is
strictly an advisory/decision-support tool.

---

## 1. Goals & Non-Goals

### Goals
- Track a configurable watchlist of stocks (tickers).
- Pull price/volume/fundamental data from Yahoo Finance (free tier, capped at
  1000 requests/hour).
- Pull news articles per company/ticker from CurrentsAPI (free tier, capped at
  100 requests/day).
- Run sentiment analysis on retrieved news (either via the LLM itself or a
  lightweight local sentiment step feeding into the LLM).
- Feed a structured summary (technical indicators + fundamentals + sentiment
  score + recent headlines) to an LLM running on an **external Ollama server**
  (host/port/model to be supplied later — build this as a config value, not a
  hardcoded constant).
- LLM produces a recommendation: **BUY / HOLD / SELL / AVOID**, with:
  - Confidence level
  - Reasoning (grounded in the data provided, not hallucinated)
  - Suggested holding horizon (e.g., "hold through Q2 earnings", "revisit in 6 months")
  - Key risk factors and key catalysts to watch
- Persist historical recommendations and data snapshots so accuracy can be
  reviewed later (did the "buy" call age well after 3/6/12 months?).
- **Emulate trading**: maintain a virtual/paper portfolio that actually acts
  on the LLM's recommendations (simulated buys, holds, sells) using
  fictional money, so the app's own performance can be tracked over time
  as if the recommendations had been followed. This is a simulation for
  tracking hypothetical performance — not a real trade execution system
  (see Non-Goals).
- Present results via a simple dashboard/report (web UI or generated report —
  see Section 8), including the simulated portfolio's performance.
- Respect both APIs' rate limits automatically — this is a hard constraint,
  not a nice-to-have.

### Non-Goals (explicitly out of scope)
- No automatic *real* trade execution (no brokerage API integration, no real
  order placement, no real money ever moves). The "trading emulation" in
  this project is a paper-trading/simulation layer only — it exists to let
  the user see how following the tool's advice *would have* performed, and
  to give the LLM/decision engine a feedback loop, not to trade on the
  user's behalf.
- No day-trading / intraday / high-frequency signals — this is strictly a
  long-term (months-to-a-year) holding advisor.
- No guarantee/backtest-certified accuracy claims — this is a decision-support
  tool, and outputs should be labeled as such (not financial advice).

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Scheduler / Orchestrator                     │
│  (cron-like: decides what runs when, enforces rate-limit budgets)     │
└───────────────┬───────────────────────────────┬──────────────────────┘
                │                               │
      ┌─────────▼─────────┐          ┌──────────▼──────────┐
      │ Market Data Module │          │  News/Sentiment      │
      │ (Yahoo Finance)    │          │  Module (CurrentsAPI) │
      └─────────┬─────────┘          └──────────┬──────────┘
                │                               │
                └───────────────┬───────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   Data Aggregator /     │
                    │   Feature Builder       │
                    │ (technical indicators,  │
                    │  fundamentals, sentiment│
                    │  score, headline digest)│
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   LLM Decision Engine   │
                    │ (calls external Ollama  │
                    │  server w/ structured   │
                    │  prompt + JSON schema)  │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  Paper Trading /        │
                    │  Simulation Engine      │
                    │ (applies BUY/SELL calls │
                    │  to a virtual portfolio,│
                    │  tracks P&L, positions) │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  Persistence Layer      │
                    │ (SQLite/Postgres):      │
                    │  price history,         │
                    │  sentiment history,     │
                    │  recommendations log,   │
                    │  virtual portfolio &    │
                    │  simulated trade log    │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  Presentation Layer     │
                    │ (Web dashboard / CLI /  │
                    │  generated report)      │
                    └─────────────────────────┘
```

### 2.1 Scheduling & Market-Hours Awareness (Automatic, Multi-Day Operation)

The app must run **unattended, continuously, across multiple days**,
automatically waking up its data-pulling and reasoning cycles only while the
market is actually open — not on weekends/holidays, and not overnight.

- **Run mode**: this should be a long-running background process/daemon (not
  a one-shot script triggered manually), started once and left running. It
  should be resilient to being left on for days/weeks at a time.
- **Market calendar awareness**: the scheduler must know the actual trading
  calendar (regular NYSE/NASDAQ hours, e.g. 9:30 AM–4:00 PM ET on trading
  days) — including weekends and market holidays — rather than assuming a
  fixed weekday schedule. Use a maintained market-calendar library (e.g.
  `pandas_market_calendars` or equivalent) rather than hardcoding a holiday
  list, since holiday schedules can shift.
  - *(Flag for opencode: confirm which exchange/timezone matters if the
    watchlist ever includes non-US-listed tickers — for now assume US
    markets unless told otherwise.)*
- **While the market is open**: run the price/technical-data polling cycle
  on a regular intraday cadence (e.g., every 15–30 minutes — configurable),
  well within the Yahoo Finance 1000/hr budget given a reasonable watchlist
  size (see Section 3.1 math). This keeps technical snapshots reasonably
  fresh throughout the trading day without over-polling.
- **While the market is closed** (nights, weekends, holidays): the scheduler
  should idle — no price polling, no LLM calls, no wasted cycles — and
  simply wait for the next market-open window. A lightweight "next open
  time" calculation should drive an efficient sleep rather than busy-polling
  a closed market.
- **News/sentiment cadence stays independent of market hours**: since
  CurrentsAPI is capped at 100/day total (Section 3.2), that rotation can run
  once (or a few times) per calendar day regardless of intraday market
  hours — news doesn't need to track the 15–30 min price cadence.
- **LLM recommendation cadence is intentionally slower than the price-poll
  cadence**: because this is a long-term holding tool, the Decision Engine
  should NOT re-run on every 15–30 minute price tick. Recommended default:
  run the full decision pipeline (technical + sentiment → LLM →
  recommendation → simulated trade action) **once per trading day** (e.g.,
  near market open or market close), with an explicit config option to
  increase frequency later if desired. Intraday price polling still feeds
  the technical indicators and the simulated portfolio's *unrealized* P&L
  in between decision runs, but new BUY/HOLD/SELL calls and paper trades
  should only fire off the once-daily (or otherwise configured) decision
  cycle — this avoids both noisy flip-flopping and unnecessary LLM/API load.
- **Crash/restart resilience**: on restart, the scheduler must reload state
  from the persistence layer (last poll times, API usage counters, open
  simulated positions) rather than assuming a fresh start — this prevents
  duplicate simulated trades or blown rate-limit budgets after a restart.
- **Logging**: every automatic cycle (data pull, sentiment pull, LLM call,
  simulated trade action) should be logged with timestamps so multi-day
  unattended runs can be audited after the fact.

---

## 3. Data Sources & Rate-Limit Strategy

### 3.1 Yahoo Finance (Free API — 1000 requests/hour)
- Suggested library: `yfinance` (Python) or direct calls to Yahoo's
  unofficial `query1.finance.yahoo.com` / `query2.finance.yahoo.com`
  endpoints if a maintained library isn't desired. Flag this decision for
  opencode to research current best-maintained option, since these
  unofficial endpoints change occasionally.
- Data needed per ticker:
  - OHLCV (open/high/low/close/volume) — daily granularity is sufficient for
    a long-term horizon; no need for intraday data.
  - Key fundamentals: P/E ratio, EPS, market cap, dividend yield, 52-week
    high/low, sector/industry.
  - Historical price series (at least 1–2 years back) for computing moving
    averages and trend indicators.
- **Rate-limit budget math**: with N tickers in the watchlist and a limit of
  1000 req/hour, design the scheduler to:
  - Batch requests where the library/API supports multi-ticker calls.
  - Space out polling intervals (e.g., if watchlist = 20 tickers requiring 3
    calls each = 60 req per cycle, you can safely run many cycles per hour —
    but design this as a configurable "requests per ticker per cycle"
    constant so it scales if the watchlist grows).
  - Implement a token-bucket or sliding-window rate limiter as a shared
    utility, not ad-hoc sleep() calls.
  - Cache responses locally (with a TTL) so repeated requests within a cycle
    don't re-hit the API unnecessarily. For long-term holding decisions,
    price data does NOT need to be pulled more than once every few hours,
    or even once daily — this stretches the budget far beyond the cap.

### 3.2 CurrentsAPI (Free tier — 100 requests/day)
- This is the tighter constraint and must drive the design.
- Data needed: recent news articles filtered by company name / ticker /
  keyword.
- **Rate-limit budget math**: 100 requests/day total, shared across the whole
  watchlist. Strategy:
  - Do NOT query news for every ticker every day. Implement a rotation /
    prioritization scheme:
    - Tier A: tickers with upcoming catalysts (earnings date within N days,
      recent price volatility, or a pending recommendation review) get
      priority for a news pull.
    - Tier B: tickers with no near-term catalyst get checked on a rotating
      schedule (e.g., once every X days) so the whole watchlist cycles
      through within the daily budget.
  - Cache news results and their derived sentiment scores; don't re-fetch
    the same headlines.
  - Make the "queries per ticker" and "watchlist size" both config-driven so
    the rotation logic can be recalculated if the user changes the watchlist
    size later (e.g., `100 // watchlist_size` queries/ticker/day, minimum 0,
    with the priority tiering to fill gaps intelligently rather than
    wasting a query on a quiet stock).
  - Build in a hard-stop safeguard: track daily usage in persistent storage
    and refuse to fire more requests once the daily cap is reached,
    regardless of what the scheduler thinks it can spend.

### 3.3 Ollama (External Server(s))
- Connection details (host, port, model name) must be fully configurable via
  environment variables or a config file — none of this should be
  hardcoded. **Chosen models (from the user's available local Ollama
  models)**: `qwen3:32b` for the Decision Engine, `qwen3:8b` for the
  sentiment model (Section 5, option 3) — same model family for consistent
  behavior/prompt style, with the smaller variant sized appropriately for
  the higher-volume, lower-complexity sentiment task. Both remain fully
  swappable via config if the user wants to test alternatives later (e.g.,
  `gpt-oss:20b` or `gpt-oss:120b` are reasonable upgrades for the Decision
  Engine if higher reasoning quality is preferred over speed).
- The system uses **two logically separate Ollama configs**, which may point
  at the same server/model or different ones:
  1. **Decision Engine model** (`OLLAMA_HOST` / `OLLAMA_PORT` /
     `OLLAMA_MODEL`) — used for the main BUY/HOLD/SELL reasoning call
     (Section 6).
  2. **Sentiment model** (`SENTIMENT_OLLAMA_HOST` / `SENTIMENT_OLLAMA_PORT` /
     `SENTIMENT_OLLAMA_MODEL`) — used for headline sentiment scoring only
     (Section 5, option 3), if that option is enabled. Should default to
     the Decision Engine's config if left unset, so a single-model setup
     still works out of the box.
- Use Ollama's `/api/chat` or `/api/generate` endpoint (opencode should check
  current Ollama API docs, since this can change between versions).
- Design the LLM client as a single swappable module that's instantiated
  twice (once per config above) rather than writing separate code paths —
  if the user later points either one at a different model or host, no
  other code should need to change.
- Since the Decision Engine's LLM is doing the actual reasoning/decision
  step, its prompt design matters a lot (see Section 6). Both models'
  responses should be requested in strict JSON schemas so they can be
  parsed reliably and stored.

---

## 4. Technical Analysis Component

Since this is a **long-term holding** tool (not day-trading), favor
indicators suited to medium/long time horizons over short-term noise:

- Moving averages: 50-day and 200-day SMA (golden cross / death cross signal)
- Trend direction over 3, 6, 12 months
- RSI (14-day) — mainly to flag extreme overbought/oversold conditions, not
  as a primary long-term signal
- MACD — secondary confirmation signal
- Volatility (e.g., historical standard deviation of returns) — long-term
  holders care about downside risk, not daily noise
- Fundamental screen: P/E relative to sector average, dividend yield trend,
  earnings growth trend, debt levels (if obtainable from Yahoo Finance data)
- 52-week high/low positioning (is the stock near a multi-month low or high?)

This module's output should be a compact structured object per ticker (not
raw time series) that gets handed to the LLM — e.g.:

```json
{
  "ticker": "AAPL",
  "price": 227.50,
  "sma_50": 220.10,
  "sma_200": 205.30,
  "trend_signal": "golden_cross_recent",
  "rsi_14": 58.2,
  "pe_ratio": 34.1,
  "sector_avg_pe": 28.7,
  "dividend_yield": 0.44,
  "52w_high": 237.23,
  "52w_low": 164.08,
  "volatility_90d": 0.021,
  "position_vs_52w_range_pct": 76.4
}
```

---

## 5. Sentiment Analysis & News Component

- Query CurrentsAPI by company name and/or ticker-related keywords (subject
  to the 100/day budget rotation logic in Section 3.2).
- For each batch of headlines/articles returned:
  - Extract: headline, snippet/description, publish date, source.
  - Run sentiment scoring. Three options for opencode to evaluate:
    1. **Same LLM as the Decision Engine**: pass headlines directly to the
       same Ollama model as part of the same reasoning call (simplest,
       fewer moving parts, but consumes more of that model's
       context/attention per ticker, and couples sentiment scoring to
       whatever model is chosen for the heavier investment-reasoning task).
    2. **Lightweight local model**: use a small local sentiment classifier
       (e.g., VADER or a small HuggingFace model) purely to pre-score
       headlines as positive/neutral/negative, then hand the Decision
       Engine's LLM the aggregate score + top headlines. This reduces LLM
       load and gives a consistent numeric sentiment trend to store
       historically, but is not itself an LLM.
    3. **A separate, dedicated LLM for sentiment only**: run a second,
       independently configured model — e.g., a smaller/faster Ollama model
       (different `OLLAMA_MODEL`, and optionally even a different
       `OLLAMA_HOST` if you want it on separate hardware) — whose only job
       is to read headlines and return a structured sentiment score
       per article/ticker. This keeps the heavier reasoning model
       dedicated to the actual BUY/HOLD/SELL decision, while a lighter/
       cheaper model handles the higher-volume, lower-complexity sentiment
       task. This is a reasonable middle ground between options 1 and 2 —
       still LLM-based judgment (better at nuance/sarcasm/context than a
       classifier), but decoupled from the decision model's cost and
       config.
    - **Recommendation**: implement option 3 as the default — a separate,
      independently configurable sentiment LLM — since it cleanly separates
      concerns and lets each model be sized appropriately for its job
      (sentiment scoring doesn't need the same reasoning depth as the
      investment decision). Config should expose this as its own
      `SENTIMENT_OLLAMA_HOST` / `SENTIMENT_OLLAMA_MODEL` (or similar),
      distinct from the Decision Engine's `OLLAMA_HOST` / `OLLAMA_MODEL`
      (Section 3.3), defaulting to the same server/model if the user
      doesn't want to run two. Option 2 (lightweight classifier) remains a
      good fallback if the user wants to avoid a second LLM call entirely.
    - Whichever option is used, still pass the actual headline text (not
      just a number) into the Decision Engine's final prompt, so that model
      can reason about *why* sentiment is what it is — numbers alone lose
      context.
  - Store a rolling sentiment score per ticker over time (so you can later
    see "sentiment has been declining for 3 weeks" as its own signal).
- Output per ticker, e.g.:

```json
{
  "ticker": "AAPL",
  "sentiment_score_avg": 0.34,
  "sentiment_trend_7d": "improving",
  "articles_considered": 6,
  "top_headlines": [
    {"title": "...", "date": "...", "sentiment": "positive"},
    {"title": "...", "date": "...", "sentiment": "negative"}
  ]
}
```

---

## 6. LLM Decision Engine

### 6.1 Prompt Design
- System prompt should clearly instruct the model:
  - It is a long-term investment research assistant, not a day trader.
  - It must base its answer strictly on the structured data provided (no
    fabricating facts, prices, or news not given to it).
  - It must respond in a fixed JSON schema (see below) so the app can parse
    it reliably.
  - It should explicitly consider both the technical/fundamental data block
    AND the sentiment/news block, and explain how each influenced the call.
  - It should state a suggested re-evaluation horizon (e.g., "revisit in 8
    weeks" or "hold until next earnings call on [date]").

### 6.2 Expected Output Schema (example)
```json
{
  "ticker": "AAPL",
  "recommendation": "BUY",
  "confidence": "MEDIUM",
  "suggested_horizon": "6 months",
  "reasoning": "string explaining technical + sentiment rationale",
  "key_catalysts": ["upcoming earnings on 2026-10-30", "..."],
  "key_risks": ["sector-wide slowdown", "..."],
  "revisit_after": "2026-12-22"
}
```

### 6.3 Design Notes for opencode
- Build the Ollama client as its own module with a clean interface, e.g.
  `get_recommendation(ticker_data, sentiment_data) -> RecommendationObject`,
  so the rest of the app doesn't care which model/host is behind it.
- Include retry/timeout handling — local/external LLM servers can be slow or
  briefly unavailable.
- Validate the JSON response (schema validation); if the model returns
  malformed JSON, retry once with a clarifying follow-up before failing
  gracefully and logging the raw response for debugging.
- Keep the model name and connection endpoint in a config file — the user
  will supply the actual model name later.

---

## 7. Paper Trading / Simulation Engine

This is the "emulate trading" layer: it doesn't just log a recommendation, it
actually *acts* on it inside a virtual portfolio, so you can watch how the
tool's own advice would have performed with real (fictional) money over
time.

### 7.1 Core Concept
- Maintain one or more **virtual portfolios** (start with a single default
  portfolio, e.g., seeded with a configurable amount of fake cash — say
  $100,000 — but make this a config value).
- On each decision cycle, when the LLM Decision Engine outputs a
  recommendation for a ticker, the simulation engine translates that into a
  simulated action:
  - **BUY** → if not already holding the ticker (or under some max
    allocation), open a simulated position at the current (simulated) market
    price, sized according to a configurable position-sizing rule (see 7.2).
  - **HOLD** → no action; position (if any) stays open.
  - **SELL** → if a position is open, close it at the current simulated
    price and realize simulated P&L.
  - **AVOID** → no action taken; explicitly logged as "considered and
    passed."
- Since this tool is aimed at 3–6 month to 1-year holds, the simulation
  should NOT re-evaluate and flip positions on every daily price wiggle —
  it should only act when the LLM Decision Engine actually issues a new
  BUY/SELL call (which itself should be rate-limited/cadence-limited per
  Section 3, not run every few minutes).

### 7.2 Position Sizing Rules (configurable)
Give opencode a few strategies to choose from / implement as swappable
config, e.g.:
- **Equal weight**: split available virtual cash evenly across the number of
  concurrently "BUY"-rated tickers.
- **Fixed amount per position**: e.g., always simulate a $X buy per BUY
  signal, capped by available cash.
- **Confidence-weighted**: size the position based on the LLM's own reported
  confidence level (HIGH/MEDIUM/LOW) for that call.
- Whatever the default, make it a named, swappable strategy — not hardcoded
  math scattered through the codebase.

### 7.3 What the Engine Must Track
- Current virtual cash balance.
- Open positions: ticker, entry date, entry price, quantity, current
  unrealized P&L (recalculated whenever fresh price data comes in from the
  Yahoo Finance module).
- Closed positions: ticker, entry/exit dates, entry/exit prices, realized
  P&L, and — importantly — **which recommendation triggered the entry and
  which triggered the exit**, so every simulated trade traces back to the
  exact LLM reasoning behind it.
- Portfolio-level metrics over time: total value (cash + open positions'
  current value), realized/unrealized P&L, a simple equity curve (portfolio
  value over time) for charting.
- A comparison benchmark: e.g., how the same virtual cash would have
  performed in a simple buy-and-hold of an index (like SPY) over the same
  period, so the user can see whether the tool's picks are adding value
  versus just holding the market.

### 7.4 Feedback Loop (optional, later phase)
- Because every simulated trade is linked back to the recommendation that
  caused it, this data becomes a natural dataset for evaluating the
  system's own track record — e.g., "of the last 20 BUY calls, how many
  were profitable 6 months later?" This can be surfaced in the dashboard
  (Section 8) as a simple accuracy/performance scorecard, and could
  eventually be fed back into prompt refinement, though automated
  self-tuning of the LLM prompt is out of scope for v1.

### 7.5 Explicit Boundary
- This entire engine operates on **simulated cash and simulated order
  fills only**. It must not be wired to any real brokerage, exchange, or
  payment system, and no code path should exist that could place a real
  order. This boundary should be enforced architecturally (e.g., the
  simulation engine has no network egress to any trading API at all), not
  just by convention.

---

## 8. Persistence Layer

Use SQLite for simplicity (upgradeable to Postgres later if needed). Suggested tables:

- `tickers` — watchlist config (ticker, added_date, active flag, notes)
- `price_snapshots` — periodic technical/fundamental snapshots per ticker
- `news_snapshots` — headlines + sentiment scores pulled per ticker, with
  timestamps (also used to enforce "don't re-query the same day" logic)
- `recommendations` — full history of LLM outputs per ticker with timestamp,
  so you can track how a recommendation evolved and check accuracy in
  hindsight (e.g., "3 months after the BUY call, was the stock up or down?")
- `api_usage_log` — tracks Yahoo Finance and CurrentsAPI call counts per
  hour/day respectively, used by the rate limiter to self-enforce budgets
- `virtual_portfolio` — current cash balance, total value snapshot history
  (for the equity curve)
- `simulated_positions` — open and closed simulated positions: ticker,
  entry/exit dates and prices, quantity, realized/unrealized P&L, and a
  foreign key back to the `recommendations` row that triggered the entry
  and (if closed) the exit

---

## 9. Presentation Layer

Keep this simple initially; can be expanded later.

**Recommended MVP**: A lightweight local web dashboard showing:
- Watchlist table with current recommendation, confidence, last updated
  date.
- Per-ticker detail view: price chart (with SMA overlays), sentiment trend
  chart, latest LLM reasoning, headline list.
- Historical recommendation log/timeline per ticker.
- **Virtual portfolio view**: current holdings, cash balance, equity curve
  chart over time, realized/unrealized P&L, and the buy-and-hold benchmark
  comparison from Section 7.3.
- **Simulated trade log**: a table of every simulated trade with entry/exit
  dates, prices, P&L, and a link to the recommendation reasoning that
  triggered it.

**Alternative/simpler MVP**: A generated daily/periodic report (HTML or
Markdown) emailed or saved locally, if a full interactive dashboard is more
than needed initially — this should still include the virtual portfolio
summary and simulated trade log, just in report form rather than
interactive charts.

*(Flag for the user: decide whether you want an interactive dashboard or a
simpler generated report for v1 — this affects tech stack choice.)*

---

## 10. Configuration & Secrets

All of the following must be externally configurable (env vars / config
file), never hardcoded:
- `OLLAMA_HOST`, `OLLAMA_PORT`, `OLLAMA_MODEL` (Decision Engine model —
  default: `qwen3:32b`)
- `SENTIMENT_OLLAMA_HOST`, `SENTIMENT_OLLAMA_PORT`, `SENTIMENT_OLLAMA_MODEL`
  (sentiment-scoring model — default: `qwen3:8b`; falls back to the
  Decision Engine's config above if unset, Section 3.3)
- `CURRENTSAPI_KEY`
- Yahoo Finance client settings (if the chosen library needs any)
- `WATCHLIST` (list of tickers) — should be easy to edit without touching
  code
- Scheduling intervals (intraday price-poll frequency, news poll frequency,
  LLM decision-cycle frequency — see Section 2.1)
- Market calendar/exchange assumption (default: US markets, ET timezone)
- Rate-limit caps (in case the free tiers change) — 1000/hr for Yahoo,
  100/day for CurrentsAPI, stored as config, not magic numbers in code
- Simulation config: starting virtual cash balance, position-sizing
  strategy (Section 7.2), optional benchmark ticker for comparison (e.g.,
  SPY)

---

## 11. Suggested Phased Implementation Plan

**Phase 1 — Foundations**
- Project scaffolding, config system, persistence layer schema.
- Long-running scheduler/daemon with market-hours awareness (Section 2.1) —
  build this early since everything else runs on top of it.
- Yahoo Finance data module + rate limiter + caching.
- Basic technical indicator calculations.

**Phase 2 — News & Sentiment**
- CurrentsAPI integration + daily-budget rotation/prioritization logic.
- Sentiment scoring pipeline + historical tracking.

**Phase 3 — LLM Decision Engine**
- Ollama client module (configurable host/model).
- Prompt design + structured JSON output parsing/validation.
- End-to-end pipeline: data → features → LLM → stored recommendation.

**Phase 4 — Paper Trading Simulation**
- Virtual portfolio model + simulated order fill logic (Section 7).
- Position sizing strategy implementation.
- Equity curve tracking + benchmark comparison.

**Phase 5 — Presentation**
- Dashboard or report generator (per Section 9 decision), including
  portfolio/trade views.
- Historical recommendation and simulated-performance tracking/visualization.

**Phase 6 — Hardening**
- Error handling, retries, logging.
- Rate-limit safeguards tested against real API behavior.
- Architectural safeguard confirming the simulation engine has no path to
  any real trading/brokerage system (Section 7.5).
- Documentation for adding/removing tickers, changing the Ollama model, etc.

---

## 12. Open Items / Placeholders for the User to Fill In Later

- [x] ~~Ollama server host/port and model name for the Decision Engine.~~
      **Decided: `qwen3:32b`** (server host/port still to be supplied).
- [x] ~~Whether to use a separate Ollama server/model for sentiment
      scoring.~~ **Decided: yes, `qwen3:8b`** (server host/port still to be
      supplied — defaults to same server as Decision Engine unless
      specified otherwise).
- [ ] Ollama server host/port (the actual network address to reach your
      Ollama instance(s) at) — model names are set, but the connection
      details still need to be supplied.
- [ ] Initial watchlist of tickers.
- [ ] Preferred presentation layer: interactive dashboard vs. generated
      report (Section 8).
- [ ] Preferred tech stack (Python is the natural fit given yfinance/Ollama
      ecosystem, but confirm before opencode scaffolds the project).
- [ ] Hosting/runtime environment (local machine, home server, container,
      etc.) — affects scheduler implementation (cron vs. long-running
      process vs. task queue).
- [ ] Whether sentiment scoring should be LLM-only or hybrid with a local
      lightweight classifier (Section 5 recommendation).
- [ ] Starting virtual cash balance for the paper-trading portfolio.
- [ ] Preferred position-sizing strategy (equal weight / fixed amount /
      confidence-weighted — Section 7.2).
- [ ] Whether to include a benchmark comparison (e.g., SPY buy-and-hold) in
      the simulation results.

---

## 13. Disclaimers to Bake Into the Product Itself

- All outputs must be clearly labeled as automated research/decision
  support, not financial advice, and the app should never claim certainty
  about future price movement.
- The paper-trading simulation results are hypothetical performance based
  on simulated fills at reported prices (no slippage, spread, or
  commissions modeled unless explicitly added later) — the dashboard/report
  should say so plainly, so simulated returns aren't mistaken for
  achievable real-world returns.
- No real trade execution capability should ever be added to this
  codebase's scope without a deliberate, separate decision — this system,
  including its trading emulation, is simulation/advisory-only by design.
