"""SQLite persistence layer (spec Section 8).

Stdlib ``sqlite3`` keeps dependencies minimal; the schema is versioned so it
can migrate cleanly if a later phase grows (and remains portable to
Postgres if that is ever needed).

Phase 1 writes to ``tickers``, ``price_snapshots`` and ``api_usage_log``.
The remaining tables are created now (so later phases only add migrations)
but have no DAOs yet.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rufus.config import PROJECT_ROOT, get_settings

log = logging.getLogger(__name__)

SCHEMA_VERSION = 6

MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS tickers (
        ticker     TEXT PRIMARY KEY,
        added_date TEXT NOT NULL,
        is_active  INTEGER NOT NULL DEFAULT 1,
        notes      TEXT
    );

    CREATE TABLE IF NOT EXISTS price_snapshots (
        id                        INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker                    TEXT NOT NULL REFERENCES tickers(ticker),
        captured_at               TEXT NOT NULL,
        price                     REAL,
        sma_50                    REAL,
        sma_200                   REAL,
        trend_signal              TEXT,
        rsi_14                    REAL,
        pe_ratio                  REAL,
        sector_avg_pe             REAL,
        dividend_yield            REAL,
        high_52w                  REAL,
        low_52w                   REAL,
        volatility_90d            REAL,
        position_vs_52w_range_pct REAL,
        data_json                 TEXT,
        UNIQUE (ticker, captured_at)
    );
    CREATE INDEX IF NOT EXISTS idx_price_snapshots_ticker_time
        ON price_snapshots (ticker, captured_at DESC);

    CREATE TABLE IF NOT EXISTS news_snapshots (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker              TEXT NOT NULL REFERENCES tickers(ticker),
        run_date            TEXT NOT NULL,
        captured_at         TEXT NOT NULL,
        query_keyword       TEXT,
        sentiment_score_avg REAL,
        sentiment_trend_7d  TEXT,
        articles_considered INTEGER,
        top_headlines_json  TEXT,
        articles_json       TEXT,
        UNIQUE (ticker, run_date)
    );
    CREATE INDEX IF NOT EXISTS idx_news_snapshots_ticker_run
        ON news_snapshots (ticker, run_date DESC);

    CREATE TABLE IF NOT EXISTS recommendations (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker             TEXT NOT NULL REFERENCES tickers(ticker),
        created_at         TEXT NOT NULL,
        recommendation     TEXT NOT NULL,
        confidence         TEXT,
        suggested_horizon  TEXT,
        reasoning          TEXT,
        key_catalysts_json TEXT,
        key_risks_json     TEXT,
        revisit_after      TEXT,
        input_data_json    TEXT,
        raw_response_json  TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_recommendations_ticker_time
        ON recommendations (ticker, created_at DESC);

    CREATE TABLE IF NOT EXISTS api_usage_log (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        provider TEXT NOT NULL,
        bucket   TEXT NOT NULL,
        count    INTEGER NOT NULL DEFAULT 0,
        UNIQUE (provider, bucket)
    );

    CREATE TABLE IF NOT EXISTS portfolios (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        name          TEXT NOT NULL UNIQUE,
        starting_cash REAL NOT NULL,
        current_cash  REAL NOT NULL,
        created_at    TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS portfolio_value_snapshots (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id   INTEGER NOT NULL REFERENCES portfolios(id),
        captured_at    TEXT NOT NULL,
        total_value    REAL NOT NULL,
        cash           REAL,
        positions_value REAL,
        UNIQUE (portfolio_id, captured_at)
    );
    CREATE INDEX IF NOT EXISTS idx_value_snapshots_portfolio_time
        ON portfolio_value_snapshots (portfolio_id, captured_at DESC);

    CREATE TABLE IF NOT EXISTS simulated_positions (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id            INTEGER NOT NULL REFERENCES portfolios(id),
        ticker                  TEXT NOT NULL,
        status                  TEXT NOT NULL DEFAULT 'open'
                                CHECK (status IN ('open', 'closed')),
        entry_recommendation_id INTEGER REFERENCES recommendations(id),
        exit_recommendation_id  INTEGER REFERENCES recommendations(id),
        entry_date              TEXT NOT NULL,
        entry_price             REAL NOT NULL,
        quantity                REAL NOT NULL,
        exit_date               TEXT,
        exit_price              REAL,
        realized_pnl            REAL
    );
    CREATE INDEX IF NOT EXISTS idx_positions_portfolio_status
        ON simulated_positions (portfolio_id, status);
    """,
    2: """
    ALTER TABLE tickers ADD COLUMN search_keywords TEXT;
    """,
    3: """
    CREATE TABLE IF NOT EXISTS job_state (
        job         TEXT PRIMARY KEY,
        last_run_at TEXT NOT NULL
    );
    """,
    4: """
    ALTER TABLE recommendations ADD COLUMN run_date TEXT;
    ALTER TABLE recommendations ADD COLUMN model TEXT;
    CREATE UNIQUE INDEX IF NOT EXISTS idx_recommendations_ticker_run
        ON recommendations (ticker, run_date);
    """,
    5: """
    CREATE TABLE IF NOT EXISTS portfolio_actions (
        recommendation_id INTEGER PRIMARY KEY
                          REFERENCES recommendations(id),
        action            TEXT NOT NULL,
        acted_at          TEXT NOT NULL,
        position_id       INTEGER REFERENCES simulated_positions(id),
        note              TEXT
    );
    """,
    6: """
    ALTER TABLE portfolio_value_snapshots ADD COLUMN benchmark_value REAL;
    CREATE TABLE IF NOT EXISTS benchmark_prices (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker     TEXT NOT NULL,
        trade_date TEXT NOT NULL,
        close      REAL,
        UNIQUE (ticker, trade_date)
    );
    CREATE INDEX IF NOT EXISTS idx_benchmark_prices_ticker_date
        ON benchmark_prices (ticker, trade_date);
    """,
}

# Columns DAOs accept for price_snapshots (whitelist to avoid junk inserts).
_PRICE_SNAPSHOT_COLUMNS = (
    "price",
    "sma_50",
    "sma_200",
    "trend_signal",
    "rsi_14",
    "pe_ratio",
    "sector_avg_pe",
    "dividend_yield",
    "high_52w",
    "low_52w",
    "volatility_90d",
    "position_vs_52w_range_pct",
    "data_json",
)


def utc_now_iso() -> str:
    """Current UTC time in ISO-8601 (seconds precision) for timestamps."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a connection to the database, creating parent dirs as needed."""
    if db_path is None:
        path = get_settings().db_path_abs
    else:
        path = Path(db_path)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()

    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.Error:
        log.warning("WAL journal mode unavailable; falling back to default")
    return conn


def initialize_database(conn: sqlite3.Connection) -> None:
    """Apply any pending schema migrations (idempotent)."""
    current = _schema_version(conn)
    log.debug("database schema version %s / target %s", current, SCHEMA_VERSION)
    for version in range(current + 1, SCHEMA_VERSION + 1):
        conn.executescript(MIGRATIONS[version])
        conn.execute(f"PRAGMA user_version = {version}")
        conn.commit()
        log.info("schema migrated to version %s", version)


def _schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


# --------------------------------------------------------------------------
# tickers
# --------------------------------------------------------------------------

def upsert_ticker(
    conn: sqlite3.Connection,
    ticker: str,
    notes: str | None = None,
    active: bool = True,
) -> None:
    """Insert a ticker, or reactivate/update it if it already exists."""
    conn.execute(
        """
        INSERT INTO tickers (ticker, added_date, is_active, notes)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            is_active = excluded.is_active,
            notes = COALESCE(excluded.notes, tickers.notes)
        """,
        (ticker.upper(), utc_now_iso(), int(active), notes),
    )
    conn.commit()


def set_ticker_active(conn: sqlite3.Connection, ticker: str, active: bool) -> None:
    conn.execute(
        "UPDATE tickers SET is_active = ? WHERE ticker = ?",
        (int(active), ticker.upper()),
    )
    conn.commit()


def remove_ticker(conn: sqlite3.Connection, ticker: str) -> None:
    conn.execute("DELETE FROM tickers WHERE ticker = ?", (ticker.upper(),))
    conn.commit()


def seed_tickers(conn: sqlite3.Connection, tickers) -> int:
    """Materialize configured watchlist tickers at daemon start.

    Inserts any ticker from ``tickers`` that is *missing* from the table
    (active); rows that already exist are left untouched — including rows
    deactivated with ``set_ticker_active``, so ``active false`` is the
    persistent "keep the row, stop trading it" switch. Because ``remove_ticker``
    hard-deletes, a ticker still configured in the watchlist is re-created on the
    next daemon start; permanently drop a ticker by removing it from the
    configured ``WATCHLIST`` instead. Returns the number of tickers inserted.
    """
    added = 0
    for ticker in tickers:
        symbol = str(ticker).strip().upper()
        if not symbol:
            continue
        exists = conn.execute(
            "SELECT 1 FROM tickers WHERE ticker = ?", (symbol,)
        ).fetchone() is not None
        if exists:
            continue
        upsert_ticker(conn, symbol)
        added += 1
    return added


def get_active_tickers(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT ticker FROM tickers WHERE is_active = 1 ORDER BY ticker"
    ).fetchall()
    return [row["ticker"] for row in rows]


def set_ticker_keywords(
    conn: sqlite3.Connection,
    ticker: str,
    keywords: str | None,
) -> None:
    """Store the CurrentsAPI search-keyword string for a ticker (empty to clear)."""
    conn.execute(
        "UPDATE tickers SET search_keywords = ? WHERE ticker = ?",
        (keywords, ticker.upper()),
    )
    conn.commit()


def get_ticker_keywords(conn: sqlite3.Connection, ticker: str) -> str | None:
    """Return the ticker's stored search keywords, or ``None`` when unset."""
    row = conn.execute(
        "SELECT search_keywords FROM tickers WHERE ticker = ?",
        (ticker.upper(),),
    ).fetchone()
    if row is None:
        return None
    return row["search_keywords"]


def get_active_tickers_with_keywords(
    conn: sqlite3.Connection,
) -> list[sqlite3.Row]:
    """Active tickers that have search keywords configured, oldest-added first."""
    return conn.execute(
        "SELECT ticker, search_keywords FROM tickers "
        "WHERE is_active = 1 AND search_keywords IS NOT NULL "
        "AND TRIM(search_keywords) != '' "
        "ORDER BY added_date"
    ).fetchall()


# --------------------------------------------------------------------------
# price_snapshots
# --------------------------------------------------------------------------

def insert_price_snapshot(
    conn: sqlite3.Connection,
    ticker: str,
    captured_at: str | None = None,
    **fields: Any,
) -> None:
    """Insert a technical/fundamental snapshot for a ticker.

    Only whitelisted column names are stored; ``captured_at`` missing keys
    are ignored. Re-snapshotting the same ticker at the same timestamp
    overwrites in place.
    """
    captured_at = captured_at or utc_now_iso()
    cols = {k: v for k, v in fields.items() if k in _PRICE_SNAPSHOT_COLUMNS}
    col_names = ["ticker", "captured_at", *cols.keys()]
    values: list[Any] = [ticker.upper(), captured_at, *cols.values()]

    if cols:
        updates = ", ".join(f"{k} = excluded.{k}" for k in cols)
        sql = (
            f"INSERT INTO price_snapshots ({', '.join(col_names)}) "
            f"VALUES ({', '.join('?' * len(col_names))}) "
            f"ON CONFLICT(ticker, captured_at) DO UPDATE SET {updates}"
        )
    else:
        sql = (
            f"INSERT INTO price_snapshots ({', '.join(col_names)}) "
            f"VALUES ({', '.join('?' * len(col_names))}) "
            f"ON CONFLICT(ticker, captured_at) DO NOTHING"
        )
    conn.execute(sql, values)
    conn.commit()


def get_recent_price_snapshots(
    conn: sqlite3.Connection,
    ticker: str,
    limit: int = 100,
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM price_snapshots "
        "WHERE ticker = ? ORDER BY captured_at DESC LIMIT ?",
        (ticker.upper(), limit),
    ).fetchall()


# --------------------------------------------------------------------------
# api_usage_log
# --------------------------------------------------------------------------

def increment_api_usage(
    conn: sqlite3.Connection,
    provider: str,
    bucket: str,
    delta: int = 1,
) -> int:
    """Add ``delta`` to a provider's usage bucket and return the new total."""
    conn.execute(
        """
        INSERT INTO api_usage_log (provider, bucket, count)
        VALUES (?, ?, ?)
        ON CONFLICT(provider, bucket) DO UPDATE SET
            count = count + excluded.count
        """,
        (provider, bucket, delta),
    )
    conn.commit()
    return get_api_usage(conn, provider, bucket)


def get_api_usage(conn: sqlite3.Connection, provider: str, bucket: str) -> int:
    row = conn.execute(
        "SELECT count FROM api_usage_log WHERE provider = ? AND bucket = ?",
        (provider, bucket),
    ).fetchone()
    return int(row["count"]) if row else 0


# --------------------------------------------------------------------------
# job_state (cycle restart markers, e.g. last news-cycle run)
# --------------------------------------------------------------------------

def get_job_last_run(conn: sqlite3.Connection, job: str) -> str | None:
    row = conn.execute(
        "SELECT last_run_at FROM job_state WHERE job = ?", (job,)
    ).fetchone()
    return row["last_run_at"] if row else None


def set_job_last_run(conn: sqlite3.Connection, job: str, last_run_at: str) -> None:
    conn.execute(
        """
        INSERT INTO job_state (job, last_run_at) VALUES (?, ?)
        ON CONFLICT(job) DO UPDATE SET last_run_at = excluded.last_run_at
        """,
        (job, last_run_at),
    )
    conn.commit()


# --------------------------------------------------------------------------
# news_snapshots
# --------------------------------------------------------------------------

def insert_news_snapshot(
    conn: sqlite3.Connection,
    ticker: str,
    run_date: str | None = None,
    captured_at: str | None = None,
    query_keyword: str | None = None,
    sentiment_score_avg: float | None = None,
    sentiment_trend_7d: str | None = None,
    articles_considered: int | None = None,
    top_headlines_json: str | None = None,
    articles_json: str | None = None,
) -> None:
    """Store (or replace) a ticker's news+score snapshot for a calendar day.

    ``run_date`` is the ``YYYY-MM-DD`` key of the UNIQUE(ticker, run_date)
    constraint; re-running the same day overwrites in place (restart-safe).
    """
    run_date = run_date or str(date.today())
    captured_at = captured_at or utc_now_iso()
    conn.execute(
        """
        INSERT INTO news_snapshots (
            ticker, run_date, captured_at, query_keyword,
            sentiment_score_avg, sentiment_trend_7d,
            articles_considered, top_headlines_json, articles_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, run_date) DO UPDATE SET
            captured_at          = excluded.captured_at,
            query_keyword        = excluded.query_keyword,
            sentiment_score_avg  = excluded.sentiment_score_avg,
            sentiment_trend_7d   = excluded.sentiment_trend_7d,
            articles_considered  = excluded.articles_considered,
            top_headlines_json   = excluded.top_headlines_json,
            articles_json        = excluded.articles_json
        """,
        (
            ticker.upper(), run_date, captured_at, query_keyword,
            sentiment_score_avg, sentiment_trend_7d,
            articles_considered, top_headlines_json, articles_json,
        ),
    )
    conn.commit()


def get_news_snapshots(
    conn: sqlite3.Connection,
    ticker: str,
    limit: int = 30,
) -> list[sqlite3.Row]:
    """A ticker's news/score history, newest run first."""
    return conn.execute(
        "SELECT * FROM news_snapshots WHERE ticker = ? "
        "ORDER BY run_date DESC LIMIT ?",
        (ticker.upper(), limit),
    ).fetchall()


def get_prior_news_scores(
    conn: sqlite3.Connection,
    ticker: str,
    before_run_date: str,
    limit: int = 7,
) -> list[float]:
    """Non-empty sentiment scores from runs strictly before ``before_run_date``."""
    rows = conn.execute(
        "SELECT sentiment_score_avg FROM news_snapshots "
        "WHERE ticker = ? AND run_date < ? "
        "AND sentiment_score_avg IS NOT NULL "
        "ORDER BY run_date DESC LIMIT ?",
        (ticker.upper(), before_run_date, limit),
    ).fetchall()
    return [r["sentiment_score_avg"] for r in rows]


# --------------------------------------------------------------------------
# recommendations
# --------------------------------------------------------------------------

def insert_recommendation(
    conn: sqlite3.Connection,
    ticker: str,
    run_date: str | None = None,
    recommendation: str = "HOLD",
    confidence: str | None = None,
    suggested_horizon: str | None = None,
    reasoning: str | None = None,
    key_catalysts: Iterable[str] | None = None,
    key_risks: Iterable[str] | None = None,
    revisit_after: str | None = None,
    model: str | None = None,
    input_data_json: str | None = None,
    raw_response_json: str | None = None,
) -> None:
    """Store (or replace) one decision cycle's recommendation for a ticker.

    ``run_date`` is the ``YYYY-MM-DD`` key of the UNIQUE(ticker, run_date)
    constraint; re-running the same day overwrites in place (restart-safe).
    """
    run_date = run_date or str(date.today())
    conn.execute(
        """
        INSERT INTO recommendations (
            ticker, run_date, created_at, recommendation, confidence,
            suggested_horizon, reasoning, key_catalysts_json, key_risks_json,
            revisit_after, model, input_data_json, raw_response_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, run_date) DO UPDATE SET
            created_at          = excluded.created_at,
            recommendation      = excluded.recommendation,
            confidence          = excluded.confidence,
            suggested_horizon   = excluded.suggested_horizon,
            reasoning           = excluded.reasoning,
            key_catalysts_json  = excluded.key_catalysts_json,
            key_risks_json      = excluded.key_risks_json,
            revisit_after       = excluded.revisit_after,
            model               = excluded.model,
            input_data_json     = excluded.input_data_json,
            raw_response_json   = excluded.raw_response_json
        """,
        (
            ticker.upper(), run_date, utc_now_iso(), recommendation, confidence,
            suggested_horizon, reasoning,
            json.dumps(list(key_catalysts)) if key_catalysts else None,
            json.dumps(list(key_risks)) if key_risks else None,
            revisit_after, model, input_data_json, raw_response_json,
        ),
    )
    conn.commit()


def get_latest_recommendations(
    conn: sqlite3.Connection, tickers: Iterable[str] | None = None
) -> list[sqlite3.Row]:
    """Each ticker's most recent recommendation, oldest ticker first."""
    params = []
    if tickers:
        return conn.execute(
            "SELECT r.* FROM recommendations r "
            "JOIN (SELECT ticker, MAX(run_date) AS md FROM recommendations "
            f"WHERE ticker IN ({','.join('?' for _ in tickers)}) GROUP BY ticker) m "
            "ON r.ticker = m.ticker AND r.run_date = m.md "
            "ORDER BY r.ticker",
            [t.upper() for t in tickers],
        ).fetchall()
    return conn.execute(
        "SELECT r.* FROM recommendations r "
        "JOIN (SELECT ticker, MAX(run_date) AS md FROM recommendations GROUP BY ticker) m "
        "ON r.ticker = m.ticker AND r.run_date = m.md "
        "ORDER BY r.ticker"
    ).fetchall()


def get_latest_recommendation(conn: sqlite3.Connection, ticker: str) -> sqlite3.Row | None:
    rows = conn.execute(
        "SELECT * FROM recommendations WHERE ticker = ? ORDER BY run_date DESC LIMIT 1",
        (ticker.upper(),),
    ).fetchall()
    return rows[0] if rows else None


def get_recommendations(
    conn: sqlite3.Connection, ticker: str, limit: int = 30
) -> list[sqlite3.Row]:
    """A ticker's recommendation history, newest first."""
    return conn.execute(
        "SELECT * FROM recommendations WHERE ticker = ? ORDER BY run_date DESC LIMIT ?",
        (ticker.upper(), limit),
    ).fetchall()


# --------------------------------------------------------------------------
# portfolios / simulated_positions / portfolio_value_snapshots / actions
# --------------------------------------------------------------------------

def ensure_default_portfolio(
    conn: sqlite3.Connection,
    name: str = "default",
    starting_cash: float = 10000.0,
) -> sqlite3.Row:
    """Get (creating if needed) a virtual portfolio seeded with paper cash."""
    row = conn.execute(
        "SELECT * FROM portfolios WHERE name = ?", (name,)
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO portfolios (name, starting_cash, current_cash, created_at) "
            "VALUES (?, ?, ?, ?)",
            (name, starting_cash, starting_cash, utc_now_iso()),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM portfolios WHERE name = ?", (name,)
        ).fetchone()
    return row


def get_portfolio(
    conn: sqlite3.Connection, portfolio_id: int | None = None, name: str = "default"
) -> sqlite3.Row | None:
    if portfolio_id is not None:
        row = conn.execute(
            "SELECT * FROM portfolios WHERE id = ?", (portfolio_id,)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM portfolios WHERE name = ?", (name,)
        ).fetchone()
    return row


def update_portfolio_cash(
    conn: sqlite3.Connection, portfolio_id: int, delta: float
) -> None:
    """Adjust a portfolio's cash by ``delta`` (negative = spend)."""
    conn.execute(
        "UPDATE portfolios SET current_cash = current_cash + ? WHERE id = ?",
        (delta, portfolio_id),
    )
    conn.commit()


def open_position(
    conn: sqlite3.Connection,
    portfolio_id: int,
    ticker: str,
    entry_price: float,
    quantity: float,
    entry_recommendation_id: int | None,
    entry_date: str | None = None,
) -> int:
    """Open a simulated long position, charging cost of entry from cash.

    Returns the new position id.
    """
    entry_date = entry_date or str(date.today())
    cost = entry_price * quantity
    conn.execute(
        """
        INSERT INTO simulated_positions (
            portfolio_id, ticker, status, entry_recommendation_id,
            entry_date, entry_price, quantity
        )
        VALUES (?, ?, 'open', ?, ?, ?, ?)
        """,
        (portfolio_id, ticker.upper(), entry_recommendation_id,
         entry_date, entry_price, quantity),
    )
    position_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    conn.execute(
        "UPDATE portfolios SET current_cash = current_cash - ? WHERE id = ?",
        (cost, portfolio_id),
    )
    conn.commit()
    return position_id


def close_position(
    conn: sqlite3.Connection,
    position_id: int,
    exit_price: float,
    exit_recommendation_id: int | None = None,
    exit_date: str | None = None,
) -> float:
    """Close a simulated position, crediting proceeds and returning realized P&L."""
    exit_date = exit_date or str(date.today())
    pos = get_position(conn, position_id)
    if pos is None or pos["status"] == "closed":
        raise ValueError(f"position {position_id} not open")
    proceeds = exit_price * pos["quantity"]
    realized = (exit_price - pos["entry_price"]) * pos["quantity"]
    conn.execute(
        """
        UPDATE simulated_positions
        SET status = 'closed', exit_recommendation_id = ?,
            exit_date = ?, exit_price = ?, realized_pnl = ?
        WHERE id = ?
        """,
        (exit_recommendation_id, exit_date, exit_price, realized, position_id),
    )
    conn.execute(
        "UPDATE portfolios SET current_cash = current_cash + ? WHERE id = ?",
        (proceeds, pos["portfolio_id"]),
    )
    conn.commit()
    return realized


def get_position(conn: sqlite3.Connection, position_id: int) -> sqlite3.Row | None:
    rows = conn.execute(
        "SELECT * FROM simulated_positions WHERE id = ?", (position_id,)
    ).fetchall()
    return rows[0] if rows else None


def list_open_positions(
    conn: sqlite3.Connection, portfolio_id: int, ticker: str | None = None
) -> list[sqlite3.Row]:
    """A portfolio's open positions; optionally filtered to one ticker."""
    if ticker is not None:
        return conn.execute(
            "SELECT * FROM simulated_positions "
            "WHERE portfolio_id = ? AND status = 'open' AND ticker = ? "
            "ORDER BY entry_date, id",
            (portfolio_id, ticker.upper()),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM simulated_positions "
        "WHERE portfolio_id = ? AND status = 'open' "
        "ORDER BY entry_date, id",
        (portfolio_id,),
    ).fetchall()


def list_positions(
    conn: sqlite3.Connection, portfolio_id: int, limit: int = 100
) -> list[sqlite3.Row]:
    """A portfolio's position history (open first), newest first overall."""
    return conn.execute(
        "SELECT * FROM simulated_positions WHERE portfolio_id = ? "
        "ORDER BY (status = 'open') DESC, entry_date DESC, id DESC LIMIT ?",
        (portfolio_id, limit),
    ).fetchall()


def insert_portfolio_value_snapshot(
    conn: sqlite3.Connection,
    portfolio_id: int,
    captured_at: str,
    total_value: float,
    cash: float | None = None,
    positions_value: float | None = None,
    benchmark_value: float | None = None,
) -> None:
    """Record the portfolio's total value at a moment (idempotent per timestamp)."""
    conn.execute(
        """
        INSERT INTO portfolio_value_snapshots (
            portfolio_id, captured_at, total_value, cash, positions_value, benchmark_value
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(portfolio_id, captured_at) DO UPDATE SET
            total_value     = excluded.total_value,
            cash            = excluded.cash,
            positions_value = excluded.positions_value,
            benchmark_value = excluded.benchmark_value
        """,
        (portfolio_id, captured_at, total_value, cash, positions_value, benchmark_value),
    )
    conn.commit()


def get_portfolio_value_snapshots(
    conn: sqlite3.Connection, portfolio_id: int, limit: int = 500
) -> list[sqlite3.Row]:
    """A portfolio's equity-curve history, newest first."""
    return conn.execute(
        "SELECT * FROM portfolio_value_snapshots WHERE portfolio_id = ? "
        "ORDER BY captured_at DESC LIMIT ?",
        (portfolio_id, limit),
    ).fetchall()


def record_portfolio_action(
    conn: sqlite3.Connection,
    recommendation_id: int,
    action: str,
    position_id: int | None = None,
    note: str | None = None,
    acted_at: str | None = None,
) -> None:
    """Mark a recommendation as acted upon (idempotency guard for trading)."""
    conn.execute(
        """
        INSERT INTO portfolio_actions (recommendation_id, action, acted_at, position_id, note)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(recommendation_id) DO UPDATE SET
            action    = excluded.action,
            acted_at  = excluded.acted_at,
            position_id = COALESCE(excluded.position_id, portfolio_actions.position_id),
            note      = COALESCE(excluded.note, portfolio_actions.note)
        """,
        (recommendation_id, action, acted_at or utc_now_iso(), position_id, note),
    )
    conn.commit()


def get_portfolio_action(
    conn: sqlite3.Connection, recommendation_id: int
) -> sqlite3.Row | None:
    rows = conn.execute(
        "SELECT * FROM portfolio_actions WHERE recommendation_id = ?",
        (recommendation_id,),
    ).fetchall()
    return rows[0] if rows else None


def set_benchmark_price(
    conn: sqlite3.Connection,
    ticker: str,
    trade_date: str,
    close: float,
) -> None:
    """Store a benchmark's close for a trade date (idempotent per day)."""
    conn.execute(
        """
        INSERT INTO benchmark_prices (ticker, trade_date, close)
        VALUES (?, ?, ?)
        ON CONFLICT(ticker, trade_date) DO UPDATE SET close = excluded.close
        """,
        (ticker.upper(), trade_date, close),
    )
    conn.commit()


def get_benchmark_price(
    conn: sqlite3.Connection, ticker: str, trade_date: str
) -> sqlite3.Row | None:
    rows = conn.execute(
        "SELECT * FROM benchmark_prices WHERE ticker = ? AND trade_date = ?",
        (ticker.upper(), trade_date),
    ).fetchall()
    return rows[0] if rows else None


def get_benchmark_price_on_or_before(
    conn: sqlite3.Connection, ticker: str, trade_date: str
) -> sqlite3.Row | None:
    """Newest stored benchmark close for a ticker at or before a date."""
    rows = conn.execute(
        "SELECT * FROM benchmark_prices WHERE ticker = ? AND trade_date <= ? "
        "ORDER BY trade_date DESC LIMIT 1",
        (ticker.upper(), trade_date),
    ).fetchall()
    return rows[0] if rows else None


def get_latest_benchmark_price(
    conn: sqlite3.Connection, ticker: str
) -> sqlite3.Row | None:
    rows = conn.execute(
        "SELECT * FROM benchmark_prices WHERE ticker = ? "
        "ORDER BY trade_date DESC LIMIT 1",
        (ticker.upper(),),
    ).fetchall()
    return rows[0] if rows else None