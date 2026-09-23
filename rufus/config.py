"""Application configuration.

Every value is externally configurable via environment variables or a
``.env`` file in the project root. Nothing here requires editing code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Single home for non-secret defaults (per spec Section 10).
DEFAULT_OLLAMA_HOST = "localhost"
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_OLLAMA_MODEL = "qwen3:32b"
DEFAULT_SENTIMENT_MODEL = "qwen3:8b"
DEFAULT_INTRADAY_POLL_MINUTES = 30
DEFAULT_NEWS_POLL_HOURS = 24
DEFAULT_DECISION_CYCLE_TIME = "15:30"
DEFAULT_EXCHANGE = "XNSE"
DEFAULT_TIMEZONE = "Asia/Kolkata"
DEFAULT_YAHOO_MAX_REQ_PER_HOUR = 1000
DEFAULT_CURRENTS_MAX_REQ_PER_DAY = 100
DEFAULT_NEWS_ROTATION_EARNINGS_WINDOW_DAYS = 14
DEFAULT_NEWS_ROTATION_OVERDUE_DAYS = 7
DEFAULT_NEWS_ROTATION_VOLATILITY_SPIKE_PCT = 50.0
DEFAULT_NEWS_SENTIMENT_MAX_ARTICLES = 15
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 90
DEFAULT_STARTING_CASH = 10_000.0
DEFAULT_POSITION_SIZING_STRATEGY = "equal_weight"
DEFAULT_BENCHMARK_TICKER = "^NSEI"
DEFAULT_PORTFOLIO_NAME = "default"
DEFAULT_MAX_ALLOCATION_PCT = 100.0
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_DIR = "logs"
DEFAULT_REPORT_DIR = "logs/reports"
DEFAULT_YAHOO_RETRY_ATTEMPTS = 2
DEFAULT_CURRENTS_RETRY_ATTEMPTS = 1
DEFAULT_OLLAMA_RETRY_ATTEMPTS = 2
DEFAULT_RETRY_BASE_DELAY_S = 1.0
DEFAULT_RETRY_JITTER_S = 0.25


class Settings(BaseSettings):
    """Reads from environment variables and an optional ``.env`` file.

    Note: ``extra="ignore"`` keeps unknown env vars from breaking startup,
    and ``case_sensitive=False`` lets e.g. ``OLLAMA_HOST`` map to
    ``ollama_host``.
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Ollama: Decision Engine ---------------------------------
    ollama_host: str = DEFAULT_OLLAMA_HOST
    ollama_port: int = DEFAULT_OLLAMA_PORT
    ollama_model: str = DEFAULT_OLLAMA_MODEL

    # --- Ollama: Sentiment scoring -------------------------------
    # Host/port fall back to the Decision Engine values when unset;
    # the model has its own default (smaller/faster variant).
    sentiment_ollama_host: str | None = None
    sentiment_ollama_port: int | None = None
    sentiment_ollama_model: str = DEFAULT_SENTIMENT_MODEL

    # --- Ollama: calls -------------------------------------------
    ollama_timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS

    # --- Data sources --------------------------------------------
    currentsapi_key: str = ""

    # --- Watchlist -----------------------------------------------
    # NoDecode stops pydantic-settings from trying JSON on the env var;
    # the before-validator handles plain comma-separated strings.
    watchlist: Annotated[list[str], NoDecode] = []

    # --- Scheduling cadences -------------------------------------
    intraday_poll_minutes: int = DEFAULT_INTRADAY_POLL_MINUTES
    news_poll_hours: int = DEFAULT_NEWS_POLL_HOURS
    decision_cycle_time: str = DEFAULT_DECISION_CYCLE_TIME

    # --- Market calendar -----------------------------------------
    market_exchange: str = DEFAULT_EXCHANGE
    market_timezone: str = DEFAULT_TIMEZONE

    # --- Rate-limit caps -----------------------------------------
    yahoo_max_req_per_hour: int = DEFAULT_YAHOO_MAX_REQ_PER_HOUR
    currents_max_req_per_day: int = DEFAULT_CURRENTS_MAX_REQ_PER_DAY

    # --- News rotation / prioritization --------------------------
    # A ticker is a Tier A catalyst when its earnings date is within this many
    # days, when its latest 90-day volatility jumped this many percent vs. the
    # prior snapshot, or when it hasn't had a news pull in this many days.
    news_rotation_earnings_window_days: int = DEFAULT_NEWS_ROTATION_EARNINGS_WINDOW_DAYS
    news_rotation_overdue_days: int = DEFAULT_NEWS_ROTATION_OVERDUE_DAYS
    news_rotation_volatility_spike_pct: float = DEFAULT_NEWS_ROTATION_VOLATILITY_SPIKE_PCT
    news_sentiment_max_articles: int = DEFAULT_NEWS_SENTIMENT_MAX_ARTICLES

    # --- Simulation / paper trading ------------------------------
    starting_cash: float = DEFAULT_STARTING_CASH
    position_sizing_strategy: str = DEFAULT_POSITION_SIZING_STRATEGY
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER
    portfolio_name: str = DEFAULT_PORTFOLIO_NAME
    # Maximum share of portfolio value any single simulated position may hold.
    max_allocation_pct: float = DEFAULT_MAX_ALLOCATION_PCT

    # --- Logging -------------------------------------------------
    log_level: str = DEFAULT_LOG_LEVEL
    log_dir: str = DEFAULT_LOG_DIR

    # --- Reporting (Phase 5) -------------------------------------
    report_dir: str = DEFAULT_REPORT_DIR

    # --- Retries (Phase 6) ----------------------------------------
    # Attempts each provider's transport/5xx failures get before giving up
    # (1 == no retry). ``currents`` never retries HTTP 429/401. Budgets are
    # charged once per logical request regardless of retry count.
    yahoo_retry_attempts: int = DEFAULT_YAHOO_RETRY_ATTEMPTS
    currents_retry_attempts: int = DEFAULT_CURRENTS_RETRY_ATTEMPTS
    ollama_retry_attempts: int = DEFAULT_OLLAMA_RETRY_ATTEMPTS
    retry_base_delay_s: float = DEFAULT_RETRY_BASE_DELAY_S
    retry_jitter_s: float = DEFAULT_RETRY_JITTER_S

    # --- Persistence ---------------------------------------------
    db_path: str = "data/rufus.db"

    @field_validator("watchlist", mode="before")
    @classmethod
    def _parse_watchlist(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [t.strip().upper() for t in value.split(",") if t.strip()]
        if isinstance(value, (list, tuple)):
            return [
                t.strip().upper()
                for t in value
                if isinstance(t, str) and t.strip()
            ]
        return value

    @field_validator("sentiment_ollama_host", "sentiment_ollama_port", mode="before")
    @classmethod
    def _empty_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    # --- Derived helpers -----------------------------------------

    @property
    def sentiment_host(self) -> str:
        return self.sentiment_ollama_host or self.ollama_host

    @property
    def sentiment_port(self) -> int:
        return self.sentiment_ollama_port or self.ollama_port

    @property
    def ollama_base_url(self) -> str:
        return _base_url(self.ollama_host, self.ollama_port)

    @property
    def sentiment_base_url(self) -> str:
        return _base_url(self.sentiment_host, self.sentiment_port)

    @property
    def log_dir_path(self) -> Path:
        return Path(self.log_dir)

    @property
    def report_dir_path(self) -> Path:
        """Absolute report directory; relative values resolve against project root."""
        p = Path(self.report_dir)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    @property
    def db_path_abs(self) -> Path:
        """Absolute DB path; relative values resolve against the project root."""
        p = Path(self.db_path)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _base_url(host: str, port: int) -> str:
    """Build an ``http(s)://host:port`` URL, tolerating a scheme in host."""
    if "://" in host:
        return f"{host}:{port}"
    return f"http://{host}:{port}"


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached settings instance."""
    return Settings()