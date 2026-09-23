from rufus.config import (
    Settings,
    _base_url,
    get_settings,
)


def test_defaults():
    s = Settings(_env_file=None)
    assert s.ollama_host == "localhost"
    assert s.ollama_port == 11434
    assert s.ollama_model == "qwen3:32b"
    assert s.sentiment_ollama_model == "qwen3:8b"
    assert s.sentiment_ollama_host is None
    assert s.sentiment_host == s.ollama_host
    assert s.sentiment_port == s.ollama_port
    assert s.intraday_poll_minutes == 30
    assert s.market_exchange == "XNSE"
    assert s.market_timezone == "Asia/Kolkata"
    assert s.yahoo_max_req_per_hour == 1000
    assert s.currents_max_req_per_day == 100
    assert s.starting_cash == 10_000.0
    assert s.position_sizing_strategy == "equal_weight"
    assert s.benchmark_ticker == "^NSEI"
    assert s.portfolio_name == "default"
    assert s.max_allocation_pct == 100.0
    assert s.report_dir == "logs/reports"
    assert s.yahoo_retry_attempts == 2
    assert s.currents_retry_attempts == 1
    assert s.ollama_retry_attempts == 2
    assert s.retry_base_delay_s == 1.0
    assert s.retry_jitter_s == 0.25


def test_watchlist_parsing():
    s = Settings(_env_file=None, watchlist="aapl, MSFT ,tsla,,NVDA")
    assert s.watchlist == ["AAPL", "MSFT", "TSLA", "NVDA"]


def test_watchlist_empty():
    s = Settings(_env_file=None, watchlist=" , , ")
    assert s.watchlist == []


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "192.168.1.50")
    monkeypatch.setenv("OLLAMA_PORT", "11435")
    monkeypatch.setenv("CURRENTSAPI_KEY", "test-key")
    monkeypatch.setenv("WATCHLIST", "aapl")
    monkeypatch.setenv("STARTING_CASH", "50000")
    monkeypatch.setenv("YAHOO_RETRY_ATTEMPTS", "4")
    monkeypatch.setenv("RETRY_BASE_DELAY_S", "0.5")
    s = Settings(_env_file=None)
    assert s.ollama_host == "192.168.1.50"
    assert s.ollama_port == 11435
    assert s.currentsapi_key == "test-key"
    assert s.watchlist == ["AAPL"]
    assert s.starting_cash == 50_000.0
    assert s.yahoo_retry_attempts == 4
    assert s.retry_base_delay_s == 0.5


def test_sentiment_fallback_to_decision():
    s = Settings(_env_file=None, ollama_host="ollama.local", ollama_port=9999)
    assert s.sentiment_host == "ollama.local"
    assert s.sentiment_port == 9999


def test_sentiment_independent_config():
    s = Settings(
        _env_file=None,
        sentiment_ollama_host="sentiment.local",
        sentiment_ollama_port=1234,
        sentiment_ollama_model="tiny",
    )
    assert s.sentiment_host == "sentiment.local"
    assert s.sentiment_port == 1234
    assert s.sentiment_ollama_model == "tiny"


def test_base_url_with_and_without_scheme():
    assert _base_url("localhost", 11434) == "http://localhost:11434"
    assert _base_url("https://ollama.local", 11434) == "https://ollama.local:11434"


def test_get_settings_is_cached_singleton():
    assert get_settings() is get_settings()


def test_log_dir_default():
    s = Settings(_env_file=None)
    assert s.log_dir_path.name == "logs"


def test_report_dir_resolves_relative_to_project_root():
    s = Settings(_env_file=None)
    assert s.report_dir_path.is_absolute()
    assert s.report_dir_path.name == "reports"
    assert s.report_dir_path.parent.name == "logs"


def test_report_dir_env_override(monkeypatch):
    monkeypatch.setenv("REPORT_DIR", str(__import__("pathlib").Path("somewhere/else")))
    s = Settings(_env_file=None)
    assert s.report_dir_path.name == "else"