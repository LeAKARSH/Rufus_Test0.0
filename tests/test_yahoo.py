import datetime

import pandas as pd
import pytest

import rufus.db as db
import rufus.yahoo as yahoo
from rufus.config import Settings
from rufus.rate_limit import RateLimiter
from rufus.yahoo import RequestBudgetExhausted, YahooClient


def make_history(closes=(200.0, 205.0)):
    idx = pd.DatetimeIndex(
        [datetime.datetime(2026, 9, 15), datetime.datetime(2026, 9, 16)], tz="UTC"
    )
    return pd.DataFrame(
        {
            "Open": list(closes),
            "High": [c + 1.0 for c in closes],
            "Low": [c - 1.0 for c in closes],
            "Close": list(closes),
            "Volume": [1_000_000, 1_000_000],
        },
        index=idx,
    )


FAKE_INFO = {
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "marketCap": 3_500_000_000_000,
    "trailingPE": 34.1,
    "forwardPE": 30.0,
    "dividendYield": 0.44,  # yfinance reports percent already (0.44 == 0.44%)
    "earningsGrowth": 0.11,
    "revenueGrowth": 0.08,
    "fiftyTwoWeekHigh": 237.23,
    "fiftyTwoWeekLow": 164.08,
    "regularMarketPrice": 227.5,
}


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    db.initialize_database(c)
    yield c
    c.close()


@pytest.fixture()
def settings():
    return Settings(
        _env_file=None,
        yahoo_max_req_per_hour=1000,
    )


def make_client(conn, settings):
    limiter = RateLimiter("yahoo", settings.yahoo_max_req_per_hour, "hourly", conn)
    return YahooClient(limiter=limiter)


def test_fetch_snapshot_builds_compact_object(conn, settings, monkeypatch):
    monkeypatch.setattr(yahoo, "_fetch_history", lambda t, period="1y", interval="1d": make_history())
    monkeypatch.setattr(yahoo, "_fetch_info", lambda t: dict(FAKE_INFO))
    client = make_client(conn, settings)
    snap = client.fetch_snapshot("aapl")
    assert snap is not None
    assert snap["price"] == 205.0  # last history close
    assert snap["pe_ratio"] == 34.1
    assert snap["dividend_yield"] == pytest.approx(0.44)  # percent, verbatim
    # 52-week range comes from the fetched history when available.
    assert snap["high_52w"] == 205.0
    assert snap["low_52w"] == 200.0
    assert "history_tail" in snap["data_json"]


def test_fetch_snapshot_uses_cache(conn, settings, monkeypatch):
    calls = {"hist": 0, "info": 0}

    def fake_history(t, period="1y", interval="1d"):
        calls["hist"] += 1
        return make_history()

    def fake_info(t):
        calls["info"] += 1
        return dict(FAKE_INFO)

    monkeypatch.setattr(yahoo, "_fetch_history", fake_history)
    monkeypatch.setattr(yahoo, "_fetch_info", fake_info)

    client = make_client(conn, settings)
    client.fetch_snapshot("AAPL")
    client.fetch_snapshot("AAPL")
    assert calls == {"hist": 1, "info": 1}


def test_fetch_snapshot_acquires_budget(conn, settings, monkeypatch):
    # One snapshot = 2 budgeted requests (history + info).
    limiter = RateLimiter("yahoo", 2, "hourly", conn)
    client = YahooClient(limiter=limiter)
    monkeypatch.setattr(yahoo, "_fetch_history", lambda t, period="1y", interval="1d": make_history())
    monkeypatch.setattr(yahoo, "_fetch_info", lambda t: dict(FAKE_INFO))

    assert client.fetch_snapshot("AAPL") is not None  # uses the 2 available reqs
    with pytest.raises(RequestBudgetExhausted):
        client.fetch_snapshot("MSFT")


def test_exhausted_budget_does_not_call_yahoo(conn, settings, monkeypatch):
    limiter = RateLimiter("yahoo", 0, "hourly", conn)
    client = YahooClient(limiter=limiter)
    called = []

    def fake_history(t, period="1y", interval="1d"):
        called.append("hist")
        return make_history()

    monkeypatch.setattr(yahoo, "_fetch_history", fake_history)
    monkeypatch.setattr(yahoo, "_fetch_info", lambda t: dict(FAKE_INFO))
    with pytest.raises(RequestBudgetExhausted):
        client.fetch_snapshot("AAPL")
    assert called == []


def test_snapshot_none_when_no_data(conn, settings, monkeypatch):
    monkeypatch.setattr(yahoo, "_fetch_history", lambda t, period="1y", interval="1d": pd.DataFrame())
    monkeypatch.setattr(yahoo, "_fetch_info", lambda t: {})
    client = make_client(conn, settings)
    assert client.fetch_snapshot("AAPL") is None


def test_snapshot_price_falls_back_to_info(conn, settings, monkeypatch):
    monkeypatch.setattr(yahoo, "_fetch_history", lambda t, period="1y", interval="1d": pd.DataFrame())
    monkeypatch.setattr(yahoo, "_fetch_info", lambda t: {"currentPrice": 99.5})
    client = make_client(conn, settings)
    snap = client.fetch_snapshot("AAPL")
    assert snap["price"] == 99.5


def test_fetch_history_retries_transient_in_single_budget_charge(conn, settings, monkeypatch):
    calls = {"n": 0}

    def flaky(t, period="1y", interval="1d"):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("net down")
        return make_history()

    monkeypatch.setattr(yahoo, "_fetch_history", flaky)
    limiter = RateLimiter("yahoo", 1000, "hourly", conn)
    client = YahooClient(
        limiter=limiter,
        retry_attempts=2, retry_base_delay_s=0.0, retry_jitter_s=0.0,
    )
    df = client.fetch_history("AAPL")
    assert not df.empty
    assert calls["n"] == 2
    # Retry must not re-charge the budget: exactly one request was spent.
    assert db.get_api_usage(conn, "yahoo", client.limiter.current_bucket()) == 1


def test_fetch_history_non_transient_failure_not_retried(conn, settings, monkeypatch):
    calls = {"n": 0}

    def flaky(t, period="1y", interval="1d"):
        calls["n"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(yahoo, "_fetch_history", flaky)
    client = make_client(conn, settings)
    with pytest.raises(RuntimeError):
        client.fetch_history("AAPL")
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Poll function integration
# ---------------------------------------------------------------------------

def test_snapshot_embeds_indicators(monkeypatch, conn, settings):
    closes = [100.0 + i * 0.5 for i in range(300)]  # slow uptrend, 52w high
    idx = pd.DatetimeIndex(
        [datetime.datetime(2025, 1, 1) + datetime.timedelta(days=i) for i in range(300)],
        tz="UTC",
    )
    df = pd.DataFrame(
        {
            "Open": [c - 1.0 for c in closes],
            "High": [c + 1.0 for c in closes],
            "Low": [c - 1.0 for c in closes],
            "Close": closes,
            "Volume": [1000] * 300,
        },
        index=idx,
    )
    monkeypatch.setattr(yahoo, "_fetch_history", lambda t, period="1y", interval="1d": df)
    monkeypatch.setattr(yahoo, "_fetch_info", lambda t: {"trailingPE": 25.0})

    client = make_client(conn, settings)
    snap = client.fetch_snapshot("AAPL")
    assert snap["sma_50"] is not None
    assert snap["sma_200"] is not None
    assert snap["trend_signal"] == "uptrend"
    assert snap["position_vs_52w_range_pct"] == 100.0
    payload = __import__("json").loads(snap["data_json"])
    assert "macd" in payload["indicators"]
    assert set(payload["indicators"]["trends_pct"]) == {"3m", "6m", "12m"}


def test_poll_fn_stores_snapshots(conn, settings, monkeypatch):
    db.upsert_ticker(conn, "AAPL")
    db.upsert_ticker(conn, "MSFT")
    monkeypatch.setattr(yahoo, "_fetch_history", lambda t, period="1y", interval="1d": make_history())
    monkeypatch.setattr(yahoo, "_fetch_info", lambda t: {"trailingPE": 20.0})
    poll = yahoo.create_poll_fn(settings, conn)
    poll(conn, db.get_active_tickers(conn))
    rows = db.get_recent_price_snapshots(conn, "AAPL")
    assert len(rows) == 1
    assert rows[0]["pe_ratio"] == 20.0


def test_poll_fn_continues_after_ticker_failure(conn, settings, monkeypatch, caplog):
    db.upsert_ticker(conn, "AAPL")
    db.upsert_ticker(conn, "MSFT")

    def fake_history(t, period="1y", interval="1d"):
        if t == "MSFT":
            raise RuntimeError("boom")
        return make_history()

    monkeypatch.setattr(yahoo, "_fetch_history", fake_history)
    monkeypatch.setattr(yahoo, "_fetch_info", lambda t: {"trailingPE": 20.0})
    with caplog.at_level("ERROR", logger="rufus.yahoo"):
        poll = yahoo.create_poll_fn(settings, conn)
        poll(conn, db.get_active_tickers(conn))
    assert len(db.get_recent_price_snapshots(conn, "AAPL")) == 1
    assert len(db.get_recent_price_snapshots(conn, "MSFT")) == 0
    assert "yahoo budget" in caplog.text


def test_poll_fn_aborts_on_budget_exhaustion(conn, settings, monkeypatch):
    # Budget of 2: AAPL (2 calls) succeeds, MSFT aborts the pass.
    settings = Settings(_env_file=None, yahoo_max_req_per_hour=2)
    db.upsert_ticker(conn, "AAPL")
    db.upsert_ticker(conn, "MSFT")
    monkeypatch.setattr(yahoo, "_fetch_history", lambda t, period="1y", interval="1d": make_history())
    monkeypatch.setattr(yahoo, "_fetch_info", lambda t: {"trailingPE": 20.0})
    poll = yahoo.create_poll_fn(settings, conn)
    poll(conn, db.get_active_tickers(conn))
    assert len(db.get_recent_price_snapshots(conn, "AAPL")) == 1
    assert len(db.get_recent_price_snapshots(conn, "MSFT")) == 0


def test_poll_fn_seeds_keywords_from_company_name(conn, settings, monkeypatch):
    db.upsert_ticker(conn, "AAPL")
    monkeypatch.setattr(yahoo, "_fetch_history", lambda t, period="1y", interval="1d": make_history())
    monkeypatch.setattr(
        yahoo,
        "_fetch_info",
        lambda t: dict(FAKE_INFO, longName="Apple Inc."),
    )
    poll = yahoo.create_poll_fn(settings, conn)
    poll(conn, db.get_active_tickers(conn))
    assert db.get_ticker_keywords(conn, "AAPL") == '"Apple" OR AAPL'


def test_company_phrase_strips_legal_suffixes():
    assert yahoo._company_phrase("Reliance Industries Limited", "RELIANCE.NS") == "Reliance Industries"
    assert yahoo._company_phrase("Tata Consultancy Services Limited", "TCS.NS") == "Tata Consultancy Services"
    assert yahoo._plain_stem("TCS.NS") == "TCS"


def test_poll_fn_does_not_overwrite_existing_keywords(conn, settings, monkeypatch):
    db.upsert_ticker(conn, "AAPL")
    db.set_ticker_keywords(conn, "AAPL", '"My Custom Query"')
    monkeypatch.setattr(yahoo, "_fetch_history", lambda t, period="1y", interval="1d": make_history())
    monkeypatch.setattr(
        yahoo,
        "_fetch_info",
        lambda t: dict(FAKE_INFO, longName="Apple Inc."),
    )
    poll = yahoo.create_poll_fn(settings, conn)
    poll(conn, db.get_active_tickers(conn))
    assert db.get_ticker_keywords(conn, "AAPL") == '"My Custom Query"'


def test_seed_keywords_skips_missing_company_name(conn, settings, monkeypatch):
    db.upsert_ticker(conn, "AAPL")
    monkeypatch.setattr(yahoo, "_fetch_history", lambda t, period="1y", interval="1d": make_history())
    monkeypatch.setattr(yahoo, "_fetch_info", lambda t: {"trailingPE": 20.0})
    poll = yahoo.create_poll_fn(settings, conn)
    poll(conn, db.get_active_tickers(conn))
    assert db.get_ticker_keywords(conn, "AAPL") is None


# ---------------------------------------------------------------------------
# Earnings dates (news rotation input)
# ---------------------------------------------------------------------------

def test_fetch_earnings_dates_dict_list(conn, settings, monkeypatch):
    monkeypatch.setattr(
        yahoo, "_fetch_calendar",
        lambda t: {"Earnings Date": [pd.Timestamp("2026-10-30")], "Earnings High": [1.0]},
    )
    client = make_client(conn, settings)
    assert client.fetch_earnings_dates("aapl") == datetime.date(2026, 10, 30)


def test_fetch_earnings_dates_dataframe_column(conn, settings, monkeypatch):
    df = pd.DataFrame({"Earnings Date": ["2026-11-05"], "Earnings High": [2.0]})
    monkeypatch.setattr(yahoo, "_fetch_calendar", lambda t: df)
    client = make_client(conn, settings)
    assert client.fetch_earnings_dates("AAPL") == datetime.date(2026, 11, 5)


def test_fetch_earnings_dates_none_when_unknown(conn, settings, monkeypatch):
    monkeypatch.setattr(yahoo, "_fetch_calendar", lambda t: {"Ex-Dividend Date": []})
    client = make_client(conn, settings)
    assert client.fetch_earnings_dates("AAPL") is None


def test_fetch_earnings_dates_cached(conn, settings, monkeypatch):
    calls = {"n": 0}

    def fake_cal(t):
        calls["n"] += 1
        return {"Earnings Date": [pd.Timestamp("2026-10-30")]}

    monkeypatch.setattr(yahoo, "_fetch_calendar", fake_cal)
    client = make_client(conn, settings)
    client.fetch_earnings_dates("AAPL")
    client.fetch_earnings_dates("AAPL")
    assert calls["n"] == 1


def test_fetch_earnings_dates_map_omits_unknown(conn, settings, monkeypatch):
    monkeypatch.setattr(
        yahoo, "_fetch_calendar",
        lambda t: {"Earnings Date": [pd.Timestamp("2026-10-30")]} if t == "AAPL" else {},
    )
    client = make_client(conn, settings)
    out = client.fetch_earnings_dates_map(["aapl", "MSFT"])
    assert out == {"AAPL": datetime.date(2026, 10, 30)}


def test_parse_earnings_date_scalar(conn, settings, monkeypatch):
    assert yahoo._parse_earnings_date({"Earnings Date": pd.Timestamp("2026-09-25")}) == datetime.date(2026, 9, 25)


def test_fetch_benchmark_close_returns_latest_session(conn, settings, monkeypatch):
    df = make_history(closes=(25900.0, 26000.0))
    monkeypatch.setattr(yahoo, "_fetch_history", lambda t, period="1y", interval="1d": df)
    client = make_client(conn, settings)
    assert client.fetch_benchmark_close("^NSEI") == (datetime.date(2026, 9, 16), 26000.0)


def test_fetch_benchmark_close_none_on_empty_history(conn, settings, monkeypatch):
    monkeypatch.setattr(yahoo, "_fetch_history", lambda t, period="1y", interval="1d": pd.DataFrame())
    client = make_client(conn, settings)
    assert client.fetch_benchmark_close("^NSEI") is None


def test_fetch_benchmark_close_cached_within_ttl(conn, settings, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(
        yahoo, "_fetch_history",
        lambda t, period="1y", interval="1d": (calls.__setitem__("n", calls["n"] + 1) or make_history()),
    )
    client = make_client(conn, settings)
    client.fetch_benchmark_close("^NSEI")
    client.fetch_benchmark_close("^NSEI")
    assert calls["n"] == 1