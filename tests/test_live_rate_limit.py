"""Rate-limit safeguards tested against real API behavior (Phase 6).

These exercises talk to the *real* providers, so the whole module is SKIPPED
unless ``RUFUS_LIVE_TESTS=1`` is set. The default suite stays fast and fully
offline.

Budget notes (free tiers):
- Yahoo Finance: 1000 req/hour. The snapshot test spends a handful of
  requests (history + info for one ticker) and asserts the retry/cache
  behavior observed against the live endpoint.
- CurrentsAPI: 100 req/day, resets 00:00 UTC. The auth test sends exactly ONE
  request with a deliberately wrong key; a rejected key does not grant usable
  quota, so this costs nothing usable for the day. The zero-budget tests
  assert the hard-stop protective behaviour and never touch the wire.

Run:
    RUFUS_LIVE_TESTS=1 python -m pytest tests/test_live_rate_limit.py -q
"""

import os

import pytest

import rufus.db as db
from rufus.config import Settings
from rufus.currents import CurrentsAuthError, CurrentsClient
from rufus.rate_limit import BudgetExhausted, RateLimiter
from rufus.yahoo import YahooClient

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUFUS_LIVE_TESTS"),
    reason="real-API checks; set RUFUS_LIVE_TESTS=1 to run",
)


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    db.initialize_database(c)
    yield c
    c.close()


def test_yahoo_snapshot_usage_and_cache(conn):
    """One real snapshot costs history+info; a second call is cache-served."""
    settings = Settings()
    limiter = RateLimiter("yahoo", settings.yahoo_max_req_per_hour, "hourly", conn)
    client = YahooClient(limiter=limiter)

    snap = client.fetch_snapshot("RELIANCE.NS")
    assert snap is not None, "no live price data for RELIANCE.NS"
    used_after_first = limiter.used()
    assert used_after_first >= 2  # history + info both hit the wire

    client.fetch_snapshot("RELIANCE.NS")  # within the TTL -> served from cache
    assert limiter.used() == used_after_first, (
        "second snapshot cost an extra budget request (cache miss)"
    )


def test_yahoo_zero_budget_is_a_hard_stop(conn, monkeypatch):
    """An exhausted Yahoo hourly budget must refuse before any wire call."""
    limiter = RateLimiter("yahoo", 0, "hourly", conn)
    client = YahooClient(limiter=limiter)
    called = []
    monkeypatch.setattr(
        "rufus.yahoo._fetch_history",
        lambda *a, **k: called.append(1),
    )
    with pytest.raises(BudgetExhausted):
        client.fetch_history("AAPL")
    assert called == []


def test_currents_bad_key_is_auth_error_from_real_endpoint(conn):
    """A deliberately wrong key must produce a 401 auth error (real API)."""
    settings = Settings()
    limiter = RateLimiter("currents", settings.currents_max_req_per_day, "daily", conn)
    client = CurrentsClient(api_key="rufus-guard-invalid-key", limiter=limiter)

    with pytest.raises(CurrentsAuthError):
        client.search('"RELIANCE OR Reliance Industries"')
    assert limiter.used() == 1  # exactly one request hit the wire


def test_currents_zero_budget_blocks_before_wire(conn, monkeypatch):
    """An exhausted CurrentsAPI daily budget must refuse before the wire."""
    settings = Settings()
    limiter = RateLimiter("currents", 0, "daily", conn)
    client = CurrentsClient(
        api_key=settings.currentsapi_key or "rufus-guard-key",
        limiter=limiter,
    )
    called = []
    monkeypatch.setattr(
        "rufus.currents._get",
        lambda *a, **k: called.append(1) or (200, {"status": "ok", "news": []}, {}),
    )
    with pytest.raises(BudgetExhausted):
        client.search("AAPL")
    assert called == []