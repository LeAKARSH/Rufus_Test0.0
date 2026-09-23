from datetime import datetime, timezone

import pytest

from rufus import db
from rufus.rate_limit import RateLimiter, bucket_key


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    db.initialize_database(c)
    yield c
    c.close()


def test_bucket_key_formats():
    when = datetime(2026, 1, 1, 15, 30, tzinfo=timezone.utc)
    assert bucket_key("yahoo", "hourly", when) == "yahoo:2026-01-01T15"
    assert bucket_key("currents", "daily", when) == "currents:2026-01-01"


def test_limiter_allows_up_to_cap(conn):
    limiter = RateLimiter("yahoo", max_requests=3, window="hourly", conn=conn)
    assert limiter.remaining() == 3
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is True
    assert limiter.remaining() == 0
    assert limiter.try_acquire() is False
    assert limiter.used() == 3


def test_limiter_no_partial_grants(conn):
    limiter = RateLimiter("yahoo", max_requests=3, window="hourly", conn=conn)
    assert limiter.try_acquire(count=2) is True
    assert limiter.try_acquire(count=2) is False
    assert limiter.used() == 2


def test_zero_or_negative_count_is_noop(conn):
    limiter = RateLimiter("yahoo", max_requests=3, window="hourly", conn=conn)
    assert limiter.try_acquire(count=0) is True
    assert limiter.try_acquire(count=-5) is True
    assert limiter.used() == 0


def test_budget_survives_new_instance(conn):
    a = RateLimiter("currents", max_requests=10, window="daily", conn=conn)
    assert a.try_acquire() is True
    assert a.try_acquire() is True

    b = RateLimiter("currents", max_requests=10, window="daily", conn=conn)
    assert b.used() == 2
    assert b.remaining() == 8


def test_daily_and_hourly_providers_are_independent(conn):
    hourly = RateLimiter("yahoo", 2, "hourly", conn=conn)
    daily = RateLimiter("currents", 2, "daily", conn=conn)
    assert hourly.try_acquire() is True
    assert daily.try_acquire() is True
    assert hourly.used() == 1
    assert daily.used() == 1


def test_usage_is_persisted_to_api_usage_log(conn):
    lima = RateLimiter("yahoo", 5, "hourly", conn=conn)
    lima.try_acquire()
    lima.try_acquire()
    provider, bucket = lima.provider, lima.current_bucket()
    assert db.get_api_usage(conn, provider, bucket) == 2