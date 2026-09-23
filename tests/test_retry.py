"""Shared retry-with-backoff unit tests (Phase 6 hardening)."""

import pytest

from rufus.retry import default_is_transient, http_status, retry_call


def _flaky(succeed_on, exc_factory, calls):
    def fn():
        calls["n"] += 1
        if calls["n"] >= succeed_on:
            return "ok"
        raise exc_factory()
    return fn


def test_retry_call_succeeds_after_transient_failures():
    calls = {"n": 0}
    sleeps = []
    out = retry_call(
        _flaky(3, OSError, calls),
        attempts=3, base_delay_s=1.0, jitter_s=0.0, sleep=sleeps.append,
    )
    assert out == "ok"
    assert calls["n"] == 3
    assert sleeps == [1.0, 2.0]  # exponential: 1s then 2s


def test_retry_call_single_attempt_no_retry():
    calls = {"n": 0}
    with pytest.raises(OSError):
        retry_call(_flaky(2, OSError, calls), attempts=1)
    assert calls["n"] == 1


def test_retry_call_exhaustion_raises_last():
    calls = {"n": 0}
    sleeps = []
    with pytest.raises(OSError, match="boom"):
        retry_call(
            _flaky(99, lambda: OSError("boom"), calls),
            attempts=2, base_delay_s=0.0, jitter_s=0.0, sleep=sleeps.append,
        )
    assert calls["n"] == 2


def test_retry_call_non_transient_propagates_immediately():
    calls = {"n": 0}
    with pytest.raises(ValueError):
        retry_call(_flaky(2, ValueError, calls), attempts=3, base_delay_s=5.0)
    assert calls["n"] == 1


def test_retry_call_custom_predicate():
    class FlakyServiceError(Exception):
        pass

    calls = {"n": 0}
    out = retry_call(
        _flaky(2, FlakyServiceError, calls),
        attempts=3, is_transient=lambda exc: isinstance(exc, FlakyServiceError),
    )
    assert out == "ok"
    assert calls["n"] == 2


def test_retry_call_rejects_zero_attempts():
    with pytest.raises(ValueError):
        retry_call(lambda: None, attempts=0)


def test_default_is_transient_covers_requests_errors():
    # requests timeouts/connection errors subclass OSError, so the default
    # predicate catches them without importing requests here.
    assert default_is_transient(OSError("net down"))
    assert default_is_transient(TimeoutError())
    assert default_is_transient(ConnectionError())
    assert not default_is_transient(ValueError("nope"))
    assert not default_is_transient(KeyError("nope"))


def test_http_status_helper():
    assert http_status(Exception()) is None
    assert http_status(ValueError()) is None


def test_retry_on_retry_callback_invoked():
    calls = {"n": 0}
    seen = []
    retry_call(
        _flaky(2, OSError, calls),
        attempts=2, base_delay_s=0.0, jitter_s=0.0, sleep=lambda d: None,
        on_retry=lambda attempt, exc: seen.append((attempt, type(exc).__name__)),
    )
    assert seen == [(1, "OSError")]