import datetime

import pytest

import rufus.db as db
import rufus.currents as currents
from rufus.currents import (
    CurrentsAuthError,
    CurrentsClient,
    CurrentsError,
    CurrentsNotConfigured,
    CurrentsQuotaExceeded,
    _parse_published,
)
from rufus.rate_limit import BudgetExhausted, RateLimiter


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    db.initialize_database(c)
    yield c
    c.close()


def make_client(conn, api_key="testkey", max_requests=100, cache=None, **kw):
    limiter = RateLimiter("currents", max_requests, "daily", conn)
    return CurrentsClient(api_key=api_key, limiter=limiter, cache=cache, **kw)


def ok_payload(*articles):
    return {"status": "ok", "news": list(articles), "page": 1}


def article(title, published="2026-09-07 14:22:08 +0000", aid=None):
    return {
        "id": aid or title,
        "title": title,
        "description": "summary",
        "url": "https://example.com/x",
        "author": "example.com",
        "image": "None",
        "language": "en",
        "category": ["finance"],
        "published": published,
    }


def test_search_normalizes_and_sorts(conn, monkeypatch):
    monkeypatch.setattr(
        currents,
        "_get",
        lambda url, params, headers: (
            200,
            ok_payload(
                article("older", "2026-09-06 09:00:00 +0000", "b"),
                article("newer", "2026-09-07 14:22:08 +0000", "a"),
            ),
            {},
        ),
    )
    results = make_client(conn).search('"Apple Inc." OR AAPL')
    assert [r["title"] for r in results] == ["newer", "older"]
    first = results[0]
    assert first["id"] == "a"
    assert first["published"] == "2026-09-07T14:22:08+00:00"
    assert first["source"] == "example.com"
    assert first["categories"] == ["finance"]
    assert first["language"] == "en"


def test_search_requires_api_key(conn, monkeypatch):
    monkeypatch.setattr(currents, "_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call")))
    client = make_client(conn, api_key="")
    with pytest.raises(CurrentsNotConfigured):
        client.search("AAPL")


def test_search_exhausted_budget_does_not_call(conn, monkeypatch):
    called = []
    monkeypatch.setattr(currents, "_get", lambda *a, **k: called.append(1))
    client = make_client(conn, api_key="k", max_requests=0)
    assert client.available is False
    with pytest.raises(BudgetExhausted):
        client.search("AAPL")
    assert called == []


def test_search_acquires_budget(conn, monkeypatch):
    monkeypatch.setattr(
        currents,
        "_get",
        lambda url, params, headers: (200, ok_payload(article("t")), {}),
    )
    client = make_client(conn)
    client.search("AAPL")
    used = db.get_api_usage(conn, "currents", client.limiter.current_bucket())
    assert used == 1


def test_search_auth_error(conn, monkeypatch):
    monkeypatch.setattr(currents, "_get", lambda *a, **k: (401, {"status": "error", "msg": "bad key"}, {}))
    with pytest.raises(CurrentsAuthError):
        make_client(conn).search("AAPL")


def test_search_quota_exceeded(conn, monkeypatch):
    monkeypatch.setattr(
        currents,
        "_get",
        lambda *a, **k: (429, {"msg": "quota"}, {"Retry-After": "3600"}),
    )
    with pytest.raises(CurrentsQuotaExceeded) as exc:
        make_client(conn).search("AAPL")
    assert exc.value.retry_after == 3600


def test_search_http_error(conn, monkeypatch):
    monkeypatch.setattr(currents, "_get", lambda *a, **k: (500, {"msg": "boom"}, {}))
    with pytest.raises(CurrentsError):
        make_client(conn).search("AAPL")


def test_search_retries_transient_transport_error(conn, monkeypatch):
    calls = {"n": 0}

    def flaky(url, params, headers):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("net down")
        return 200, ok_payload(article("t")), {}

    monkeypatch.setattr(currents, "_get", flaky)
    client = make_client(conn, retry_attempts=2, retry_base_delay_s=0.0, retry_jitter_s=0.0)
    results = client.search("AAPL")
    assert results and calls["n"] == 2
    # Retry must not re-charge the budget: exactly one request was spent.
    assert db.get_api_usage(conn, "currents", client.limiter.current_bucket()) == 1


def test_search_server_error_retried_then_raises(conn, monkeypatch):
    calls = {"n": 0}

    def busy(url, params, headers):
        calls["n"] += 1
        return 503, {"msg": "temporarily unavailable"}, {}

    monkeypatch.setattr(currents, "_get", busy)
    client = make_client(conn, retry_attempts=2, retry_base_delay_s=0.0, retry_jitter_s=0.0)
    with pytest.raises(CurrentsError) as exc:
        client.search("AAPL")
    assert exc.value.status == 503
    assert calls["n"] == 2


def test_search_quota_exceeded_is_never_retried(conn, monkeypatch):
    calls = {"n": 0}

    def quota(url, params, headers):
        calls["n"] += 1
        return 429, {"msg": "quota"}, {"Retry-After": "3600"}

    monkeypatch.setattr(currents, "_get", quota)
    client = make_client(conn, retry_attempts=3, retry_base_delay_s=0.0, retry_jitter_s=0.0)
    with pytest.raises(CurrentsQuotaExceeded):
        client.search("AAPL")
    assert calls["n"] == 1


def test_search_auth_error_is_never_retried(conn, monkeypatch):
    calls = {"n": 0}

    def auth(url, params, headers):
        calls["n"] += 1
        return 401, {"status": "error", "msg": "bad key"}, {}

    monkeypatch.setattr(currents, "_get", auth)
    client = make_client(conn, retry_attempts=3, retry_base_delay_s=0.0, retry_jitter_s=0.0)
    with pytest.raises(CurrentsAuthError):
        client.search("AAPL")
    assert calls["n"] == 1


def test_search_bad_status_body(conn, monkeypatch):
    monkeypatch.setattr(
        currents,
        "_get",
        lambda url, params, headers: (200, {"status": "error", "msg": "no articles"}, {}),
    )
    with pytest.raises(CurrentsError):
        make_client(conn).search("AAPL")


def test_search_uses_cache(conn, monkeypatch):
    calls = {"n": 0}

    def fake_get(url, params, headers):
        calls["n"] += 1
        return 200, ok_payload(article("t")), {}

    monkeypatch.setattr(currents, "_get", fake_get)
    client = make_client(conn)
    client.search("AAPL")
    client.search("AAPL")
    assert calls["n"] == 1
    # Only one request hit the budget despite two searches.
    assert db.get_api_usage(conn, "currents", client.limiter.current_bucket()) == 1


def test_search_url_and_auth_passed(conn, monkeypatch):
    captured = {}

    def fake_get(url, params, headers):
        captured.update(url=url, params=params, headers=headers)
        return 200, ok_payload(article("t")), {}

    monkeypatch.setattr(currents, "_get", fake_get)
    make_client(conn, api_key="sekrit").search("APPL OR AAPL", language="en")
    assert captured["url"] == currents.BASE_URL
    assert captured["params"]["keywords"] == "APPL OR AAPL"
    assert captured["params"]["language"] == "en"
    assert captured["params"]["limit"] == currents.RESULT_LIMIT
    assert "start_date" in captured["params"] and "end_date" in captured["params"]
    assert captured["headers"]["Authorization"] == "Bearer sekrit"


def test_search_default_window_is_seven_days(conn, monkeypatch):
    captured = {}

    def fake_get(url, params, headers):
        captured["params"] = params
        return 200, ok_payload(article("t")), {}

    monkeypatch.setattr(currents, "_get", fake_get)
    make_client(conn).search("AAPL")
    start = datetime.datetime.fromisoformat(captured["params"]["start_date"].replace("Z", "+00:00"))
    end = datetime.datetime.fromisoformat(captured["params"]["end_date"].replace("Z", "+00:00"))
    assert (end - start).days == currents.SEARCH_DAYS


def test_parse_published_variants():
    assert _parse_published("2026-09-07 14:22:08 +0000") == "2026-09-07T14:22:08+00:00"
    assert _parse_published(None) is None
    assert _parse_published("not a date") is None


def test_article_without_title_dropped(conn, monkeypatch):
    bad = article("keep me")
    bad_no_title = dict(article("ignored"))
    bad_no_title["title"] = None
    monkeypatch.setattr(
        currents,
        "_get",
        lambda url, params, headers: (200, ok_payload(bad_no_title, bad), {}),
    )
    results = make_client(conn).search("AAPL")
    assert [r["title"] for r in results] == ["keep me"]