import time
from datetime import timedelta

from rufus.cache import TTLCache


def test_set_get_roundtrip():
    cache = TTLCache()
    cache.set("a", {"x": 1})
    assert cache.get("a") == {"x": 1}
    assert cache.get("missing") is None


def test_expiry():
    cache = TTLCache(default_ttl=timedelta(milliseconds=50))
    cache.set("short", "v")
    time.sleep(0.1)
    assert cache.get("short") is None


def test_per_entry_ttl_overrides_default():
    cache = TTLCache(default_ttl=timedelta(hours=1))
    cache.set("never", "v", ttl=None)
    cache.set("short", "v", ttl=timedelta(milliseconds=50))
    time.sleep(0.1)
    assert cache.get("never") == "v"
    assert cache.get("short") is None


def test_delete_and_clear():
    cache = TTLCache()
    cache.set("a", 1)
    cache.set("b", 2)
    cache.delete("a")
    assert cache.get("a") is None
    assert cache.get("b") == 2
    cache.clear()
    assert cache.get("b") is None


def test_size_purges_expired():
    cache = TTLCache(default_ttl=timedelta(milliseconds=50))
    cache.set("gone", "x")
    time.sleep(0.1)
    cache.set("kept", "y")
    assert cache.size() == 1
    assert cache.keys() == ["kept"]


def test_get_or_set_computes_once():
    cache = TTLCache()
    calls = []

    def factory():
        calls.append(1)
        return "value"

    assert cache.get_or_set("k", factory) == "value"
    assert cache.get_or_set("k", factory) == "value"
    assert len(calls) == 1
    assert cache.get("k") == "value"


def test_disabled_ttl_never_expires():
    cache = TTLCache()
    cache.set("k", "v", ttl=timedelta(0))
    assert cache.get("k") == "v"