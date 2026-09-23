import pytest

import rufus.ollama as ollama
from rufus.ollama import (
    OllamaClient,
    OllamaConnectionError,
    OllamaHTTPError,
    OllamaResponseError,
    extract_json,
)


def make_client(**kw):
    params = {
        "base_url": "http://127.0.0.1:9",
        "model": "qwen3:8b",
        "max_attempts": 2,
    }
    params.update(kw)
    return OllamaClient(**params)


def chat_reply(content):
    return {"message": {"role": "assistant", "content": content}}


def test_extract_json_from_fenced_block():
    text = '```json\n{"score": 0.5}\n```'
    assert extract_json(text) == {"score": 0.5}
    text = 'Thoughts:\n```\n{"a": 1}\n```\ndone'
    assert extract_json(text) == {"a": 1}


def test_extract_json_from_prose():
    text = 'The answer is {"score": -0.25}. Trust it.'
    assert extract_json(text) == {"score": -0.25}


def test_extract_json_plain():
    assert extract_json('{"x": [1, 2]}') == {"x": [1, 2]}


def test_extract_json_invalid():
    assert extract_json("no json here") is None
    assert extract_json("") is None


def test_chat_json_parses_reply(monkeypatch):
    monkeypatch.setattr(
        ollama, "_post",
        lambda url, body, timeout=90.0: (200, chat_reply('{"sentiment": "positive"}')),
    )
    client = make_client()
    out = client.chat_json([{"role": "user", "content": "hi"}])
    assert out == {"sentiment": "positive"}


def test_chat_json_sends_model_and_messages(monkeypatch):
    captured = {}

    def fake_post(url, body, timeout=90.0):
        captured.update(url=url, body=body)
        return 200, chat_reply('{"ok": true}')

    monkeypatch.setattr(ollama, "_post", fake_post)
    make_client(model="qwen3:8b", base_url="http://h:11434").chat_json(
        [{"role": "user", "content": "hello"}], temperature=0.2
    )
    assert captured["url"] == "http://h:11434/api/chat"
    assert captured["body"]["model"] == "qwen3:8b"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["body"]["stream"] is False
    assert captured["body"]["options"]["temperature"] == 0.2


def test_chat_json_retries_connection_error(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, body, timeout=90.0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ollama.OllamaConnectionError("boom")
        return 200, chat_reply('{"ok": true}')

    monkeypatch.setattr(ollama, "_post", fake_post)
    assert make_client().chat_json([{}]) == {"ok": True}
    assert calls["n"] == 2


def test_chat_json_retries_invalid_json_once(monkeypatch):
    replies = iter([chat_reply("sorry, not json"), chat_reply('{"ok": true}')])
    monkeypatch.setattr(
        ollama, "_post",
        lambda url, body, timeout=90.0: (200, next(replies)),
    )
    assert make_client().chat_json([{}]) == {"ok": True}


def test_chat_json_retries_validation_failure(monkeypatch):
    replies = iter([chat_reply('{"ok": false}'), chat_reply('{"score": 1}')])
    monkeypatch.setattr(
        ollama, "_post",
        lambda url, body, timeout=90.0: (200, next(replies)),
    )
    out = make_client().chat_json([{}], validate=lambda d: d.get("score") is not None)
    assert out == {"score": 1}


def test_chat_json_gives_up_after_attempts(monkeypatch):
    monkeypatch.setattr(
        ollama, "_post",
        lambda url, body, timeout=90.0: (200, chat_reply("still not json")),
    )
    with pytest.raises(OllamaResponseError):
        make_client(max_attempts=2).chat_json([{}])


def test_chat_json_http_error(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, body, timeout=90.0):
        calls["n"] += 1
        raise ollama.OllamaHTTPError("model not found")

    monkeypatch.setattr(ollama, "_post", fake_post)
    with pytest.raises(OllamaHTTPError):
        make_client().chat_json([{}])
    assert calls["n"] == 2


def test_chat_json_backs_off_on_transient_connection_error(monkeypatch):
    sleeps = []
    monkeypatch.setattr(ollama.time, "sleep", sleeps.append)
    calls = {"n": 0}

    def fake_post(url, body, timeout=90.0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ollama.OllamaConnectionError("down")
        return 200, chat_reply('{"ok": true}')

    monkeypatch.setattr(ollama, "_post", fake_post)
    client = make_client(max_attempts=3, retry_base_delay_s=1.0, retry_jitter_s=0.0)
    assert client.chat_json([{}]) == {"ok": True}
    assert calls["n"] == 2
    assert sleeps == [1.0]  # base * 2**0 before the second attempt


def test_chat_json_retries_http_5xx_with_exponential_backoff(monkeypatch):
    sleeps = []
    monkeypatch.setattr(ollama.time, "sleep", sleeps.append)
    calls = {"n": 0}

    def fake_post(url, body, timeout=90.0):
        calls["n"] += 1
        raise ollama.OllamaHTTPError("boom", status=503)

    monkeypatch.setattr(ollama, "_post", fake_post)
    client = make_client(max_attempts=3, retry_base_delay_s=0.5, retry_jitter_s=0.0)
    with pytest.raises(OllamaHTTPError):
        client.chat_json([{}])
    assert calls["n"] == 3
    assert sleeps == [0.5, 1.0]


def test_chat_json_http_4xx_not_backed_off(monkeypatch):
    sleeps = []
    monkeypatch.setattr(ollama.time, "sleep", sleeps.append)

    def fake_post(url, body, timeout=90.0):
        raise ollama.OllamaHTTPError("model not found", status=404)

    monkeypatch.setattr(ollama, "_post", fake_post)
    client = make_client(max_attempts=2, retry_base_delay_s=5.0, retry_jitter_s=0.0)
    with pytest.raises(OllamaHTTPError):
        client.chat_json([{}])
    assert sleeps == []


def test_chat_json_logs_raw_reply_on_parse_failure(monkeypatch, caplog):
    monkeypatch.setattr(
        ollama, "_post",
        lambda url, body, timeout=90.0: (200, chat_reply("definitely not json")),
    )
    with caplog.at_level("DEBUG", logger="rufus.ollama"):
        with pytest.raises(OllamaResponseError):
            make_client(max_attempts=1).chat_json([{}])
    assert "ollama raw reply (attempt 1)" in caplog.text
    assert "definitely not json" in caplog.text


def test_raw_reply_log_is_clipped():
    long_reply = "x" * 5000
    assert ollama._clip(long_reply).endswith("...[truncated]")
    assert len(ollama._clip(long_reply)) <= 2000 + len("...[truncated]")
    assert ollama._clip("short") == "short"
    assert ollama._clip(None) == ""