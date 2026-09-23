"""Swappable Ollama client (spec Section 3.3).

One thin client serves both the sentiment model and the Phase 3 Decision
Engine (each with its own base URL/model from Settings, built at the call
site). It wraps the ``/api/chat`` endpoint and adds:

- JSON extraction from the assistant reply (tolerates code fences and a
  surrounding prose block) so models trained for chat don't need a separate
  structured-output path;
- an optional ``validate`` callable for lightweight schema checks, with one
  recovery retry when the reply isn't usable JSON or fails validation;
- one retry on transient connection errors (the LLM has no per-day quota, so
  a single patient retry is cheap);

The module-level ``_post`` is factored out so tests can monkeypatch it.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any, Callable, Mapping

import requests

from rufus.config import (
    DEFAULT_OLLAMA_RETRY_ATTEMPTS,
    DEFAULT_RETRY_BASE_DELAY_S,
    DEFAULT_RETRY_JITTER_S,
)

log = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class OllamaError(Exception):
    """Base class for Ollama client failures."""


class OllamaConnectionError(OllamaError):
    """Could not reach the Ollama server (network/timeout)."""


class OllamaHTTPError(OllamaError):
    """Ollama answered with a non-200 status."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class OllamaResponseError(OllamaError):
    """The reply had no usable JSON or failed the schema validation."""


Validator = Callable[[dict[str, Any]], bool]


def _post(url: str, body: Mapping[str, Any], timeout: float = 90.0) -> tuple[int, Any]:
    """POST JSON to Ollama; separated so tests can monkeypatch it."""
    try:
        resp = requests.post(url, json=dict(body), timeout=timeout)
    except requests.RequestException as exc:
        raise OllamaConnectionError(f"could not reach Ollama at {url}: {exc}") from exc
    try:
        payload = resp.json()
    except (ValueError, json.JSONDecodeError):
        payload = None
    if resp.status_code != 200:
        raise OllamaHTTPError(
            f"Ollama HTTP {resp.status_code}: {payload}", status=resp.status_code
        )
    return resp.status_code, payload


def extract_json(content: str) -> Any:
    """Pull a JSON object out of a chat reply, tolerating fences/prose."""
    if not content:
        return None
    text = content.strip()
    code_blocks = _JSON_FENCE_RE.findall(text)
    candidates: list[str] = []
    if code_blocks:
        candidates.append(code_blocks[0].strip())
    candidates.append(text)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except ValueError:
            pass
        # Otherwise try the first balanced {...} block inside prose.
        start = candidate.find("{")
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(candidate)):
            ch = candidate[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start : i + 1])
                    except ValueError:
                        break
    return None


class OllamaClient:
    """JSON-first chat client for one Ollama model endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 90.0,
        max_attempts: int = DEFAULT_OLLAMA_RETRY_ATTEMPTS,
        retry_base_delay_s: float = DEFAULT_RETRY_BASE_DELAY_S,
        retry_jitter_s: float = DEFAULT_RETRY_JITTER_S,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_base_delay_s = retry_base_delay_s
        self.retry_jitter_s = retry_jitter_s

    def chat_json(
        self,
        messages: list[dict[str, str]],
        validate: Validator | None = None,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Send a chat turn and return the parsed JSON object.

        Retries once per category (connection error / unusable reply) before
        giving up. ``validate`` keys the schema judgement so callers can have
        e.g. per-article score alignment enforced inside the retry loop.
        """
        last_error: OllamaError | None = None
        for attempt in range(self.max_attempts):
            try:
                content = self._chat(messages, temperature)
            except OllamaError as exc:  # connection/HTTP failures
                last_error = exc
                log.warning("ollama attempt %d failed: %s", attempt + 1, exc)
                if attempt < self.max_attempts - 1 and self._is_transient(exc):
                    time.sleep(self._backoff(attempt))
                continue

            obj = extract_json(content)
            if obj is None:
                last_error = OllamaResponseError("reply contained no usable JSON")
                log.warning("ollama reply had no JSON (attempt %d)", attempt + 1)
                log.debug("ollama raw reply (attempt %d): %r", attempt + 1, _clip(content))
                continue
            if validate is not None and not validate(obj):
                last_error = OllamaResponseError("reply failed schema validation")
                log.warning("ollama reply failed validation (attempt %d)", attempt + 1)
                log.debug("ollama raw reply (attempt %d): %r", attempt + 1, _clip(content))
                continue
            return obj

        assert last_error is not None
        raise last_error

    def _chat(self, messages: list[dict[str, str]], temperature: float) -> str:
        url = f"{self.base_url}/api/chat"
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        _status, payload = _post(url, body, timeout=self.timeout)
        if not isinstance(payload, dict):
            raise OllamaResponseError("Ollama returned a non-JSON response body")
        return str(payload.get("message", {}).get("content") or "")

    @staticmethod
    def _is_transient(exc: OllamaError) -> bool:
        """Retry connection failures and HTTP 5xx; never auth/config 4xx."""
        if isinstance(exc, OllamaConnectionError):
            return True
        status = getattr(exc, "status", None)
        return isinstance(exc, OllamaHTTPError) and status is not None and status >= 500

    def _backoff(self, failed_attempt_index: int) -> float:
        delay = self.retry_base_delay_s * (2 ** failed_attempt_index)
        return delay + random.uniform(0, self.retry_jitter_s)


def create_client(base_url: str, model: str, timeout: float = 90.0) -> OllamaClient:
    """Convenience factory for call sites that wire up a single model."""
    return OllamaClient(base_url=base_url, model=model, timeout=timeout)


# Cap on how much of a raw malformed reply is logged for debugging (spec 6.3).
_CLIP_LEN = 2000


def _clip(content: str) -> str:
    if content is None:
        return ""
    return content if len(content) <= _CLIP_LEN else content[:_CLIP_LEN] + "...[truncated]"