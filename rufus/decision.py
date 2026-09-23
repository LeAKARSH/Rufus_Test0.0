"""LLM Decision Engine (spec Section 6 / Phase 3).

Turns the persisted feature digest (technical + fundamentals + sentiment +
headline text) into a single structured BUY / HOLD / SELL / AVOID call per
ticker, using the configurable Decision-Engine Ollama model. The response is
validated against a fixed JSON schema and stored in ``recommendations``.

Design notes:
- Reuses the swappable :class:`rufus.ollama.OllamaClient` (one of the two
  clients built per Settings, distinct from the sentiment model).
- The prompt forces grounding in the provided data only; the schema validator
  normalizes case and fills nullable fields so the DB row is always clean.
- Per-ticker failures are logged and skipped; they never abort the rest of
  the watchlist, and the persistent ``job_state`` marker keeps the scheduler
  from double-running a completed day.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

import rufus.db as db
from rufus.config import Settings
from rufus.features import build_ticker_feature
from rufus.ollama import OllamaClient, OllamaError

log = logging.getLogger(__name__)

RECOMMENDATIONS = ("BUY", "HOLD", "SELL", "AVOID")
CONFIDENCE_LEVELS = ("HIGH", "MEDIUM", "LOW")

# Decision-cycle identity used for the ``job_state`` restart marker.
JOB_DECISION = "decision"

_SYSTEM_PROMPT = (
    "You are a long-term investment research assistant (3-6 month to 1-year "
    "holding horizon), not a day trader. You will be given a structured block "
    "of data for one company: technical indicators, fundamentals, sentiment "
    "score and recent headlines. Decide strictly from that data: never "
    "fabricate facts, prices, earnings dates, or news that are not in the "
    "block. Explicitly weigh BOTH the technical/fundamental data and the "
    "sentiment/news data, and explain how each influenced your call.\n\n"
    "Respond with a single JSON object only, using exactly these keys:\n"
    '{"ticker": string, "recommendation": "BUY|HOLD|SELL|AVOID", '
    '"confidence": "HIGH|MEDIUM|LOW", '
    '"suggested_horizon": short string, '
    '"reasoning": multi-sentence string, '
    '"key_catalysts": [strings], "key_risks": [strings], '
    '"revisit_after": "YYYY-MM-DD" or null}\n\n'
    "AVOID means a long-term investor should not initiate or persist a "
    "position now. This is decision support, not financial advice."
)


def _valid_recommendation(obj: Any) -> bool:
    """Schema gate for the chat reply (lenient on casing, strict on keys)."""
    if not isinstance(obj, dict):
        return False
    rec = (obj.get("recommendation") or "").upper()
    conf = (obj.get("confidence") or "").upper()
    if rec not in RECOMMENDATIONS:
        return False
    if conf not in CONFIDENCE_LEVELS:
        return False
    if not isinstance(obj.get("suggested_horizon"), str) or not obj.get("suggested_horizon"):
        return False
    if not isinstance(obj.get("reasoning"), str) or not obj.get("reasoning"):
        return False
    if not isinstance(obj.get("key_catalysts"), list) or not isinstance(obj.get("key_risks"), list):
        return False
    for lst in (obj["key_catalysts"], obj["key_risks"]):
        if any(not isinstance(x, str) for x in lst):
            return False
    return True


def normalize_recommendation(obj: dict[str, Any]) -> dict[str, Any]:
    """Normalize a validated reply into its canonical stored shape."""
    revisit = obj.get("revisit_after")
    return {
        "recommendation": (obj.get("recommendation") or "").upper(),
        "confidence": (obj.get("confidence") or "").upper(),
        "suggested_horizon": obj.get("suggested_horizon"),
        "reasoning": obj.get("reasoning"),
        "key_catalysts": list(obj.get("key_catalysts") or []),
        "key_risks": list(obj.get("key_risks") or []),
        "revisit_after": (str(revisit) if revisit else None),
    }


def build_messages(feature: dict[str, Any]) -> list[dict[str, str]]:
    """Assemble the system + user turn for one ticker's decision call."""
    block = json.dumps(feature, indent=2, default=str)
    user = (
        "Evaluate this company for a long-term (3-6 month to 1-year) holding "
        f"decision. Data block:\n\n{block}\n\n"
        "Return the required JSON object for this exact ticker."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def get_recommendation(
    client: OllamaClient,
    feature: dict[str, Any],
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Ask the Decision-Engine LLM for one ticker; returns normalized JSON.

    Raises :class:`rufus.ollama.OllamaError` when the model cannot be reached
    or keeps failing schema validation (the client retries internally).
    """
    obj = client.chat_json(
        build_messages(feature),
        validate=_valid_recommendation,
        temperature=temperature,
    )
    return normalize_recommendation(obj)


def latest_price_row(conn, ticker: str):
    return conn.execute(
        "SELECT * FROM price_snapshots WHERE ticker = ? "
        "ORDER BY captured_at DESC LIMIT 1",
        (ticker.upper(),),
    ).fetchone()


def latest_news_row(conn, ticker: str):
    return conn.execute(
        "SELECT * FROM news_snapshots WHERE ticker = ? "
        "ORDER BY run_date DESC LIMIT 1",
        (ticker.upper(),),
    ).fetchone()


def run_decision_cycle(
    conn,
    settings: Settings,
    client: OllamaClient,
    today: date | None = None,
) -> list[dict]:
    """Score every watchlist ticker: feature -> LLM -> stored recommendation.

    Returns one stats dict per ticker processed (or failed). Tickers without a
    price snapshot are skipped (nothing to ground the call on); tickers whose
    LLM call fails are logged and skipped without a row.
    """
    today = today or _market_today(settings.market_timezone)
    step = str(today)
    tickers = db.get_active_tickers(conn)

    stats = []
    for ticker in tickers:
        price = latest_price_row(conn, ticker)
        if price is None:
            log.warning("no price snapshot for %s; skipping decision", ticker)
            continue
        news = latest_news_row(conn, ticker)
        try:
            feature = build_ticker_feature(price, news)
            rec = get_recommendation(client, feature)
        except OllamaError as exc:
            log.error("decision failed for %s: %s", ticker, exc)
            stats.append({"ticker": ticker, "recommendation": None, "error": str(exc)})
            continue

        db.insert_recommendation(
            conn,
            ticker,
            run_date=step,
            recommendation=rec["recommendation"],
            confidence=rec["confidence"],
            suggested_horizon=rec["suggested_horizon"],
            reasoning=rec["reasoning"],
            key_catalysts=rec["key_catalysts"],
            key_risks=rec["key_risks"],
            revisit_after=rec["revisit_after"],
            model=client.model,
            input_data_json=json.dumps(feature, default=str),
            raw_response_json=json.dumps(rec),
        )
        stats.append({"ticker": ticker, "recommendation": rec["recommendation"]})
        log.info(
            "decision stored for %s: %s (%s), horizon=%s",
            ticker, rec["recommendation"], rec["confidence"], rec["suggested_horizon"],
        )

    log.info("decision cycle done for %s: %d ticker(s)", step, len(stats))
    return stats


def create_decision_poll_fn(settings: Settings, conn) -> Callable[[object], None]:
    """Build the scheduler's decision-cycle callback backed by this module."""
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout_seconds,
        max_attempts=settings.ollama_retry_attempts,
        retry_base_delay_s=settings.retry_base_delay_s,
        retry_jitter_s=settings.retry_jitter_s,
    )

    def cycle(conn_obj) -> None:
        run_decision_cycle(conn_obj, settings, client)

    return cycle


def _market_today(tz_name: str) -> date:
    try:
        return datetime.now(ZoneInfo(tz_name)).date()
    except Exception:
        return date.today()