"""News sentiment scoring pipeline (spec Section 5, option 3).

Tickers' fetched articles are batch-scored by the dedicated (small) Ollama
sentiment model, then rolled up into the persisted ``news_snapshots`` row:
average score (scale -1..1), a 7-day trend label, article count and the top
headlines. Scoring is deliberately swappable: everything enters through the
shared :class:`rufus.ollama.OllamaClient`, so switching models (or hosts)
is a config change only.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

import rufus.db as db
from rufus.ollama import OllamaClient, OllamaError

log = logging.getLogger(__name__)

SCORE_MIN, SCORE_MAX = -1.0, 1.0
TREND_THRESHOLD = 0.1  # |diff| below this keeps the previous mood "stable"
TOP_HEADLINES = 5
DEFAULT_TRIM_CHARS = 300

_SYSTEM_PROMPT = (
    "You are a financial news sentiment analyst. For each article, return your "
    "sentiment as a JSON object only, with a single 'articles' array. Each entry "
    "must be {\"title\": <exact article title>, \"score\": <float -1.0 to 1.0>, "
    "\"note\": <one short sentence>}. Positive news -> positive score, negative "
    "news -> negative score, neutral or mixed -> near 0. No prose outside the JSON."
)


class SentimentScoreError(Exception):
    """The sentiment model could not produce usable scores."""


def _valid_scores(obj: Any, expected_titles: set[str]) -> bool:
    """Schema check for the sentiment model's reply (used as the retry key)."""
    if not isinstance(obj, dict):
        return False
    articles = obj.get("articles")
    if not isinstance(articles, list) or not articles:
        return False
    for entry in articles:
        if not isinstance(entry, dict):
            return False
        title = entry.get("title")
        score = entry.get("score")
        if not isinstance(title, str) or title not in expected_titles:
            return False
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return False
        if not SCORE_MIN <= float(score) <= SCORE_MAX:
            return False
    return True


def build_messages(
    articles: Sequence[dict[str, Any]],
    trim_chars: int = DEFAULT_TRIM_CHARS,
) -> list[dict[str, str]]:
    """Chat messages requesting per-article scores for ``articles``."""
    feed = [
        {
            "title": a["title"],
            "description": (a.get("description") or "")[:trim_chars],
        }
        for a in articles
    ]
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"articles": feed})},
    ]


def score_batch(
    client: OllamaClient,
    articles: Sequence[dict[str, Any]],
    max_articles: int,
    trim_chars: int = DEFAULT_TRIM_CHARS,
) -> list[dict[str, Any]]:
    """Score up to ``max_articles`` articles; results aligned to input order.

    Entries the model didn't score (or couldn't match by title) are omitted.
    Raises :class:`SentimentScoreError` when no article got a usable score.
    """
    feed = list(articles)[: max_articles if max_articles > 0 else len(articles)]
    if not feed:
        return []
    expected = {a["title"] for a in feed}
    reply = client.chat_json(
        build_messages(feed, trim_chars),
        validate=lambda obj: _valid_scores(obj, expected),
    )
    by_title = {
        entry["title"]: float(entry["score"])
        for entry in reply.get("articles", [])
        if isinstance(entry, dict) and entry.get("title") in expected
    }
    scored = [
        {"title": a["title"], "score": by_title[a["title"]]}
        for a in feed
        if a["title"] in by_title
    ]
    if not scored:
        raise SentimentScoreError("no usable scores returned for the articles")
    return scored


def aggregate_sentiment(
    articles: Sequence[dict[str, Any]],
    scored: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Roll per-article scores into the persisted summary dict."""
    by_title = {s["title"]: s["score"] for s in scored}
    scored_articles = [
        {
            "id": a.get("id"),
            "title": a["title"],
            "description": a.get("description") or "",
            "url": a.get("url"),
            "source": a.get("source"),
            "published": a.get("published"),
            "score": by_title[a["title"]],
        }
        for a in articles
        if a["title"] in by_title
    ]
    values = [s["score"] for s in scored_articles]
    avg = sum(values) / len(values) if values else None
    top = sorted(
        scored_articles,
        key=lambda s: abs(s["score"]),
        reverse=True,
    )[:TOP_HEADLINES]
    return {
        "sentiment_score_avg": round(avg, 4) if avg is not None else None,
        "articles_considered": len(scored_articles),
        "top_headlines": [
            {k: s[k] for k in ("title", "score", "published", "url", "source")}
            for s in top
        ],
        "articles": scored_articles,
    }


def sentiment_trend(prior_scores: Sequence[float], today_score: float | None) -> str | None:
    """Label today's score versus the mean of recent prior scores."""
    if today_score is None or not prior_scores:
        return None
    delta = today_score - (sum(prior_scores) / len(prior_scores))
    if delta >= TREND_THRESHOLD:
        return "improving"
    if delta <= -TREND_THRESHOLD:
        return "worsening"
    return "stable"


def score_ticker_news(
    client: OllamaClient,
    conn,
    ticker: str,
    keywords: str,
    run_date: str,
    articles: Sequence[dict[str, Any]],
    max_articles: int,
) -> dict[str, Any] | None:
    """Score today's articles for one ticker and persist the ``news_snapshots`` row.

    A zero-article day is still recorded (empty pull) so the rotation's overdue
    clock resets; a scoring failure stores the row without a score but still
    records the pull. Returns the aggregate when scoring succeeded.
    """
    if not articles:
        db.insert_news_snapshot(
            conn, ticker, run_date=run_date, query_keyword=keywords,
            articles_considered=0,
        )
        return None

    try:
        scored = score_batch(client, articles, max_articles)
    except (SentimentScoreError, OllamaError):
        log.warning("sentiment scoring failed for %s; storing pull w/o score", ticker)
        db.insert_news_snapshot(
            conn, ticker, run_date=run_date, query_keyword=keywords,
            articles_considered=len(articles),
        )
        return None

    agg = aggregate_sentiment(articles, scored)
    trend = sentiment_trend(
        db.get_prior_news_scores(conn, ticker, before_run_date=run_date),
        agg["sentiment_score_avg"],
    )
    db.insert_news_snapshot(
        conn,
        ticker,
        run_date=run_date,
        query_keyword=keywords,
        sentiment_score_avg=agg["sentiment_score_avg"],
        sentiment_trend_7d=trend,
        articles_considered=agg["articles_considered"],
        top_headlines_json=json.dumps(agg["top_headlines"]),
        articles_json=json.dumps(agg["articles"]),
    )
    return agg