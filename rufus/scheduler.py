"""Scheduler / daemon with market-hours awareness (spec Section 2.1).

Wires three cycles: the intraday Yahoo price poll (while the market is
open), the 24h news/sentiment rotation, and the once-per-trading-day LLM
decision cycle — each optionally followed by the Phase 4 paper-trading pass
(decisions -> simulated fills -> equity-curve snapshot) when wired in
``main``.

Design notes:
- ``MarketCalendar`` wraps ``pandas_market_calendars`` for US market hours and
  holidays (weekends + market holidays handled by the calendar library, not a
  hardcoded list).
- ``Scheduler`` runs a small state machine each tick: while the market is
  open, invoke ``poll_fn`` when the poll interval has elapsed; while the
  market is closed, simply sleep until the next market open.
- Last-poll time is recovered from the latest ``price_snapshots`` row on
  every pass, so a process restart mid-day will not double-poll.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Callable
from zoneinfo import ZoneInfo

import pandas_market_calendars as pmc

import rufus.db as db
from rufus.config import Settings, get_settings
from rufus.decision import JOB_DECISION, create_decision_poll_fn
from rufus.logging_config import setup_logging
from rufus.news import create_news_poll_fn
from rufus.paper import create_paper_cycle, refresh_equity
from rufus.report import generate_daily_report
from rufus.yahoo import default_poll_fn

log = logging.getLogger(__name__)

# Lookahead for "next session open": the longest holiday gap (Christmas /
# New Year) fits comfortably inside this window.
_LOOKAHEAD_DAYS = 12

PollFn = Callable[[sqlite3.Connection, list[str]], None]
NewsPollFn = Callable[[sqlite3.Connection], None]
CycleFn = Callable[[sqlite3.Connection], None]

JOB_NEWS = "news"


class MarketCalendar:
    """Trading-hours/holiday awareness for one exchange calendar."""

    def __init__(self, exchange: str = "XNSE", tz: str = "Asia/Kolkata") -> None:
        self.exchange = exchange
        self.tz = tz
        self._cal = pmc.get_calendar(exchange)
        self._lookahead_days = _LOOKAHEAD_DAYS

    def _schedule(self, start: datetime, days: int, tz: timezone = timezone.utc):
        end = start.date() + timedelta(days=days)
        return self._cal.schedule(start, end)

    def is_open(self, at: datetime) -> bool:
        """True if ``at`` falls inside a trading session for that calendar."""
        at = at.astimezone(timezone.utc)
        df = self._schedule(at, days=0)
        if df.empty:
            return False
        row = df.iloc[0]
        return row["market_open"] <= at <= row["market_close"]

    def next_open(self, at: datetime) -> datetime:
        """UTC datetime of the next market open strictly after ``at``."""
        at = at.astimezone(timezone.utc)
        df = self._schedule(at, days=self._lookahead_days)
        future = df[df["market_open"] > at]
        if future.empty:
            raise RuntimeError(
                f"no market open found within {self._lookahead_days} days of {at}"
            )
        return future["market_open"].iloc[0].to_pydatetime().astimezone(timezone.utc)

    def next_close(self, at: datetime) -> datetime:
        """UTC datetime of the next market close at or after ``at``."""
        at = at.astimezone(timezone.utc)
        df = self._schedule(at, days=self._lookahead_days)
        upcoming = df[df["market_close"] >= at]
        if upcoming.empty:
            raise RuntimeError(
                f"no market close found within {self._lookahead_days} days of {at}"
            )
        return upcoming["market_close"].iloc[0].to_pydatetime().astimezone(timezone.utc)

    def is_trading_day(self, day) -> bool:
        """True when ``day`` has at least one scheduled trading session."""
        df = self._cal.schedule(day, day + timedelta(days=1))
        return not df.empty


class NewsCycle:
    """Calendar-day news cycle, independent of market hours.

    Gated purely by elapsed time since the last run (marker persisted in
    ``job_state``), so a restart never double-runs it and the interval may
    elapse on weekends/holidays without waiting for a market session.
    """

    def __init__(
        self,
        settings: Settings,
        conn: sqlite3.Connection,
        poll_fn: NewsPollFn | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.conn = conn
        self.poll_fn = poll_fn or self._default_news_poll
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.interval = timedelta(hours=settings.news_poll_hours)

    def due(self, now: datetime) -> bool:
        last = db.get_job_last_run(self.conn, JOB_NEWS)
        if last is None:
            return True
        last_dt = _parse_utc(last)
        if last_dt is None:
            return True
        return now - last_dt >= self.interval

    def run(self, now: datetime) -> None:
        db.set_job_last_run(self.conn, JOB_NEWS, now.isoformat())
        log.info("news cycle due at %s", now.isoformat())
        self.poll_fn(self.conn)

    def next_run(self, now: datetime) -> datetime:
        last = db.get_job_last_run(self.conn, JOB_NEWS)
        if last is None:
            return now
        last_dt = _parse_utc(last)
        if last_dt is None:
            return now
        return last_dt + self.interval

    def _default_news_poll(self, conn: sqlite3.Connection) -> None:
        log.info("(dry-run) would run the news cycle")


class DecisionCycle:
    """Once-per-trading-day LLM decision cycle (spec Section 2.1).

    Fires when the market-local clock passes the configured daily decision
    time on a trading day, never more than once per calendar day (marker
    persisted in ``job_state``), and never on weekends/holidays.

    When ``paper_fn`` is provided it runs immediately after the decisions are
    stored, so one pass produces recommendations *and* their paper fills.
    ``report_fn`` (i.e. the daily report) runs last, after the fills.
    """

    def __init__(
        self,
        settings: Settings,
        conn: sqlite3.Connection,
        cycle_fn: CycleFn | None = None,
        clock: Callable[[], datetime] | None = None,
        market: MarketCalendar | None = None,
        paper_fn: CycleFn | None = None,
        report_fn: CycleFn | None = None,
    ) -> None:
        self.settings = settings
        self.conn = conn
        self.cycle_fn = cycle_fn or self._default_cycle
        self.paper_fn = paper_fn
        self.report_fn = report_fn
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.market = market or MarketCalendar(
            settings.market_exchange, settings.market_timezone
        )
        self._tz = ZoneInfo(settings.market_timezone)

    def due(self, now: datetime) -> bool:
        if self._ran_today(now):
            return False
        sched = self._schedule_for(now.astimezone(self._tz).date())
        return sched is not None and now >= sched

    def run(self, now: datetime) -> None:
        db.set_job_last_run(self.conn, JOB_DECISION, now.isoformat())
        log.info("decision cycle due at %s", now.isoformat())
        started = time.monotonic()
        self.cycle_fn(self.conn)
        log.info("decision cycle completed in %.1fs", time.monotonic() - started)
        if self.paper_fn is not None:
            started = time.monotonic()
            try:
                self.paper_fn(self.conn)
            except Exception:
                log.exception("paper cycle failed after decisions were stored")
            else:
                log.info("paper cycle completed in %.1fs", time.monotonic() - started)
        if self.report_fn is not None:
            started = time.monotonic()
            try:
                self.report_fn(self.conn)
            except Exception:
                log.exception("report generation failed after paper cycle")
            else:
                log.info("report generated in %.1fs", time.monotonic() - started)

    def next_run(self, now: datetime) -> datetime:
        if not self._ran_today(now):
            sched = self._schedule_for(now.astimezone(self._tz).date())
            if sched is not None and now < sched:
                return sched  # awaiting today's decision slot
            if sched is not None:
                return now  # due this instant
        # Already ran today, or no decision slot today (holiday/weekend).
        day = now.astimezone(self._tz).date() + timedelta(days=1)
        for _ in range(_LOOKAHEAD_DAYS):
            sched = self._schedule_for(day)
            if sched is not None:
                return sched
            day += timedelta(days=1)
        return now + timedelta(days=1)

    def _ran_today(self, now: datetime) -> bool:
        last = db.get_job_last_run(self.conn, JOB_DECISION)
        if last is None:
            return False
        last_dt = _parse_utc(last)
        if last_dt is None:
            return False
        return last_dt.astimezone(self._tz).date() == now.astimezone(self._tz).date()

    def _schedule_for(self, day) -> datetime | None:
        tz_checker = self.market
        is_trading = getattr(tz_checker, "is_trading_day", None)
        if is_trading and not is_trading(day):
            return None
        hours, _, minutes = self.settings.decision_cycle_time.partition(":")
        try:
            local = datetime.combine(
                day, dt_time(int(hours), int(minutes)), tzinfo=self._tz
            )
        except (TypeError, ValueError):
            log.warning("bad DECISION_CYCLE_TIME=%r", self.settings.decision_cycle_time)
            return None
        return local.astimezone(timezone.utc)

    def _default_cycle(self, conn: sqlite3.Connection) -> None:
        log.info("(dry-run) would run the decision cycle")


class Scheduler:
    """Long-running daemon loop: poll while the market is open, idle otherwise.

    ``poll_fn(conn, tickers)`` is invoked whenever a poll is due; the default
    is a harmless dry-run so the daemon can run before the Yahoo module exists.
    An optional ``news_cycle`` runs on its own interval regardless of market
    hours. ``clock`` and ``market`` are injectable for deterministic tests.
    """

    def __init__(
        self,
        settings: Settings,
        conn: sqlite3.Connection,
        poll_fn: PollFn | None = None,
        clock: Callable[[], datetime] | None = None,
        market: MarketCalendar | None = None,
        news_cycle: NewsCycle | None = None,
        decision_cycle: DecisionCycle | None = None,
        paper_refresh_fn: Callable[[sqlite3.Connection], None] | None = None,
    ) -> None:
        self.settings = settings
        self.conn = conn
        self.poll_fn = poll_fn or self._default_poll
        self.market = market or MarketCalendar(
            settings.market_exchange, settings.market_timezone
        )
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.poll_interval = timedelta(minutes=settings.intraday_poll_minutes)
        self.news_cycle = news_cycle
        self.decision_cycle = decision_cycle
        self.paper_refresh_fn = paper_refresh_fn
        self._last_poll = self._load_last_poll()
        self._stop = threading.Event()

    # ------------------------------------------------------------------ #
    # Public API

    def tick(self) -> float:
        """Run one scheduler pass; return seconds until the next wake-up."""
        now = self.clock()
        try:
            for cycle, name in ((self.news_cycle, "news"), (self.decision_cycle, "decision")):
                if cycle and cycle.due(now):
                    started = time.monotonic()
                    try:
                        cycle.run(now)
                    except Exception:
                        log.exception("%s cycle failed", name)
                    else:
                        log.info(
                            "%s cycle completed in %.1fs",
                            name, time.monotonic() - started,
                        )

            if self.market.is_open(now):
                if self._poll_due(now):
                    self._run_poll(now)
                wake = self._next_wakeup_while_open(now)
            else:
                wake = self.market.next_open(now)
                log.info(
                    "market closed; next open at %s", _fmt_local(wake, self.settings.market_timezone)
                )

            for cycle in (self.news_cycle, self.decision_cycle):
                if cycle:
                    next_cycle = cycle.next_run(now)
                    if next_cycle < wake:
                        wake = next_cycle
        except Exception:
            log.exception("scheduler pass failed")
            wake = now + timedelta(minutes=1)

        delay = (wake - now).total_seconds()
        if delay <= 0:
            delay = 1.0  # avoid a busy loop at session boundaries
        log.debug("next scheduler wake-up in %.0fs", delay)
        return delay

    def run_forever(self) -> None:
        """Run the daemon until ``stop()`` is called or interrupted."""
        log.info("scheduler started (exchange=%s)", self.settings.market_exchange)
        try:
            while not self._stop.is_set():
                delay = self.tick()
                self._stop.wait(delay)
        except KeyboardInterrupt:
            log.info("interrupted; shutting down")
        finally:
            log.info("scheduler stopped")

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ #
    # Internals

    def _load_last_poll(self) -> datetime | None:
        row = self.conn.execute(
            "SELECT MAX(captured_at) AS latest FROM price_snapshots"
        ).fetchone()
        latest = row["latest"]
        if not latest:
            return None
        parsed = datetime.fromisoformat(latest)
        return parsed.astimezone(timezone.utc)

    def _poll_due(self, now: datetime) -> bool:
        if self._last_poll is None:
            return True
        return (now - self._last_poll) >= self.poll_interval

    def _run_poll(self, now: datetime) -> None:
        tickers = db.get_active_tickers(self.conn)
        log.info("poll cycle due: %d ticker(s) at %s", len(tickers), now.isoformat())
        self.poll_fn(self.conn, tickers)
        self._last_poll = now
        if self.paper_refresh_fn is not None:
            try:
                self.paper_refresh_fn(self.conn)
            except Exception:
                log.exception("equity refresh failed after price poll")

    def _next_wakeup_while_open(self, now: datetime) -> datetime:
        assert self._last_poll is not None
        next_poll = self._last_poll + self.poll_interval
        next_close = self.market.next_close(now)
        return min(next_poll, next_close)

    def _default_poll(self, conn: sqlite3.Connection, tickers: list[str]) -> None:
        log.info("(dry-run) would poll %d ticker(s)", len(tickers))


def _fmt_local(when: datetime, tz: str) -> str:
    tz_info = __import__("zoneinfo").ZoneInfo(tz)
    return when.astimezone(tz_info).isoformat()


def _parse_utc(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def main() -> None:
    """Console entry point: load config, open the DB, run the daemon."""
    settings = get_settings()
    setup_logging(settings)
    conn = db.connect(settings.db_path_abs)
    db.initialize_database(conn)
    db.seed_tickers(conn, settings.watchlist)
    market = MarketCalendar(settings.market_exchange, settings.market_timezone)
    poll_fn = default_poll_fn(conn)
    news_cycle = NewsCycle(settings, conn, create_news_poll_fn(settings, conn))
    decision_cycle = DecisionCycle(
        settings, conn, create_decision_poll_fn(settings, conn), market=market,
        paper_fn=create_paper_cycle(settings, conn),
        report_fn=lambda c: generate_daily_report(c, settings),
    )
    Scheduler(
        settings, conn, poll_fn=poll_fn, news_cycle=news_cycle,
        decision_cycle=decision_cycle,
        paper_refresh_fn=lambda c: refresh_equity(c, settings),
    ).run_forever()


if __name__ == "__main__":
    main()