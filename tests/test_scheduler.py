"""Tests for the scheduler state machine (fake clock/market) and the real
market calendar."""

from datetime import date, datetime, timedelta, timezone

import pytest

from rufus import db
from rufus.config import Settings
from rufus.decision import JOB_DECISION
from rufus.scheduler import DecisionCycle, JOB_NEWS, MarketCalendar, NewsCycle, Scheduler

UTC = timezone.utc

# 2026-09-22 is a Tuesday and a regular NSE trading day.
# NSE session: 09:15-15:30 IST == 03:45-10:00 UTC.
_OPEN_TUE = datetime(2026, 9, 22, 5, 0, tzinfo=UTC)          # 10:30 IST
_CLOSE_TUE = datetime(2026, 9, 22, 10, 0, tzinfo=UTC)         # 15:30 IST close
_OPEN_WED = datetime(2026, 9, 23, 3, 45, tzinfo=UTC)          # 09:15 IST
_OPEN_MON = datetime(2026, 9, 28, 3, 45, tzinfo=UTC)          # 2026-09-28


class FakeMarket:
    def __init__(self, is_open=True, next_open=None, next_close=None):
        self._is_open = is_open
        self._next_open = next_open or _OPEN_WED
        self._next_close = next_close or _CLOSE_TUE

    def is_open(self, at):
        return self._is_open

    def next_open(self, at):
        return self._next_open

    def next_close(self, at):
        return self._next_close


class FakeClock:
    def __init__(self, times):
        self.times = list(times)

    def __call__(self):
        return self.times.pop(0)


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    db.initialize_database(c)
    yield c
    c.close()


def make_scheduler(conn, *, poll_fn=None, clock=None, market=None, interval_minutes=30):
    settings = Settings(
        _env_file=None,
        intraday_poll_minutes=interval_minutes,
        market_exchange="XNSE",
        market_timezone="Asia/Kolkata",
    )
    return Scheduler(settings, conn, poll_fn=poll_fn, clock=clock, market=market)


def test_market_closed_sleeps_until_next_open(conn):
    calls = []
    scheduler = make_scheduler(
        conn,
        poll_fn=lambda c, t: calls.append(1),
        clock=FakeClock([_CLOSE_TUE + timedelta(seconds=1)]),
        market=FakeMarket(is_open=False, next_open=_OPEN_WED),
    )
    delay = scheduler.tick()
    assert calls == []
    assert delay == pytest.approx((_OPEN_WED - (_CLOSE_TUE + timedelta(seconds=1))).total_seconds())


def test_open_market_polls_when_due(conn):
    calls = []
    clock = FakeClock([_OPEN_TUE])
    scheduler = make_scheduler(
        conn,
        poll_fn=lambda c, t: calls.append(1),
        clock=clock,
        market=FakeMarket(),
    )
    delay = scheduler.tick()
    assert calls == [1]
    assert delay == pytest.approx(30 * 60)


def test_no_repoll_within_interval(conn):
    calls = []
    clock = FakeClock([_OPEN_TUE, _OPEN_TUE + timedelta(minutes=5)])
    scheduler = make_scheduler(
        conn,
        poll_fn=lambda c, t: calls.append(1),
        clock=clock,
        market=FakeMarket(),
    )
    scheduler.tick()
    delay2 = scheduler.tick()
    assert calls == [1]
    assert delay2 == pytest.approx(25 * 60)


def test_repolls_after_interval(conn):
    calls = []
    clock = FakeClock(
        [_OPEN_TUE, _OPEN_TUE + timedelta(minutes=31)]
    )
    scheduler = make_scheduler(
        conn,
        poll_fn=lambda c, t: calls.append(1),
        clock=clock,
        market=FakeMarket(next_close=_CLOSE_TUE),
    )
    scheduler.tick()
    scheduler.tick()
    assert calls == [1, 1]


def test_wake_capped_at_market_close(conn):
    calls = []
    clock = FakeClock([_CLOSE_TUE - timedelta(minutes=10)])
    scheduler = make_scheduler(
        conn,
        poll_fn=lambda c, t: calls.append(1),
        clock=clock,
        market=FakeMarket(is_open=True, next_close=_CLOSE_TUE),
        interval_minutes=60,
    )
    delay = scheduler.tick()
    # Poll fires, but the next wake-up is capped at the 15:30 IST close, not
    # the next poll slot an hour away.
    assert calls == [1]
    assert delay == pytest.approx(10 * 60)


def test_restart_without_history_polls_immediately(conn):
    # Fresh DB (no price snapshots) + market open -> poll on first tick.
    calls = []
    scheduler = make_scheduler(
        conn,
        poll_fn=lambda c, t: calls.append(1),
        clock=FakeClock([_OPEN_TUE]),
        market=FakeMarket(),
    )
    scheduler.tick()
    assert calls == [1]


def test_restart_reloads_last_poll_from_db(conn):
    db.upsert_ticker(conn, "AAPL")

    # Simulate a previous run that polled 5 minutes ago (fresh snapshot).
    db.insert_price_snapshot(
        conn,
        "AAPL",
        captured_at=(_OPEN_TUE - timedelta(minutes=5)).isoformat(),
        price=100.0,
    )

    # New process starts at 10:30 IST: last poll was 5 min ago within the
    # 30-min interval, so it must NOT re-poll and must sleep ~25 min more.
    calls = []
    s = make_scheduler(
        conn,
        poll_fn=lambda c, t: calls.append(1),
        clock=FakeClock([_OPEN_TUE]),
        market=FakeMarket(next_close=_CLOSE_TUE),
    )
    delay = s.tick()
    assert calls == []
    assert delay == pytest.approx(25 * 60)

    # And when the interval HAS fully elapsed, a restarted process polls.
    calls2 = []
    s2 = make_scheduler(
        conn,
        poll_fn=lambda c, t: calls2.append(1),
        clock=FakeClock([_OPEN_TUE + timedelta(minutes=31)]),
        market=FakeMarket(next_close=_CLOSE_TUE),
    )
    s2.tick()
    assert calls2 == [1]


def test_poll_fn_receives_active_tickers(conn):
    db.upsert_ticker(conn, "AAPL")
    db.upsert_ticker(conn, "MSFT", active=False)
    seen = []
    scheduler = make_scheduler(
        conn,
        poll_fn=lambda c, t: seen.append(t),
        clock=FakeClock([_OPEN_TUE]),
        market=FakeMarket(),
    )
    scheduler.tick()
    assert seen == [["AAPL"]]


# ---------------------------------------------------------------------------
# News cycle (market-independent)
# ---------------------------------------------------------------------------

def make_news_cycle(conn, poll_fn=None, hours=24):
    settings = Settings(_env_file=None, news_poll_hours=hours)
    return NewsCycle(settings, conn, poll_fn=poll_fn)


def test_news_cycle_runs_when_never_done_and_gaps(conn):
    calls = []
    cycle = make_news_cycle(conn, lambda c: calls.append(1))
    now = _OPEN_TUE
    assert cycle.due(now) is True
    cycle.run(now)
    assert calls == [1]
    assert cycle.due(now + timedelta(hours=1)) is False
    assert cycle.due(now + timedelta(hours=25)) is True


def test_news_cycle_restart_safe_via_job_state(conn):
    first = make_news_cycle(conn, lambda c: None)
    first.run(_OPEN_TUE)
    second = make_news_cycle(conn, lambda c: None)
    assert second.due(_OPEN_TUE + timedelta(hours=5)) is False


def test_scheduler_runs_news_cycle_when_market_closed(conn):
    calls = []
    settings = Settings(_env_file=None, intraday_poll_minutes=30, news_poll_hours=24)
    news_cycle = NewsCycle(settings, conn, poll_fn=lambda c: calls.append("news"))
    scheduler = Scheduler(
        settings, conn, news_cycle=news_cycle,
        clock=FakeClock([_CLOSE_TUE + timedelta(seconds=1)]),
        market=FakeMarket(is_open=False, next_open=_OPEN_WED),
    )
    delay = scheduler.tick()
    assert calls == ["news"]
    assert delay == pytest.approx((_OPEN_WED - (_CLOSE_TUE + timedelta(seconds=1))).total_seconds())


def test_scheduler_news_wake_can_precede_next_open(conn):
    now = _CLOSE_TUE + timedelta(seconds=1)
    # Last news run 20h ago with a 24h interval -> next due in 4h, well
    # before the next market open (~17.5h away).
    db.set_job_last_run(conn, JOB_NEWS, (now - timedelta(hours=20)).isoformat())
    calls = []
    settings = Settings(_env_file=None, intraday_poll_minutes=30, news_poll_hours=24)
    news_cycle = NewsCycle(settings, conn, poll_fn=lambda c: calls.append("news"))
    scheduler = Scheduler(
        settings, conn, news_cycle=news_cycle,
        clock=FakeClock([now]),
        market=FakeMarket(is_open=False, next_open=_OPEN_WED),
    )
    delay = scheduler.tick()
    assert calls == []
    assert delay == pytest.approx(4 * 3600)


def test_scheduler_news_cycle_failure_is_nonfatal(conn):
    calls = []

    def bad_news(c):
        raise RuntimeError("boom")

    def bad_news_wrap(c):
        bad_news(c)

    settings = Settings(_env_file=None, intraday_poll_minutes=30, news_poll_hours=24)
    news_cycle = NewsCycle(settings, conn, poll_fn=bad_news_wrap)
    scheduler = Scheduler(
        settings, conn, poll_fn=lambda c, t: calls.append("intraday"),
        news_cycle=news_cycle,
        clock=FakeClock([_OPEN_TUE]),
        market=FakeMarket(),
    )
    scheduler.tick()
    assert calls == ["intraday"]


# ---------------------------------------------------------------------------
# Decision cycle (once per trading day, at the configured local slot)
# ---------------------------------------------------------------------------

# 15:30 IST == 10:00 UTC decision slot.
_DECISION_TUE = datetime(2026, 9, 22, 10, 0, tzinfo=UTC)
_DECISION_WED = datetime(2026, 9, 23, 10, 0, tzinfo=UTC)
_DECISION_MON = datetime(2026, 9, 28, 10, 0, tzinfo=UTC)


class FakeTradingDayMarket:
    def __init__(self, trading_days):
        self._days = set(trading_days)

    def is_trading_day(self, day):
        return day in self._days


def make_decision_cycle(conn, cycle_fn=None, trading_days=None):
    settings = Settings(
        _env_file=None,
        decision_cycle_time="15:30",
        market_timezone="Asia/Kolkata",
    )
    return DecisionCycle(
        settings,
        conn,
        cycle_fn=cycle_fn or (lambda c: None),
        market=FakeTradingDayMarket(trading_days or {date(2026, 9, 22)}),
    )


def test_decision_cycle_not_due_before_slot(conn):
    cycle = make_decision_cycle(conn)
    assert cycle.due(_DECISION_TUE - timedelta(minutes=1)) is False


def test_decision_cycle_due_at_slot_on_trading_day(conn):
    calls = []
    cycle = make_decision_cycle(conn, lambda c: calls.append("decision"))
    assert cycle.due(_DECISION_TUE) is True
    cycle.run(_DECISION_TUE)
    assert calls == ["decision"]


def test_decision_cycle_not_due_on_holiday(conn):
    cycle = make_decision_cycle(conn, trading_days={date(2026, 9, 23)})
    assert cycle.due(_DECISION_TUE) is False


def test_decision_cycle_restart_safe_via_job_state(conn):
    first = make_decision_cycle(conn)
    first.run(_DECISION_TUE)
    second = make_decision_cycle(conn)
    assert second.due(_DECISION_TUE + timedelta(hours=3)) is False


def test_decision_cycle_next_run_before_slot_is_today(conn):
    cycle = make_decision_cycle(conn)
    assert cycle.next_run(_DECISION_TUE - timedelta(hours=1)) == _DECISION_TUE


def test_decision_cycle_next_run_after_run_rolls_to_next_trading_day(conn):
    cycle = make_decision_cycle(conn, trading_days={date(2026, 9, 22), date(2026, 9, 23)})
    cycle.run(_DECISION_TUE)
    assert cycle.next_run(_DECISION_TUE + timedelta(minutes=5)) == _DECISION_WED


def test_decision_cycle_next_run_skips_weekend(conn):
    cycle = make_decision_cycle(conn, trading_days={date(2026, 9, 22), date(2026, 9, 28)})
    cycle.run(_DECISION_TUE)
    saturday = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
    assert cycle.next_run(saturday) == _DECISION_MON


def test_scheduler_runs_decision_cycle_when_market_closed(conn):
    calls = []
    settings = Settings(_env_file=None, intraday_poll_minutes=30, decision_cycle_time="15:30")
    decision_cycle = make_decision_cycle(conn, lambda c: calls.append("decision"))
    scheduler = Scheduler(
        settings, conn, decision_cycle=decision_cycle,
        clock=FakeClock([_DECISION_TUE]),
        market=FakeMarket(is_open=False, next_open=_OPEN_WED),
    )
    delay = scheduler.tick()
    assert calls == ["decision"]


def test_scheduler_decision_cycle_failure_is_nonfatal(conn):
    calls = []

    def bad_cycle(c):
        raise RuntimeError("boom")

    settings = Settings(_env_file=None, intraday_poll_minutes=30)
    decision_cycle = make_decision_cycle(conn, bad_cycle)
    scheduler = Scheduler(
        settings, conn, poll_fn=lambda c, t: calls.append("intraday"),
        decision_cycle=decision_cycle,
        clock=FakeClock([_OPEN_TUE]),
        market=FakeMarket(),
    )
    scheduler.tick()
    assert calls == ["intraday"]


def test_scheduler_decision_wake_factors_into_delay(conn):
    now = _CLOSE_TUE + timedelta(seconds=1)
    db.set_job_last_run(conn, JOB_DECISION, (now - timedelta(days=3)).isoformat())
    settings = Settings(_env_file=None, intraday_poll_minutes=30, decision_cycle_time="15:30")
    decision_cycle = DecisionCycle(
        settings, conn, cycle_fn=lambda c: None,
        clock=FakeClock([now]),
        market=FakeTradingDayMarket({date(2026, 9, 23)}),
    )
    scheduler = Scheduler(
        settings, conn, decision_cycle=decision_cycle,
        clock=FakeClock([now]),
        market=FakeMarket(is_open=False, next_open=_OPEN_WED),
    )
    # Decision slot 2026-09-23 15:30 IST (10:00 UTC) < next open 03:45 UTC?
    # No: next open (03:45) precedes the decision slot, so wake stays the open.
    block = scheduler.tick()
    assert block == pytest.approx((_OPEN_WED - now).total_seconds())


def test_decision_cycle_runs_paper_fn_after_cycle(conn):
    calls = []
    settings = Settings(_env_file=None, decision_cycle_time="15:30")
    cycle = DecisionCycle(
        settings, conn,
        cycle_fn=lambda c: calls.append("decision"),
        paper_fn=lambda c: calls.append("paper"),
        market=FakeTradingDayMarket({date(2026, 9, 22)}),
    )
    cycle.run(_DECISION_TUE)
    assert calls == ["decision", "paper"]


def test_decision_cycle_runs_report_fn_after_paper(conn):
    calls = []
    settings = Settings(_env_file=None, decision_cycle_time="15:30")
    cycle = DecisionCycle(
        settings, conn,
        cycle_fn=lambda c: calls.append("decision"),
        paper_fn=lambda c: calls.append("paper"),
        report_fn=lambda c: calls.append("report"),
        market=FakeTradingDayMarket({date(2026, 9, 22)}),
    )
    cycle.run(_DECISION_TUE)
    assert calls == ["decision", "paper", "report"]


def test_decision_cycle_report_fn_failure_is_nonfatal(conn):
    settings = Settings(_env_file=None, decision_cycle_time="15:30")
    cycle = DecisionCycle(
        settings, conn,
        cycle_fn=lambda c: None,
        paper_fn=lambda c: None,
        report_fn=lambda c: (_ for _ in ()).throw(RuntimeError("boom")),
        market=FakeTradingDayMarket({date(2026, 9, 22)}),
    )
    cycle.run(_DECISION_TUE)  # must not raise


def test_scheduler_invokes_paper_refresh_after_poll(conn):
    calls = []
    settings = Settings(_env_file=None, intraday_poll_minutes=30)
    scheduler = Scheduler(
        settings, conn,
        poll_fn=lambda c, t: calls.append("poll"),
        paper_refresh_fn=lambda c: calls.append("refresh"),
        clock=FakeClock([_OPEN_TUE]),
        market=FakeMarket(),
    )
    scheduler.tick()
    assert calls == ["poll", "refresh"]


def test_scheduler_paper_refresh_failure_is_nonfatal(conn):
    settings = Settings(_env_file=None, intraday_poll_minutes=30)
    scheduler = Scheduler(
        settings, conn,
        poll_fn=lambda c, t: None,
        paper_refresh_fn=lambda c: (_ for _ in ()).throw(RuntimeError("boom")),
        clock=FakeClock([_OPEN_TUE]),
        market=FakeMarket(),
    )
    scheduler.tick()  # must not raise


def test_decision_cycle_logs_durations(conn, caplog):
    calls = []
    settings = Settings(_env_file=None, decision_cycle_time="15:30")
    cycle = DecisionCycle(
        settings, conn,
        cycle_fn=lambda c: calls.append("decision"),
        paper_fn=lambda c: calls.append("paper"),
        report_fn=lambda c: calls.append("report"),
        market=FakeTradingDayMarket({date(2026, 9, 22)}),
    )
    with caplog.at_level("INFO", logger="rufus.scheduler"):
        cycle.run(_DECISION_TUE)
    assert "decision cycle completed in" in caplog.text
    assert "paper cycle completed in" in caplog.text
    assert "report generated in" in caplog.text


def test_scheduler_logs_cycle_completion(conn, caplog):
    settings = Settings(_env_file=None, intraday_poll_minutes=30, news_poll_hours=24)
    news_cycle = NewsCycle(settings, conn, poll_fn=lambda c: None)
    scheduler = Scheduler(
        settings, conn, news_cycle=news_cycle,
        clock=FakeClock([_OPEN_TUE]),
        market=FakeMarket(),
    )
    with caplog.at_level("INFO", logger="rufus.scheduler"):
        scheduler.tick()
    assert "news cycle completed in" in caplog.text


# ---------------------------------------------------------------------------
# Real market calendar
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def calendar():
    return MarketCalendar()


def test_real_calendar_open_during_session(calendar):
    assert calendar.is_open(datetime(2026, 9, 22, 5, 0, tzinfo=UTC))       # 10:30 IST Tue
    assert calendar.is_open(datetime(2026, 9, 22, 3, 45, tzinfo=UTC))      # 09:15 IST open
    assert calendar.is_open(datetime(2026, 9, 22, 9, 59, tzinfo=UTC))      # 14:59 IST
    assert not calendar.is_open(datetime(2026, 9, 22, 10, 1, tzinfo=UTC))  # after close
    assert not calendar.is_open(datetime(2026, 9, 22, 1, 0, tzinfo=UTC))   # pre-open


def test_real_calendar_closed_on_weekend(calendar):
    saturday = datetime(2026, 9, 26, 15, 0, tzinfo=UTC)
    assert not calendar.is_open(saturday)


def test_real_calendar_next_open_skips_weekend(calendar):
    friday_close = datetime(2026, 9, 25, 10, 1, tzinfo=UTC)
    assert calendar.next_open(friday_close) == _OPEN_MON


def test_real_calendar_next_open_after_close_is_next_day(calendar):
    tue_close = datetime(2026, 9, 22, 10, 1, tzinfo=UTC)
    assert calendar.next_open(tue_close) == _OPEN_WED


def test_real_calendar_next_close_while_open(calendar):
    assert calendar.next_close(datetime(2026, 9, 22, 5, 0, tzinfo=UTC)) == _CLOSE_TUE