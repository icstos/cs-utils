"""
Time utilities module — Python 3.12 optimized version

Provides time-related utility functions:
- Time retrieval and formatting
- Timezone handling
- Timers
- Time difference calculation

Time formats:
1. timestamp: seconds since 1970-01-01 00:00:00 UTC (float)
2. struct_time: time.struct_time
3. string time: '2024-01-01 12:00:00'
4. datetime object: datetime.datetime
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable, Self

__all__ = [
    "Time",
    "DEFAULT_FORMAT",
    "ISO_FORMAT",
    "DATE_FORMAT",
    "LOCAL_TZ",
    "human_time",
    "now",
    "today",
    "yesterday",
    "tomorrow",
    "timestamp",
    "to_datetime",
    "to_timestamp",
    "to_str",
    "to_struct",
    "format_diff",
    "get_time_dif",
    "timer",
    "timeit",
]

# === Common format constants ===
DEFAULT_FORMAT = "%Y-%m-%d %H:%M:%S"
ISO_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
DATE_FORMAT = "%Y-%m-%d"


# === Local timezone (cached, avoid repeated detection) ===
def _detect_local_tz() -> timezone:
    """Detect local timezone via system offset, no third-party library needed."""
    offset = -time.altzone if time.daylight else -time.timezone
    return timezone(timedelta(seconds=offset))


LOCAL_TZ: timezone = _detect_local_tz()


# === Greeting (preserve original behavior) ===
def human_time() -> str:
    """Return English greeting based on current hour."""
    h = datetime.now().hour
    return (
        "Morning"
        if 5 <= h < 12
        else ("Afternoon" if 12 <= h < 18 else ("Evening" if 18 <= h < 22 else "Night"))
    )


# === Convenient functions (recommended calling style) ===
def now(tz: timezone | None = None) -> datetime:
    """Get current time. Uses local timezone if ``tz`` is None."""
    return datetime.now(tz or LOCAL_TZ)


def today() -> date:
    """Get today's date."""
    return date.today()


def yesterday() -> date:
    """Get yesterday's date."""
    return today() - timedelta(days=1)


def tomorrow() -> date:
    """Get tomorrow's date."""
    return today() + timedelta(days=1)


def timestamp() -> float:
    """Get current timestamp (seconds)."""
    return time.time()


def get_time_dif(start: float) -> timedelta:
    """Get elapsed time since ``start`` (timedelta)."""
    return timedelta(seconds=int(round(time.time() - start)))


def to_datetime(
    value: str | datetime | time.struct_time, fmt: str = DEFAULT_FORMAT
) -> datetime:
    """Convert string / struct_time to ``datetime`` with local timezone."""
    match value:
        case datetime() as dt:
            return dt if dt.tzinfo else dt.replace(tzinfo=LOCAL_TZ)
        case time.struct_time() as st:
            return datetime(*st[:6], tzinfo=LOCAL_TZ)
        case str() as s:
            return datetime.strptime(s, fmt).replace(tzinfo=LOCAL_TZ)
        case _:
            raise TypeError(f"unsupported type: {type(value).__name__}")


def to_timestamp(
    value: str | datetime | time.struct_time, fmt: str = DEFAULT_FORMAT
) -> float:
    """Convert to timestamp."""
    match value:
        case str() as s:
            return time.mktime(time.strptime(s, fmt))
        case datetime() as dt:
            return dt.timestamp()
        case time.struct_time() as st:
            return time.mktime(st)
        case _:
            raise TypeError(f"unsupported type: {type(value).__name__}")


def to_str(
    value: float | datetime | time.struct_time | None = None, fmt: str = DEFAULT_FORMAT
) -> str:
    """Format as string. Uses current time if ``value`` is None."""
    if value is None:
        return time.strftime(fmt, time.localtime())
    match value:
        case datetime() as dt:
            return dt.strftime(fmt)
        case time.struct_time() as st:
            return time.strftime(fmt, st)
        case int() | float() as t:
            return time.strftime(fmt, time.localtime(t))
        case _:
            raise TypeError(f"unsupported type: {type(value).__name__}")


def to_struct(
    value: str | float | datetime, fmt: str = DEFAULT_FORMAT
) -> time.struct_time:
    """Convert to struct_time."""
    match value:
        case str() as s:
            return time.strptime(s, fmt)
        case datetime() as dt:
            return dt.timetuple()
        case int() | float() as t:
            return time.localtime(t)
        case _:
            raise TypeError(f"unsupported type: {type(value).__name__}")


def format_diff(seconds: float) -> str:
    """Format seconds as human-readable string like 'X days X hours X min X sec'."""
    td = timedelta(seconds=int(seconds))
    days = td.days
    hours, rem = divmod(td.seconds, 3600)
    minutes, secs = divmod(rem, 60)
    parts = (
        f"{days}d" if days else "",
        f"{hours}h" if hours else "",
        f"{minutes}m" if minutes else "",
        f"{secs}s",
    )
    return "".join(parts) or "0s"


# === Timers ===
@contextmanager
def timeit(name: str = ""):
    """Timing context manager::

    with timeit("step"):
        do_something()
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    label = f"【{name}】" if name else ""
    print(f"{label}use_time: {elapsed:.6f}s")


def timer(func: Callable | None = None, *, print_result: bool = True):
    """Timing decorator, supports ``@timer`` and ``@timer(print_result=False)``."""

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            if not print_result:
                return f(*args, **kwargs)
            start = time.perf_counter()
            result = f(*args, **kwargs)
            print(f"【{f.__name__}】 used_time: {time.perf_counter() - start:.6f}s")
            return result

        return wrapper

    return decorator(func) if func is not None else decorator


# === Time class (backward compatible: preserve original method names and signatures) ===
@dataclass(slots=True)
class Time:
    """Time data class.

    Fields: ``year / month / day / hour / minute / second / microsecond``.
    ``add`` / ``subtract`` return new instances immutably (fixed the original bug
    where ``self.month = self.year + months`` incorrectly wrote to ``self.month``,
    and uses keyword-only arguments to avoid positional ambiguity).
    """

    year: int = 0
    month: int = 0
    day: int = 0
    hour: int = 0
    minute: int = 0
    second: int = 0
    microsecond: int = 0

    # --- Instance methods: offset from self ---
    def add(
        self,
        *,
        years: int = 0,
        months: int = 0,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        microseconds: int = 0,
    ) -> Self:
        """Add time interval to current instance, return new instance."""
        return self.__class__(
            year=self.year + years,
            month=self.month + months,
            day=self.day + days,
            hour=self.hour + hours,
            minute=self.minute + minutes,
            second=self.second + seconds,
            microsecond=self.microsecond + microseconds,
        )

    def subtract(
        self,
        *,
        years: int = 0,
        months: int = 0,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        microseconds: int = 0,
    ) -> Self:
        """Subtract time interval from current instance, return new instance."""
        return self.__class__(
            year=self.year - years,
            month=self.month - months,
            day=self.day - days,
            hour=self.hour - hours,
            minute=self.minute - minutes,
            second=self.second - seconds,
            microsecond=self.microsecond - microseconds,
        )

    def diff(self, other: Self) -> timedelta:
        """Calculate time difference from another ``Time`` (not implemented: fields don't form valid timestamps)."""
        raise NotImplementedError(
            "Time fields do not form valid timestamps, cannot compute difference directly"
        )

    # --- Static/class methods: preserve original Time public API ---
    @staticmethod
    def now() -> datetime:
        """Get current local time."""
        return now()

    @staticmethod
    def utcnow() -> datetime:
        """Get current UTC time."""
        return datetime.now(UTC)

    @staticmethod
    def today() -> str:
        """Get today's date string ``YYYY-MM-DD``."""
        return to_str(fmt=DATE_FORMAT)

    @staticmethod
    def yesterday() -> date:
        """Get yesterday's date."""
        return yesterday()

    @staticmethod
    def tomorrow() -> date:
        """Get tomorrow's date."""
        return tomorrow()

    @staticmethod
    def get_date() -> str:
        """Get today's date string."""
        return to_str(fmt=DATE_FORMAT)

    @classmethod
    def get_time_str(cls) -> str:
        """Get current time string ``YYYY-MM-DD-HH-MM-SS``."""
        return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    @staticmethod
    def to_timestamp(value: str | None, str_format: str = DEFAULT_FORMAT) -> float:
        """Convert string (may contain ``%f`` microseconds) to timestamp. Returns current timestamp if ``value`` is None."""
        if value is None:
            return time.time()
        if "%f" in str_format:
            return (
                time.mktime(time.strptime(value, str_format))
                + float(value[-3:]) / 1000.0
            )
        return time.mktime(time.strptime(value, str_format))

    @staticmethod
    def to_time_str(
        value: float | time.struct_time | None = None, str_format: str = DEFAULT_FORMAT
    ) -> str:
        """Convert value to time string."""
        return to_str(value, str_format)

    @staticmethod
    def to_time_struct(
        value: str | float, str_format: str = DEFAULT_FORMAT
    ) -> time.struct_time:
        """Convert value to struct_time."""
        return to_struct(value, str_format)

    @staticmethod
    def get_time_diff(timestamp_str_1: str, timestamp_str_2: str) -> float:
        """Time difference between two ``'%Y-%m-%d %H:%M:%S.%f'`` strings (seconds, absolute)."""
        fmt = "%Y-%m-%d %H:%M:%S.%f"

        def _to_ts(s: str) -> float:
            return time.mktime(time.strptime(s, fmt)) + float(s[-3:]) / 1000.0

        return abs(_to_ts(timestamp_str_1) - _to_ts(timestamp_str_2))

    @staticmethod
    def get_weekly_date() -> str:
        """Get weekly date range ``YYYY.MM.DD-YYYY.MM.DD``."""
        d = date.today()
        start = d - timedelta(days=d.weekday())
        end = start + timedelta(days=6)
        return _format_range(start, end)

    @staticmethod
    def get_month_date() -> str:
        """Get monthly date range ``YYYY.MM.DD-YYYY.MM.DD``."""
        d = date.today()
        first = d.replace(day=1)
        if first.month == 12:
            last = first.replace(year=first.year + 1, month=1) - timedelta(days=1)
        else:
            last = first.replace(month=first.month + 1) - timedelta(days=1)
        return _format_range(first, last)


def _format_range(start: date, end: date) -> str:
    """Format start and end dates as ``YYYY.MM.DD-YYYY.MM.DD``."""
    return (
        f"{start.year}.{start.month:0>2}.{start.day:0>2}"
        f"-{end.year}.{end.month:0>2}.{end.day:0>2}"
    )


if __name__ == "__main__":
    # Demonstrate convenient calls
    print("now           :", now())
    print("now(UTC)      :", now(UTC))
    print("today         :", today())
    print("timestamp     :", timestamp())
    print("time_str      :", Time.get_time_str())
    print("weekly        :", Time.get_weekly_date())
    print("monthly       :", Time.get_month_date())
    print("human_time    :", human_time())
    print("format_diff   :", format_diff(3725))
    print("to_timestamp  :", to_timestamp("2024-01-01 12:00:00"))

    @timer
    def demo():
        sum(i * i for i in range(1000))

    demo()
