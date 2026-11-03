"""
时间工具模块——面向 Python 3.12 优化

提供与时间相关的工具函数：
- 时间获取与格式化
- 时区处理
- 计时器
- 时间差计算

支持的时间表示：
1. timestamp：自 1970-01-01 00:00:00 UTC 起的秒数（float）
2. struct_time：time.struct_time
3. 字符串时间：'2024-01-01 12:00:00'
4. datetime 对象：datetime.datetime
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

# === 常用格式常量 ===
DEFAULT_FORMAT = "%Y-%m-%d %H:%M:%S"
ISO_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
DATE_FORMAT = "%Y-%m-%d"


# === 本地时区（缓存一次，避免重复探测） ===
def _detect_local_tz() -> timezone:
    """根据系统偏移探测本地时区，无需第三方库。"""
    offset = -time.altzone if time.daylight else -time.timezone
    return timezone(timedelta(seconds=offset))


LOCAL_TZ: timezone = _detect_local_tz()


# === 问候语（保持原有行为） ===
def human_time() -> str:
    """按当前小时返回英文问候语。"""
    h = datetime.now().hour
    return (
        "Morning"
        if 5 <= h < 12
        else ("Afternoon" if 12 <= h < 18 else ("Evening" if 18 <= h < 22 else "Night"))
    )


# === 便捷函数（推荐的调用方式） ===
def now(tz: timezone | None = None) -> datetime:
    """获取当前时间；``tz`` 为 None 时使用本地时区。"""
    return datetime.now(tz or LOCAL_TZ)


def today() -> date:
    """获取今天的日期。"""
    return date.today()


def yesterday() -> date:
    """获取昨天的日期。"""
    return today() - timedelta(days=1)


def tomorrow() -> date:
    """获取明天的日期。"""
    return today() + timedelta(days=1)


def timestamp() -> float:
    """获取当前时间戳（秒）。"""
    return time.time()


def get_time_dif(start: float) -> timedelta:
    """获取自 ``start`` 起经过的时间（timedelta）。"""
    return timedelta(seconds=int(round(time.time() - start)))


def to_datetime(
    value: str | datetime | time.struct_time, fmt: str = DEFAULT_FORMAT
) -> datetime:
    """把字符串 / struct_time 转换为带本地时区的 ``datetime``。"""
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
    """转换为时间戳。"""
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
    """格式化为字符串；``value`` 为 None 时使用当前时间。"""
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
    """转换为 struct_time。"""
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
    """把秒数格式化为可读字符串，如“X天X小时X分X秒”。"""
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


# === 计时器 ===
@contextmanager
def timeit(name: str = ""):
    """计时上下文管理器::

    with timeit("step"):
        do_something()
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    label = f"【{name}】" if name else ""
    print(f"{label}use_time: {elapsed:.6f}s")


def timer(func: Callable | None = None, *, print_result: bool = True):
    """计时装饰器，支持 ``@timer`` 与 ``@timer(print_result=False)``。"""

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


# === Time 类（向后兼容：保留原有方法名与签名） ===
@dataclass(slots=True)
class Time:
    """时间数据类。

    字段：``year / month / day / hour / minute / second / microsecond``。
    ``add`` / ``subtract`` 以不可变方式返回新实例（修正了原先
    ``self.month = self.year + months`` 误写入 ``self.month`` 的缺陷），
    并改用仅关键字参数，避免位置参数歧义。
    """

    year: int = 0
    month: int = 0
    day: int = 0
    hour: int = 0
    minute: int = 0
    second: int = 0
    microsecond: int = 0

    # --- 实例方法：基于自身做偏移 ---
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
        """在当前实例上增加时间间隔，返回新实例。"""
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
        """在当前实例上减去时间间隔，返回新实例。"""
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
        """与另一个 ``Time`` 计算时间差（未实现：各字段拼不成合法时间戳）。"""
        raise NotImplementedError(
            "Time fields do not form valid timestamps, cannot compute difference directly"
        )

    # --- 静态 / 类方法：保留原有 Time 公开接口 ---
    @staticmethod
    def now() -> datetime:
        """获取当前本地时间。"""
        return now()

    @staticmethod
    def utcnow() -> datetime:
        """获取当前 UTC 时间。"""
        return datetime.now(UTC)

    @staticmethod
    def today() -> str:
        """获取今天的日期字符串 ``YYYY-MM-DD``。"""
        return to_str(fmt=DATE_FORMAT)

    @staticmethod
    def yesterday() -> date:
        """获取昨天的日期。"""
        return yesterday()

    @staticmethod
    def tomorrow() -> date:
        """获取明天的日期。"""
        return tomorrow()

    @staticmethod
    def get_date() -> str:
        """获取今天的日期字符串。"""
        return to_str(fmt=DATE_FORMAT)

    @classmethod
    def get_time_str(cls) -> str:
        """获取当前时间字符串 ``YYYY-MM-DD-HH-MM-SS``。"""
        return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    @staticmethod
    def to_timestamp(value: str | None, str_format: str = DEFAULT_FORMAT) -> float:
        """把字符串（可含 ``%f`` 微秒）转换为时间戳；``value`` 为 None 时返回当前时间戳。"""
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
        """把时间值转换为字符串。"""
        return to_str(value, str_format)

    @staticmethod
    def to_time_struct(
        value: str | float, str_format: str = DEFAULT_FORMAT
    ) -> time.struct_time:
        """把时间值转换为 struct_time。"""
        return to_struct(value, str_format)

    @staticmethod
    def get_time_diff(timestamp_str_1: str, timestamp_str_2: str) -> float:
        """两个 ``'%Y-%m-%d %H:%M:%S.%f'`` 字符串之间的时间差（秒，取绝对值）。"""
        fmt = "%Y-%m-%d %H:%M:%S.%f"

        def _to_ts(s: str) -> float:
            return time.mktime(time.strptime(s, fmt)) + float(s[-3:]) / 1000.0

        return abs(_to_ts(timestamp_str_1) - _to_ts(timestamp_str_2))

    @staticmethod
    def get_weekly_date() -> str:
        """获取本周的日期区间 ``YYYY.MM.DD-YYYY.MM.DD``。"""
        d = date.today()
        start = d - timedelta(days=d.weekday())
        end = start + timedelta(days=6)
        return _format_range(start, end)

    @staticmethod
    def get_month_date() -> str:
        """获取本月的日期区间 ``YYYY.MM.DD-YYYY.MM.DD``。"""
        d = date.today()
        first = d.replace(day=1)
        if first.month == 12:
            last = first.replace(year=first.year + 1, month=1) - timedelta(days=1)
        else:
            last = first.replace(month=first.month + 1) - timedelta(days=1)
        return _format_range(first, last)


def _format_range(start: date, end: date) -> str:
    """把起止日期格式化为 ``YYYY.MM.DD-YYYY.MM.DD``。"""
    return (
        f"{start.year}.{start.month:0>2}.{start.day:0>2}"
        f"-{end.year}.{end.month:0>2}.{end.day:0>2}"
    )


if __name__ == "__main__":
    # 演示便捷调用
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
