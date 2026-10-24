"""采样、统计与呈现：``watch`` / ``sample`` / 表格 / CSV。

时间语义
--------
* 采样间隔按"开始时刻"计时，长时间采样不会累积漂移；
* 单次采样本身耗时不从间隔里扣除，避免把 NVML 的抖动放大成频率误差；
* 采样期间 GPU 掉线会抛 :class:`~cs.nvidia.errors.GPULostError`，
  生成器随之终止，调用方可以用 ``except`` 兜住。
"""

from __future__ import annotations

import csv
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import GPUStatus, Severity, bytes_to

if TYPE_CHECKING:
    from .core import GPUHandle

__all__ = ["SampleSeries", "iter_status", "render_table", "sample_status"]

# CSV / 表格里用到的数值字段（顺序即列序）
_NUMERIC_FIELDS: tuple[tuple[str, str], ...] = (
    ("gpu_util", "%"),
    ("mem_util", "%"),
    ("mem_used", "MiB"),
    ("temperature", "°C"),
    ("power", "W"),
    ("fan", "%"),
    ("sm_clock", "MHz"),
)


def _validate(interval: float, count: int | None, duration: float | None) -> None:
    if interval <= 0:
        raise ValueError(f"interval 必须为正数，收到 {interval!r}")
    if count is None and duration is None:
        raise ValueError("count 与 duration 至少提供一个，避免无限采样")
    if count is not None and count < 1:
        raise ValueError(f"count 必须 ≥ 1，收到 {count!r}")
    if duration is not None and duration <= 0:
        raise ValueError(f"duration 必须为正数，收到 {duration!r}")


def iter_status(
    handle: GPUHandle,
    *,
    interval: float = 1.0,
    count: int | None = None,
    duration: float | None = None,
) -> Iterator[GPUStatus]:
    """按 ``interval`` 秒产出状态快照，到 ``count`` 次或 ``duration`` 秒为止。"""
    _validate(interval, count, duration)
    deadline = None if duration is None else time.monotonic() + duration
    produced = 0
    while True:
        if count is not None and produced >= count:
            return
        if deadline is not None and time.monotonic() >= deadline:
            return

        started = time.monotonic()
        yield handle.status()
        produced += 1

        if count is not None and produced >= count:
            return
        wait = interval - (time.monotonic() - started)  # 扣掉采样耗时，稳定频率
        if deadline is not None:
            wait = min(wait, deadline - time.monotonic())
            if wait <= 0:
                return
        time.sleep(max(wait, 0.0))


def _aggregate(values: Iterable[float | int | None]) -> dict[str, float] | None:
    data = [float(value) for value in values if value is not None]
    if not data:
        return None
    return {
        "min": round(min(data), 2),
        "max": round(max(data), 2),
        "avg": round(sum(data) / len(data), 2),
    }


@dataclass(frozen=True, slots=True)
class SampleSeries:
    """一段采样结果及其统计。"""

    index: int
    uuid: str
    name: str
    interval: float
    states: tuple[GPUStatus, ...]

    @property
    def count(self) -> int:
        return len(self.states)

    @property
    def window(self) -> float:
        """首末采样的实际时间跨度（秒）。"""
        if len(self.states) < 2:
            return 0.0
        return round(self.states[-1].timestamp - self.states[0].timestamp, 3)

    def stats(self) -> dict[str, dict[str, float] | None]:
        """各指标的 min / max / avg（跳过读不到的样本，显存统一为 MiB）。"""
        return {
            "gpu_util": _aggregate(state.utilization.gpu for state in self.states),
            "mem_util": _aggregate(state.utilization.memory for state in self.states),
            "mem_used": _aggregate(
                bytes_to(state.memory.used, "MiB", digits=0) if state.memory else None
                for state in self.states
            ),
            "temperature": _aggregate(state.temperature for state in self.states),
            "power": _aggregate(state.power.watts for state in self.states),
            "fan": _aggregate(state.fan_percent for state in self.states),
            "sm_clock": _aggregate(state.clocks.sm.current for state in self.states),
        }

    def to_dict(self, *, include_states: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "uuid": self.uuid,
            "name": self.name,
            "interval": self.interval,
            "count": self.count,
            "window_seconds": self.window,
            "stats": self.stats(),
        }
        if include_states:
            payload["states"] = [state.to_dict() for state in self.states]
        return payload

    def to_csv(self, path: str | Path) -> Path:
        """导出为 CSV（UTF-8 BOM，Excel 直接双击不乱码）。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "timestamp",
            "iso_time",
            "gpu_index",
            "uuid",
            "name",
            "gpu_util_pct",
            "mem_util_pct",
            "mem_used_mib",
            "mem_total_mib",
            "temperature_c",
            "power_w",
            "fan_pct",
            "pstate",
            "sm_clock_mhz",
            "throttle",
        ]
        with target.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for state in self.states:
                writer.writerow(
                    [
                        f"{state.timestamp:.3f}",
                        time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(state.timestamp)
                        ),
                        state.index,
                        state.uuid,
                        state.name,
                        state.utilization.gpu,
                        state.utilization.memory,
                        bytes_to(state.memory.used, "MiB", digits=0)
                        if state.memory
                        else None,
                        bytes_to(state.memory.total, "MiB", digits=0)
                        if state.memory
                        else None,
                        state.temperature,
                        state.power.watts,
                        state.fan_percent,
                        int(state.pstate),
                        state.clocks.sm.current,
                        str(state.throttle) if state.throttle else None,
                    ]
                )
        return target

    def __str__(self) -> str:
        stats = self.stats()
        parts = [f"[{self.index}] {self.name} 采样 {self.count} 次 / {self.window}s"]
        for field, unit in _NUMERIC_FIELDS:
            if (entry := stats.get(field)) is not None:
                parts.append(
                    f"{field}: avg {entry['avg']}{unit} "
                    f"(min {entry['min']} / max {entry['max']})"
                )
        return "\n  ".join(parts)


def sample_status(
    handle: GPUHandle, *, interval: float = 1.0, count: int = 10
) -> SampleSeries:
    """同步采样 ``count`` 次并打包成 :class:`SampleSeries`。"""
    states = tuple(iter_status(handle, interval=interval, count=count, duration=None))
    return SampleSeries(
        index=handle.index,
        uuid=handle.uuid,
        name=handle.name,
        interval=interval,
        states=states,
    )


# ---------------------------------------------------------------- 呈现
_HEADERS = (
    ("index", "GPU", 3),
    ("name", "Name", 4),
    ("util", "Util", 4),
    ("memory", "Memory", 6),
    ("temp", "Temp", 4),
    ("power", "Power", 5),
    ("fan", "Fan", 3),
    ("pstate", "P-State", 7),
    ("throttle", "Throttle", 8),
)


def _row(index: int, state: GPUStatus) -> dict[str, str]:
    memory = (
        f"{bytes_to(state.memory.used)}/{bytes_to(state.memory.total)}"
        if state.memory
        else "?"
    )
    power = (
        f"{state.power.watts:.0f}/{state.power.limit_watts:.0f}"
        if state.power.watts is not None and state.power.limit_watts is not None
        else "?"
    )
    return {
        "index": str(index),
        "name": state.name,
        "util": f"{state.utilization.gpu}%"
        if state.utilization.gpu is not None
        else "-",
        "memory": f"{memory} GiB",
        "temp": f"{state.temperature:.0f}°C" if state.temperature is not None else "-",
        "power": f"{power}W",
        "fan": f"{state.fan_percent}%" if state.fan_percent is not None else "-",
        "pstate": str(state.pstate),
        "throttle": str(state.throttle) if state.throttle else "-",
    }


def render_table(states: Sequence[GPUStatus]) -> str:
    """把若干快照渲染成等宽文本表（``nvidia-smi`` 风格）。"""
    if not states:
        return "没有可显示的 GPU"

    rows = [_row(position, state) for position, state in enumerate(states)]
    widths = {
        key: max(len(title), *(len(row[key]) for row in rows))
        for key, title, _ in _HEADERS
    }
    header = "  ".join(title.ljust(widths[key]) for key, title, _ in _HEADERS)
    separator = "  ".join("-" * widths[key] for key, _, _ in _HEADERS)
    body = [
        "  ".join(row[key].ljust(widths[key]) for key, _, _ in _HEADERS) for row in rows
    ]
    return "\n".join([header, separator, *body])


def summarize_health(reports: Iterable[Any]) -> str:
    """把多份体检结果拼成一段人读文本。"""
    lines = []
    for report in reports:
        flag = "OK " if report.level is Severity.INFO else report.level.value.upper()
        lines.append(f"{flag:<9} {report}")
    return "\n".join(lines) or "没有可显示的 GPU"
