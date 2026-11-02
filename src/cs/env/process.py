"""进程的列举、检索与结束。

- 列表用 ``process_iter(attrs=...)`` 批量取字段，比逐个 ``Process`` 快一个量级；
- 无权限的进程（系统进程）用 ``ad_value=None`` 降级成空值，不中断整轮列举；
- 结束进程前做**安全护栏**：拒绝 ``pid<=0``、当前进程、父进程。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import psutil

from .errors import ProcessError, ProcessNotFoundError
from .models import ProcessInfo
from .system import gpu_processes

__all__ = ["current", "kill", "process", "processes", "top"]

#: 列表用字段。刻意不含 ``exe`` / ``num_threads``：在 Windows 上分别多耗
#: ~0.4s / ~2s（每个进程都要 OpenProcess），列表场景收益极低，留给详情接口。
_LIST_ATTRS: tuple[str, ...] = (
    "pid",
    "ppid",
    "name",
    "username",
    "status",
    "cpu_percent",
    "memory_info",
    "memory_percent",
    "create_time",
    "cmdline",
)

#: 详情用字段：列表字段 + 可执行文件与线程数
_DETAIL_ATTRS: tuple[str, ...] = (*_LIST_ATTRS, "exe", "num_threads")

type SortKey = str


def _join_cmd(parts: list[str] | tuple[str, ...]) -> str:
    """把 argv 拼成便于阅读的一行（含空格的参数加引号）。"""
    return " ".join(f'"{part}"' if " " in part else part for part in parts)


def current() -> ProcessInfo:
    """当前进程（脚本自身）的信息。"""
    return process(os.getpid())


def _build(
    info: Mapping[str, Any], gpu: Mapping[int, tuple[int, int | None]]
) -> ProcessInfo:
    pid = int(info.get("pid") or 0)
    memory = info.get("memory_info")
    rss = int(getattr(memory, "rss", 0) or 0)
    cmdline = info.get("cmdline") or []
    gpu_index, gpu_memory = gpu.get(pid, (None, None))
    return ProcessInfo(
        pid=pid,
        ppid=int(info.get("ppid") or 0),
        name=info.get("name") or "",
        username=info.get("username") or "",
        status=info.get("status") or "unknown",
        cpu_percent=round(float(info.get("cpu_percent") or 0.0), 1),
        memory_rss=rss,
        memory_percent=round(float(info.get("memory_percent") or 0.0), 2),
        create_time=float(info.get("create_time") or 0.0),
        cmdline=_join_cmd(cmdline),
        exe=info.get("exe"),
        threads=info.get("num_threads"),
        gpu_memory=gpu_memory,
        gpu_index=gpu_index,
    )


def _detail(pid: int, gpu: Mapping[int, tuple[int, int | None]]) -> ProcessInfo | None:
    """读取单个进程的完整字段；进程已退出或无权限时返回 ``None``。"""
    try:
        info = psutil.Process(pid).as_dict(attrs=_DETAIL_ATTRS, ad_value=None)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None
    return _build(info, gpu)


def _sort(items: list[ProcessInfo], key: SortKey) -> list[ProcessInfo]:
    match key:
        case "memory" | "mem" | "rss":
            return sorted(items, key=lambda item: item.memory_rss, reverse=True)
        case "pid":
            return sorted(items, key=lambda item: item.pid)
        case "name":
            return sorted(items, key=lambda item: item.name.lower())
        case "age":
            return sorted(items, key=lambda item: item.create_time)
        case _:
            # 首次调用时 process_iter 的 cpu_percent 全是 0，退回按内存排序更有意义
            if all(item.cpu_percent == 0 for item in items):
                return sorted(items, key=lambda item: item.memory_rss, reverse=True)
            return sorted(items, key=lambda item: item.cpu_percent, reverse=True)


def processes(
    *,
    limit: int | None = None,
    sort: SortKey = "cpu",
    name: str | None = None,
    include_gpu: bool = True,
) -> list[ProcessInfo]:
    """列举进程；``name`` 做不区分大小写的子串过滤。

    ``sort`` 取值：``cpu``（默认）/ ``memory`` / ``pid`` / ``name`` / ``age``。

    性能：Windows 上每个进程的 ``OpenProcess`` 约 5~10ms，全量列举通常 2~3 秒；
    带 ``name`` 时先只读 ``pid/name`` 做筛选（毫秒级），再取命中进程的详情。
    列表不含 ``exe`` 与线程数（各多耗 0.4s / 2s），需要时用 :func:`process`。
    """
    gpu = gpu_processes() if include_gpu else {}
    if name:
        needle = name.lower()
        pids = [
            int(proc.info["pid"])
            for proc in psutil.process_iter(("pid", "name"))
            if needle in (proc.info.get("name") or "").lower()
        ]
        items = [
            item for item in (_detail(pid, gpu) for pid in pids) if item is not None
        ]
    else:
        items = [_build(proc.info, gpu) for proc in psutil.process_iter(_LIST_ATTRS)]
    items = _sort(items, sort)
    return items[:limit] if limit is not None and limit > 0 else items


def top(count: int = 5, *, sort: SortKey = "cpu") -> list[ProcessInfo]:
    """资源占用前 ``count`` 名。"""
    return processes(limit=count, sort=sort)


def process(pid: int, *, interval: float | None = 0.1) -> ProcessInfo:
    """单个进程的详情；``interval`` 决定 CPU 占用的采样窗口。"""
    try:
        proc = psutil.Process(pid)
        cpu = proc.cpu_percent(interval=interval)
        info = proc.as_dict(attrs=_DETAIL_ATTRS, ad_value=None)
    except psutil.NoSuchProcess as exc:
        raise ProcessNotFoundError(f"进程 {pid} 不存在或已退出") from exc
    except psutil.AccessDenied as exc:
        raise ProcessError(f"无权读取进程 {pid}：{exc}") from exc
    info["cpu_percent"] = cpu
    return _build(info, gpu_processes())


def kill(pid: int, *, force: bool = False, timeout: float = 3.0) -> bool:
    """结束进程；``force=True`` 直接 SIGKILL，否则先 SIGTERM 再兜底。

    拒绝结束 ``pid<=0``、当前进程与父进程，避免脚本把自己或宿主杀掉。
    """
    if pid <= 0:
        raise ProcessError(f"非法的 pid：{pid}")
    if pid == os.getpid():
        raise ProcessError("拒绝对当前进程执行 kill")
    if pid == os.getppid():
        raise ProcessError("拒绝对父进程执行 kill")

    try:
        proc = psutil.Process(pid)
        if force:
            proc.kill()
        else:
            proc.terminate()
        proc.wait(timeout)
    except psutil.NoSuchProcess:
        return True  # 已经退出了，视为成功
    except psutil.AccessDenied as exc:
        raise ProcessError(f"无权结束进程 {pid}（可能需要管理员权限）") from exc
    except psutil.TimeoutExpired:
        if force:
            raise ProcessError(f"进程 {pid} 在 {timeout}s 内未能结束") from None
        return kill(pid, force=True, timeout=timeout)
    return True
