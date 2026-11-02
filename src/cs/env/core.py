"""``Env`` 门面：把系统、进程、命令、环境变量四组能力收进一个类。

全部是静态方法（无状态、可并发），既方便脚本调用，也方便 MCP 层薄封装：

>>> Env.platform().distro
'Windows 11（10.0.26100）'
>>> Env.memory().total_gb
31.82
>>> Env.run(["git", "status", "--short"]).ok
True
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from . import environ, system
from .models import (
    CommandResult,
    CPUInfo,
    DiskInfo,
    GPURuntime,
    MemoryInfo,
    NetworkInfo,
    PlatformInfo,
    ProcessInfo,
    Snapshot,
    Throughput,
)
from .process import kill as _kill
from .process import process as _process
from .process import processes as _processes
from .shell import Command, open_url, run, run_all, which

__all__ = ["Env"]

type SortKey = str


class Env:
    """操作系统级信息的统一入口。"""

    __slots__ = ()

    # ---------------------------------------------------------- 平台

    @staticmethod
    def platform() -> PlatformInfo:
        """操作系统 / 发行版 / Python 运行时。"""
        return system.platform_info()

    @staticmethod
    def python_path() -> Path:
        """当前解释器路径。"""
        return Path(system.platform_info().python_executable)

    @staticmethod
    def pid() -> int:
        """当前进程号。"""
        return os.getpid()

    @staticmethod
    def admin() -> bool | None:
        """是否具备管理员 / root 权限（未知为 ``None``）。"""
        return system.is_elevated()

    # ---------------------------------------------------------- 硬件

    @staticmethod
    def cpu(*, interval: float | None = 0.1, per_core: bool = False) -> CPUInfo:
        """CPU 型号、核心数与占用率。"""
        return system.cpu_info(interval=interval, per_core=per_core)

    @staticmethod
    def memory() -> MemoryInfo:
        """物理内存与交换分区。"""
        return system.memory_info()

    @staticmethod
    def disks(*, all: bool = False) -> tuple[DiskInfo, ...]:
        """所有挂载点。"""
        return system.disks(all=all)

    @staticmethod
    def disk_usage(path: str | Path = ".") -> DiskInfo:
        """某个路径所在分区的占用。"""
        return system.disk_usage(path)

    @staticmethod
    def disk_io(interval: float = 1.0) -> Throughput:
        """磁盘读写速率（阻塞 ``interval`` 秒）。"""
        return system.disk_io(interval)

    @staticmethod
    def network() -> NetworkInfo:
        """累计网络收发量。"""
        return system.network_info()

    @staticmethod
    def gpus() -> GPURuntime:
        """GPU 驱动、CUDA 版本与逐卡状态。"""
        return system.gpu_runtime()

    @staticmethod
    def snapshot(
        *, top: int = 5, interval: float | None = 0.1, include_gpu: bool = True
    ) -> Snapshot:
        """整机快照（平台 + 硬件 + 网络 + Top 进程）。"""
        return system.snapshot(top=top, interval=interval, with_gpu=include_gpu)

    # ---------------------------------------------------------- 进程

    @staticmethod
    def processes(
        *,
        limit: int | None = None,
        sort: SortKey = "cpu",
        name: str | None = None,
        include_gpu: bool = True,
    ) -> list[ProcessInfo]:
        """列举进程，可按 ``name`` 过滤、按 ``cpu`` / ``memory`` 排序。"""
        return _processes(limit=limit, sort=sort, name=name, include_gpu=include_gpu)

    @staticmethod
    def process(pid: int, *, interval: float | None = 0.1) -> ProcessInfo:
        """单个进程详情。"""
        return _process(pid, interval=interval)

    @staticmethod
    def kill(pid: int, *, force: bool = False, timeout: float = 3.0) -> bool:
        """结束进程（拒绝当前进程与父进程）。"""
        return _kill(pid, force=force, timeout=timeout)

    # ---------------------------------------------------------- 命令

    @staticmethod
    def run(command: Command | str | Sequence[str], **overrides: Any) -> CommandResult:
        """执行命令（不经 shell，参数按列表传递）。"""
        return run(command, **overrides)

    @staticmethod
    def run_all(
        *commands: Command | str | Sequence[str], stop_on_error: bool = True
    ) -> tuple[CommandResult, ...]:
        """顺序执行多条命令，默认首条失败即停。"""
        return run_all(*commands, stop_on_error=stop_on_error)

    @staticmethod
    def which(name: str) -> Path | None:
        """在 ``PATH`` 中查找可执行文件。"""
        return which(name)

    @staticmethod
    def open_url(url: str) -> bool:
        """用默认浏览器打开链接。"""
        return open_url(url)

    # ---------------------------------------------------------- 环境变量

    @staticmethod
    def getenv(name: str, default: str | None = None) -> str | None:
        """读取环境变量。"""
        return environ.get(name, default)

    @staticmethod
    def environ(prefix: str | None = None, *, reveal: bool = False) -> dict[str, str]:
        """读取全部变量，敏感项默认脱敏。"""
        return environ.read(prefix, reveal=reveal)

    @staticmethod
    def setenv(name: str, value: str) -> None:
        """设置变量（仅当前进程及其子进程）。"""
        environ.put(name, value)

    @staticmethod
    def unsetenv(name: str) -> bool:
        """删除变量，返回是否曾经存在。"""
        return environ.drop(name)

    @staticmethod
    def patched(
        values: Mapping[str, str | None],
    ) -> AbstractContextManager[None]:
        """上下文管理器：临时覆盖变量并在退出时还原。"""
        return environ.patched(values)
