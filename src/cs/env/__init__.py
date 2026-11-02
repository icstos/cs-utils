"""操作系统级环境信息与操作：平台、CPU、内存、磁盘、GPU、网络、进程、命令执行。

三种调用方式：

- 门面类 :class:`Env`（全部静态方法）::

    from cs.env import Env
    Env.platform().distro
    Env.snapshot(top=5).report()
    Env.run(["git", "status", "--short"]).text

- 细粒度函数：:func:`platform_info` / :func:`cpu_info` / :func:`memory_info` /
  :func:`disks` / :func:`gpu_runtime` / :func:`processes` / :func:`run` …

- MCP：``python -m cs.env serve``，工具见 :mod:`cs.env.mcp_server`。

安全约定：命令执行永不经过 shell（argv 列表直传），字符串形式出现
``| & ; < > $ ` `` 会被拒绝；进程结束拒绝 ``pid<=0``、当前进程与父进程；
环境变量读取默认把密码/令牌类字段脱敏。
"""

from __future__ import annotations

from .core import Env
from .environ import drop, get, is_sensitive, mask, patched, put, read, scoped
from .errors import (
    CommandError,
    CommandNotFoundError,
    CommandTimeout,
    EnvError,
    ProcessError,
    ProcessNotFoundError,
    UnsafeCommandError,
)
from .models import (
    AVX,
    CommandResult,
    CPUInfo,
    DiskInfo,
    GPUDevice,
    GPURuntime,
    MemoryInfo,
    NetworkInfo,
    PlatformInfo,
    ProcessInfo,
    Snapshot,
    Throughput,
    human_bytes,
    jsonable,
)
from .process import current, kill, process, processes, top
from .shell import Command, execute, open_url, run, run_all, split, which
from .system import (
    cpu_avx,
    cpu_info,
    cpu_model,
    disk_io,
    disk_usage,
    disks,
    distro,
    gpu_runtime,
    memory_info,
    network_info,
    platform_info,
    platform_kind,
    snapshot,
)

__all__ = [
    "AVX",
    "CPUInfo",
    "Command",
    "CommandError",
    "CommandNotFoundError",
    "CommandResult",
    "CommandTimeout",
    "DiskInfo",
    "Env",
    "EnvError",
    "GPUDevice",
    "GPURuntime",
    "MemoryInfo",
    "NetworkInfo",
    "PlatformInfo",
    "ProcessError",
    "ProcessInfo",
    "ProcessNotFoundError",
    "Snapshot",
    "Throughput",
    "UnsafeCommandError",
    "cpu_avx",
    "cpu_info",
    "cpu_model",
    "current",
    "disk_io",
    "disk_usage",
    "disks",
    "distro",
    "drop",
    "execute",
    "get",
    "gpu_runtime",
    "human_bytes",
    "is_sensitive",
    "jsonable",
    "kill",
    "mask",
    "memory_info",
    "network_info",
    "open_url",
    "patched",
    "platform_info",
    "platform_kind",
    "process",
    "processes",
    "put",
    "read",
    "run",
    "run_all",
    "scoped",
    "snapshot",
    "split",
    "top",
    "which",
]
