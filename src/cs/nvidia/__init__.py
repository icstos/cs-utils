"""``cs.nvidia`` —— NVIDIA 显卡信息查询 + FastMCP 服务。

对外只暴露一个门面 :class:`NVIDIA`（以及句柄 :class:`GPUHandle`）::

    from cs.nvidia import NVIDIA

    NVIDIA.driver()            # 驱动 / CUDA 版本
    NVIDIA.list()              # 型号、显存、架构
    NVIDIA.status(0)           # 运行状态快照
    NVIDIA.health(0)           # 温度 / 显存 / 降频体检

    with NVIDIA.open(0) as gpu:
        gpu.processes()
        gpu.sample(interval=1, count=5).to_csv("gpu.csv")

MCP 能力按需加载，未安装 ``fastmcp`` 时类方法照常可用::

    from cs.nvidia import create_server
    create_server().run()

依赖 ``nvidia-ml-py``（提供 ``pynvml`` 命名空间），缺失时抛出
:class:`MissingDependencyError` 并给出安装命令；非 NVIDIA 机器上
`NVIDIA.available()` 返回 ``False`` 而不会报错。
"""

from __future__ import annotations

from typing import Any

from .core import NVIDIA, GPUHandle, Selector
from .errors import (
    GPUError,
    GPULostError,
    GPUNotFoundError,
    GPUNotSupportedError,
    GPUUnavailableError,
    MissingDependencyError,
)
from .models import (
    Architecture,
    Brand,
    ClockReading,
    Clocks,
    ComputeMode,
    DriverInfo,
    GPUInfo,
    GPUProcess,
    GPUStatus,
    HealthIssue,
    HealthReport,
    MemoryUsage,
    PciLink,
    PowerInfo,
    ProcessKind,
    PState,
    Severity,
    TemperatureLimits,
    ThrottleInfo,
    ThrottleReason,
    TopologyLevel,
    TopologyMatrix,
    Utilization,
    bytes_to,
    cuda_version_str,
)
from .monitor import SampleSeries, iter_status, render_table, sample_status

__all__ = [
    "NVIDIA",
    "Architecture",
    "Brand",
    "ClockReading",
    "Clocks",
    "ComputeMode",
    "DriverInfo",
    "GPUError",
    "GPUHandle",
    "GPUInfo",
    "GPULostError",
    "GPUNotFoundError",
    "GPUNotSupportedError",
    "GPUProcess",
    "GPUStatus",
    "GPUUnavailableError",
    "HealthIssue",
    "HealthReport",
    "MemoryUsage",
    "MissingDependencyError",
    "PState",
    "PciLink",
    "PowerInfo",
    "ProcessKind",
    "SampleSeries",
    "Selector",
    "Severity",
    "TemperatureLimits",
    "ThrottleInfo",
    "ThrottleReason",
    "TopologyLevel",
    "TopologyMatrix",
    "Utilization",
    "bytes_to",
    "create_server",
    "cuda_version_str",
    "iter_status",
    "render_table",
    "sample_status",
]


def create_server(*args: Any, **kwargs: Any) -> Any:
    """创建 FastMCP 服务器实例（惰性导入 ``fastmcp``）。"""
    from .mcp_server import create_server as _create_server

    return _create_server(*args, **kwargs)


def __getattr__(name: str) -> Any:
    """``cs.nvidia.mcp`` 惰性暴露，避免可选依赖在 import 期被拉起。"""
    if name == "mcp":
        from .mcp_server import mcp as server

        return server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
