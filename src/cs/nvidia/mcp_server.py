"""把 :class:`cs.nvidia.NVIDIA` 暴露为 MCP 工具（FastMCP ≥ 4）。

启动方式::

    python -m cs.nvidia serve                 # stdio，供 MCP 客户端拉起
    python -m cs.nvidia serve --transport http --port 8000
    fastmcp run src/cs/nvidia/mcp_server.py:mcp

约定
----
* 工具名统一 ``nvidia_*``，全部只读（唯一写操作是导出 CSV）；
* 成功返回 ``{"ok": true, ...}``，业务异常统一转成 ``ToolError``，
  不把 traceback 抛给模型；
* 读不到的指标返回 ``null``（不是 0），模型据此区分"空闲"和"不支持"。
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from .core import NVIDIA, GPUHandle
from .errors import GPUError
from .models import HealthReport
from .monitor import render_table

__all__ = ["create_server", "mcp"]

_SERVER_NAME = "cs-nvidia"
_MAX_SAMPLES = 600
_INSTRUCTIONS = """\
本服务器只读地查询本机 NVIDIA 显卡：型号/静态信息、实时状态（占用率、显存、温度、
功耗、风扇、频率、P-State）、占用进程、降频原因、多卡拓扑与健康体检。

使用建议：
1. 先调用 nvidia_list 或 nvidia_driver 确认有几块卡、型号是什么；
2. 问"现在状态如何"用 nvidia_status；问"谁在占显存"用 nvidia_processes；
3. 要看一段时间的波动用 nvidia_watch（有次数上限，别无限采样）；
4. gpu 参数接受序号 "0"、UUID "GPU-xxxx"、PCI 地址 "0000:01:00.0" 或 "all"；
5. 指标为 null 表示驱动/平台不提供（消费级卡的风扇、ECC 常见），不代表 0。
"""


def _guard[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """把领域异常转换为 MCP 可读错误。"""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except (GPUError, OSError, ValueError) as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


def _done(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def _select(gpu: str | None) -> list[GPUHandle]:
    """``"all"`` / ``""`` / ``None`` 表示全部显卡。"""
    if gpu is None or gpu.strip().lower() in {"", "all", "*"}:
        return NVIDIA.gpus()
    return [NVIDIA.gpu(gpu.strip())]


def _health_payload(report: HealthReport) -> dict[str, Any]:
    return report.to_dict()


def create_server(
    name: str = _SERVER_NAME, *, instructions: str = _INSTRUCTIONS
) -> FastMCP:
    """构造一个独立的服务器实例（便于测试或多实例部署）。"""
    server: FastMCP = FastMCP(name, instructions=instructions)

    # ---------------------------------------------------------- 只读
    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def nvidia_driver() -> dict[str, Any]:
        """查看驱动版本、CUDA 版本、NVML 版本与 GPU 数量。"""
        return _done(**NVIDIA.driver().to_dict())

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def nvidia_list() -> dict[str, Any]:
        """列出所有显卡的型号、UUID、架构、显存总量、VBIOS、PCI 地址等静态信息。"""
        gpus = [info.to_dict() for info in NVIDIA.list()]
        return _done(count=len(gpus), gpus=gpus)

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def nvidia_info(gpu: str = "0") -> dict[str, Any]:
        """查看单块显卡的详细静态信息（型号、架构、显存、核心数、ECC/MIG/计算模式等）。"""
        handles = _select(gpu)
        if len(handles) != 1:
            raise ValueError(
                "nvidia_info 只接受单块显卡，请传具体序号 / UUID / PCI 地址"
            )
        return _done(**handles[0].info().to_dict())

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def nvidia_status(gpu: str = "all") -> dict[str, Any]:
        """查看实时运行状态：占用率、显存、温度、功耗、风扇、频率、P-State、PCIe 链路。"""
        states = [handle.status() for handle in _select(gpu)]
        return _done(count=len(states), gpus=[state.to_dict() for state in states])

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def nvidia_processes(gpu: str = "all") -> dict[str, Any]:
        """查看占用显卡的进程（pid、进程名、显存占用、占用方式 compute/graphics/MPS）。

        used_memory 为 null 表示驱动未提供（Windows WDDM 常见），不代表 0。
        """
        processes = [
            process for handle in _select(gpu) for process in handle.processes()
        ]
        return _done(
            count=len(processes),
            processes=[process.to_dict() for process in processes],
        )

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def nvidia_throttle(gpu: str = "all") -> dict[str, Any]:
        """查看当前时钟降频原因（温度/功耗/可靠性等）及该卡支持的全部原因位。"""
        payload = [
            {
                "index": handle.index,
                "uuid": handle.uuid,
                "name": handle.name,
                **(handle.throttle().to_dict()),
            }
            for handle in _select(gpu)
        ]
        return _done(count=len(payload), gpus=payload)

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def nvidia_topology() -> dict[str, Any]:
        """查看多卡互联拓扑矩阵（SINGLE/MULTIPLE/HOST_BRIDGE/NODE/SYSTEM）。"""
        matrix = NVIDIA.topologies()
        return _done(**matrix.to_dict())

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def nvidia_health(gpu: str = "all") -> dict[str, Any]:
        """健康体检：温度是否接近降频阈值、显存是否吃紧、是否被硬件限速。

        level 为 info / warning / critical，issues 里给出具体原因。
        """
        reports = [_health_payload(handle.health()) for handle in _select(gpu)]
        critical = [item for item in reports if not item["healthy"]]
        return _done(
            count=len(reports),
            healthy=not critical,
            gpus=reports,
        )

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def nvidia_watch(
        gpu: str = "0",
        interval: float = 1.0,
        count: int = 5,
        duration: float | None = None,
    ) -> dict[str, Any]:
        """采样一段时间并返回统计（min/max/avg）与逐次快照。

        用于回答"刚才有没有跑满""温度会不会飙升"这类问题。
        count 最多 600 次，duration 最多 600 秒，interval 最小 0.1 秒。
        """
        if not 0.1 <= interval <= 60:
            raise ValueError("interval 必须在 0.1~60 秒之间")
        if not 1 <= count <= _MAX_SAMPLES:
            raise ValueError(f"count 必须在 1~{_MAX_SAMPLES} 之间")
        if duration is not None and not 0 < duration <= 600:
            raise ValueError("duration 必须在 0~600 秒之间")
        series = [
            handle.series(interval=interval, count=count) for handle in _select(gpu)
        ]
        return _done(
            count=len(series),
            series=[item.to_dict(include_states=True) for item in series],
        )

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def nvidia_text(gpu: str = "all") -> dict[str, Any]:
        """以 nvidia-smi 风格的等宽文本表格输出，适合直接贴给用户看。"""
        return _done(table=render_table([h.status() for h in _select(gpu)]))

    # ---------------------------------------------------------- 写操作
    @server.tool(annotations={"readOnlyHint": False})
    @_guard
    def nvidia_export_csv(
        path: str,
        gpu: str = "0",
        interval: float = 1.0,
        count: int = 10,
    ) -> dict[str, Any]:
        """采样并导出为 CSV（温度、功耗、占用率、显存、频率的时间序列）。

        path 为输出文件（相对路径按服务进程的工作目录解析），目录会自动创建。
        """
        if not 0.1 <= interval <= 60:
            raise ValueError("interval 必须在 0.1~60 秒之间")
        if not 1 <= count <= _MAX_SAMPLES:
            raise ValueError(f"count 必须在 1~{_MAX_SAMPLES} 之间")
        handles = _select(gpu)
        files: list[str] = []
        for handle in handles:
            target = path
            if len(handles) > 1:  # 多卡时按序号分文件，避免互相覆盖
                stem, _, suffix = path.rpartition(".")
                target = f"{stem or path}_gpu{handle.index}.{suffix or 'csv'}"
            files.append(
                str(handle.series(interval=interval, count=count).to_csv(target))
            )
        return _done(files=files, count=count, interval=interval)

    return server


#: 模块级实例，供 ``fastmcp run src/cs/nvidia/mcp_server.py:mcp`` 直接引用
mcp = create_server()
