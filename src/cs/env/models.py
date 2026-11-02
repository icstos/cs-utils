"""``cs.env`` 的数据模型。

全部是 ``frozen`` + ``slots`` 的 dataclass：创建后不可变、字段紧凑，
``to_dict()`` / ``to_json()`` 可直接交给 MCP 工具或日志。

>>> Env.memory().to_dict()["total_gb"] > 0
True
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

from ..my_types import PlatformType

__all__ = [
    "AVX",
    "CPUInfo",
    "CommandResult",
    "DiskInfo",
    "GPUDevice",
    "GPURuntime",
    "MemoryInfo",
    "Model",
    "NetworkInfo",
    "PlatformInfo",
    "ProcessInfo",
    "Snapshot",
    "Throughput",
    "human_bytes",
    "jsonable",
]

KB = 1024
MB = KB * 1024
GB = MB * 1024

type Json = dict[str, Any] | list[Any] | str | int | float | bool | None


def human_bytes(size: float) -> str:
    """把字节数格式化成 ``1.2 GB`` 这样的可读单位。"""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < KB:
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= KB
    return f"{value:.1f} PB"


def jsonable(value: Any) -> Json:
    """递归转成 JSON 可序列化结构（枚举取 value、Path 取字符串）。"""
    match value:
        case None | bool() | int() | float() | str():
            return value
        case Enum():
            return value.value
        case Path():
            return str(value)
        case bytes():
            return value.decode("utf-8", "replace")
        case Mapping():
            return {str(key): jsonable(item) for key, item in value.items()}
        case Sequence():
            return [jsonable(item) for item in value]
        case _:
            pass
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: jsonable(getattr(value, item.name)) for item in fields(value)
        }
    return str(value)


class Model:
    """记录类型的公共行为：字典 / JSON。"""

    __slots__ = ()

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)  # type: ignore[return-value]

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class AVX(StrEnum):
    """CPU 指令集等级，用于挑选预编译的推理后端。"""

    BASIC = "basic"
    AVX = "avx"
    AVX2 = "avx2"
    AVX512 = "avx512"

    @classmethod
    def parse(cls, value: str | None) -> AVX:
        """宽松解析：``AVX512F`` / ``avx_512`` / ``AVX2`` 都能认。"""
        text = (value or "").strip().lower().replace("_", "").replace("-", "")
        match text:
            case "avx512" | "avx512f" | "avx512bw":
                return cls.AVX512
            case "avx2":
                return cls.AVX2
            case "avx":
                return cls.AVX
            case _:
                return cls.BASIC


# ---------------------------------------------------------------- 平台


@dataclass(frozen=True, slots=True)
class PlatformInfo(Model):
    """操作系统 + Python 运行时信息。"""

    system: str
    kind: PlatformType
    release: str
    version: str
    machine: str
    processor: str
    distro: str
    hostname: str
    python_version: str
    python_executable: str
    python_implementation: str
    is_64bit: bool
    elevated: bool | None = None

    def __str__(self) -> str:
        admin = {True: "是", False: "否", None: "未知"}[self.elevated]
        return (
            f"{self.distro}（{self.kind.value}）\n"
            f"- 内核/版本：{self.release} / {self.version}\n"
            f"- 架构：{self.machine}（64 位：{'是' if self.is_64bit else '否'}）\n"
            f"- 主机名：{self.hostname}；管理员权限：{admin}\n"
            f"- Python：{self.python_version} [{self.python_implementation}]"
            f" -> {self.python_executable}"
        )


# ---------------------------------------------------------------- CPU


@dataclass(frozen=True, slots=True)
class CPUInfo(Model):
    """CPU 型号、核心数与实时占用。"""

    model: str
    vendor: str
    arch: str
    physical_cores: int
    logical_cores: int
    percent: float
    per_core: tuple[float, ...] = ()
    freq_mhz: float | None = None
    freq_max_mhz: float | None = None
    avx: AVX = AVX.BASIC
    load_avg: tuple[float, float, float] | None = None

    def __str__(self) -> str:
        freq = f"{self.freq_mhz:.0f} MHz" if self.freq_mhz else "未知"
        return (
            f"{self.model}（{self.vendor} / {self.arch}）\n"
            f"- 核心：{self.physical_cores} 物理 / {self.logical_cores} 逻辑\n"
            f"- 占用：{self.percent:.1f}%；频率：{freq}；指令集：{self.avx.value.upper()}"
        )


# ---------------------------------------------------------------- 内存


@dataclass(frozen=True, slots=True)
class MemoryInfo(Model):
    """物理内存与交换分区（字节）。"""

    total: int
    available: int
    used: int
    free: int
    percent: float
    cached: int | None = None
    swap_total: int = 0
    swap_used: int = 0
    swap_percent: float = 0.0

    @property
    def total_gb(self) -> float:
        return round(self.total / GB, 2)

    @property
    def used_gb(self) -> float:
        return round(self.used / GB, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "total_gb": self.total_gb,
            "used_gb": self.used_gb,
            "available_gb": round(self.available / GB, 2),
        }

    def __str__(self) -> str:
        return (
            f"内存 {human_bytes(self.used)} / {human_bytes(self.total)}"
            f"（{self.percent:.1f}%），可用 {human_bytes(self.available)}\n"
            f"- 交换分区 {human_bytes(self.swap_used)} / "
            f"{human_bytes(self.swap_total)}（{self.swap_percent:.1f}%）"
        )


# ---------------------------------------------------------------- 磁盘


@dataclass(frozen=True, slots=True)
class DiskInfo(Model):
    """单个挂载点的容量与占用。"""

    device: str
    mountpoint: str
    fstype: str
    total: int
    used: int
    free: int
    percent: float
    read_only: bool = False

    def __str__(self) -> str:
        mode = "只读" if self.read_only else "读写"
        return (
            f"{self.mountpoint}（{self.device} / {self.fstype or '?'} / {mode}）"
            f" {human_bytes(self.used)} / {human_bytes(self.total)}"
            f"（{self.percent:.1f}%），剩余 {human_bytes(self.free)}"
        )


@dataclass(frozen=True, slots=True)
class Throughput(Model):
    """磁盘或网络的累计量与采样速率（字节 / 字节每秒）。"""

    read_bytes: int = 0
    write_bytes: int = 0
    read_rate: float = 0.0
    write_rate: float = 0.0
    seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f"读 {human_bytes(self.read_bytes)}"
            f"（{human_bytes(self.read_rate)}/s），"
            f"写 {human_bytes(self.write_bytes)}"
            f"（{human_bytes(self.write_rate)}/s），采样 {self.seconds:.2f}s"
        )


# ---------------------------------------------------------------- 网络


@dataclass(frozen=True, slots=True)
class NetworkInfo(Model):
    """网络累计收发量与在线网卡。"""

    bytes_sent: int = 0
    bytes_recv: int = 0
    packets_sent: int = 0
    packets_recv: int = 0
    interfaces: tuple[str, ...] = ()

    def __str__(self) -> str:
        return (
            f"发送 {human_bytes(self.bytes_sent)} / 接收 "
            f"{human_bytes(self.bytes_recv)}；在线网卡："
            f"{', '.join(self.interfaces) or '无'}"
        )


# ---------------------------------------------------------------- GPU


@dataclass(frozen=True, slots=True)
class GPUDevice(Model):
    """单块显卡的运行状态（字节 / 百分比 / 摄氏度 / 瓦）。"""

    index: int
    name: str
    total_memory: int | None = None
    used_memory: int | None = None
    free_memory: int | None = None
    utilization: float | None = None
    temperature: float | None = None
    power: float | None = None

    @property
    def memory_percent(self) -> float | None:
        if not self.total_memory or self.used_memory is None:
            return None
        return round(self.used_memory / self.total_memory * 100, 2)

    def __str__(self) -> str:
        memory = (
            f"{human_bytes(self.used_memory)} / {human_bytes(self.total_memory)}"
            if self.total_memory
            else "显存未知"
        )
        parts = [f"GPU{self.index} {self.name}：{memory}"]
        if self.utilization is not None:
            parts.append(f"利用率 {self.utilization:.0f}%")
        if self.temperature is not None:
            parts.append(f"{self.temperature:.0f}°C")
        if self.power:
            parts.append(f"{self.power:.0f}W")
        return "，".join(parts)


@dataclass(frozen=True, slots=True)
class GPURuntime(Model):
    """GPU 运行环境：驱动、CUDA 版本与逐卡状态。"""

    available: bool = False
    driver: str | None = None
    cuda: str | None = None
    devices: tuple[GPUDevice, ...] = ()

    @property
    def count(self) -> int:
        return len(self.devices)

    def __str__(self) -> str:
        if not self.available:
            return "未检测到可用 GPU（或未安装 NVML）"
        head = (
            f"{self.count} 块 GPU；驱动 {self.driver or '?'}，CUDA {self.cuda or '?'}"
        )
        return "\n".join([head, *(f"- {device}" for device in self.devices)])


# ---------------------------------------------------------------- 进程


@dataclass(frozen=True, slots=True)
class ProcessInfo(Model):
    """单个进程的快照。"""

    pid: int
    ppid: int
    name: str
    username: str
    status: str
    cpu_percent: float
    memory_rss: int
    memory_percent: float
    create_time: float
    cmdline: str
    exe: str | None = None
    threads: int | None = None
    gpu_memory: int | None = None
    gpu_index: int | None = None

    @property
    def started(self) -> str:
        """本地时区的启动时间。"""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.create_time))

    @property
    def age(self) -> float:
        """已运行秒数。"""
        return round(time.time() - self.create_time, 1)

    def to_dict(self) -> dict[str, Any]:
        return {"started": self.started, "age": self.age, **super().to_dict()}

    def __str__(self) -> str:
        gpu = (
            f"，显存 {human_bytes(self.gpu_memory)}（GPU{self.gpu_index}）"
            if self.gpu_memory
            else ""
        )
        return (
            f"[{self.pid}] {self.name}（{self.username}）{self.status}\n"
            f"- CPU {self.cpu_percent:.1f}%，内存 {human_bytes(self.memory_rss)}"
            f"（{self.memory_percent:.1f}%）{gpu}\n"
            f"- 启动于 {self.started}；命令：{self.cmdline or self.exe or '?'}"
        )


# ---------------------------------------------------------------- 命令


@dataclass(frozen=True, slots=True)
class CommandResult(Model):
    """一条命令的执行结果（非零退出不抛异常，看 ``ok``）。"""

    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    timed_out: bool = False
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def text(self) -> str:
        """去掉首尾空白的 stdout。"""
        return self.stdout.strip()

    @property
    def output(self) -> str:
        """stdout，为空时退回 stderr。"""
        return self.stdout.strip() or self.stderr.strip()

    def __str__(self) -> str:
        state = "超时" if self.timed_out else f"退出码 {self.returncode}"
        return (
            f"$ {' '.join(self.argv)}\n[{state}，{self.duration:.2f}s]\n{self.output}"
        )


# ---------------------------------------------------------------- 快照


@dataclass(frozen=True, slots=True)
class Snapshot(Model):
    """整机状态快照：平台、CPU、内存、磁盘、GPU、网络、Top 进程。"""

    timestamp: float
    platform: PlatformInfo
    cpu: CPUInfo
    memory: MemoryInfo
    disks: tuple[DiskInfo, ...] = ()
    gpu: GPURuntime = GPURuntime()
    network: NetworkInfo = NetworkInfo()
    processes: tuple[ProcessInfo, ...] = ()

    @property
    def created(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))

    def to_dict(self) -> dict[str, Any]:
        return {"created": self.created, **super().to_dict()}

    def report(self, *, top: int = 5) -> str:
        """Markdown 风格的整机报告，可直接丢给模型或终端。"""
        lines = [
            f"# 系统快照 {self.created}",
            (
                f"- {self.platform.distro} / {self.platform.machine}"
                f" / Python {self.platform.python_version}"
            ),
            f"- {self.cpu}",
            f"- {self.memory}",
        ]
        lines.extend(f"- 磁盘 {disk}" for disk in self.disks)
        if self.gpu.available:
            lines.extend(f"- {device}" for device in self.gpu.devices)
        else:
            lines.append("- 未检测到 GPU（或已跳过采集）")
        lines.append(f"- 网络 {self.network}")
        if self.processes:
            lines.append(f"## 资源占用 Top {min(top, len(self.processes))}")
            lines.extend(
                f"- [{item.pid}] {item.name}：CPU {item.cpu_percent:.1f}%，"
                f"内存 {human_bytes(item.memory_rss)}"
                for item in self.processes
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.report(top=len(self.processes))
