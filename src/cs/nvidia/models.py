"""``cs.nvidia`` 的数据模型：纯值对象，只做承载，不含 I/O 副作用。

约定
----
* 所有 dataclass 都是 ``frozen=True, slots=True``，可安全跨线程传递；
* 单位统一为 **字节 / 瓦特 / 摄氏度 / 百分比 / MHz**，需要其它单位请用 ``to_dict()``；
* 驱动未提供（或平台不支持）的指标一律为 ``None``，**不用 0 冒充**，
  这样调用方才能区分"真的空闲"与"读不到"。
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag, StrEnum
from typing import Any

__all__ = [
    "Architecture",
    "Brand",
    "ClockReading",
    "Clocks",
    "ComputeMode",
    "DriverInfo",
    "GPUInfo",
    "GPUProcess",
    "GPUStatus",
    "HealthIssue",
    "HealthReport",
    "MemoryUsage",
    "PState",
    "PciLink",
    "PowerInfo",
    "ProcessKind",
    "Severity",
    "TemperatureLimits",
    "ThrottleInfo",
    "ThrottleReason",
    "TopologyLevel",
    "TopologyMatrix",
    "Utilization",
    "bytes_to",
    "cuda_version_str",
]


# ---------------------------------------------------------------- 枚举
class Brand(StrEnum):
    """产品线（对应 ``NVML_BRAND_*``）。"""

    UNKNOWN = "unknown"
    QUADRO = "quadro"
    TESLA = "tesla"
    NVS = "nvs"
    GRID = "grid"
    GEFORCE = "geforce"
    TITAN = "titan"
    VAPPS = "vapps"
    VPC = "vpc"
    VCS = "vcs"
    VWS = "vws"
    CLOUD_GAMING = "cloud-gaming"
    QUADRO_RTX = "quadro-rtx"
    RTX = "rtx"
    NVIDIA = "nvidia"
    GEFORCE_RTX = "geforce-rtx"
    TITAN_RTX = "titan-rtx"
    DLA = "dla"
    VGAMEDEV = "vgamedev"
    NPU = "npu"


class Architecture(StrEnum):
    """GPU 架构（对应 ``NVML_DEVICE_ARCH_*``）。"""

    UNKNOWN = "unknown"
    KEPLER = "kepler"
    MAXWELL = "maxwell"
    PASCAL = "pascal"
    VOLTA = "volta"
    TURING = "turing"
    AMPERE = "ampere"
    ADA = "ada"
    HOPPER = "hopper"
    BLACKWELL = "blackwell"
    RUBIN = "rubin"
    DLA = "dla"
    DLA2 = "dla2"
    NPU3 = "npu3"


class ComputeMode(StrEnum):
    DEFAULT = "default"
    EXCLUSIVE_THREAD = "exclusive-thread"
    PROHIBITED = "prohibited"
    EXCLUSIVE_PROCESS = "exclusive-process"


class ProcessKind(StrEnum):
    """进程占用显卡的方式。"""

    COMPUTE = "compute"
    GRAPHICS = "graphics"
    MPS = "mps"


class TopologyLevel(StrEnum):
    """两块 GPU 之间最近公共祖先的连接层级。"""

    SELF = "self"
    INTERNAL = "internal"
    SINGLE = "single"  # 同一条 PCIe 总线上
    MULTIPLE = "multiple"  # 同一 PCIe 交换机下
    HOST_BRIDGE = "host-bridge"  # 同一颗 CPU 下的不同 PCIe 根桥
    NODE = "node"  # 同一个 NUMA 节点
    SYSTEM = "system"  # 跨 NUMA / 跨 socket
    UNKNOWN = "unknown"


class PState(IntEnum):
    """性能状态，数字越大越省电（P0 满血，P8 空闲）。"""

    P0 = 0
    P1 = 1
    P2 = 2
    P3 = 3
    P4 = 4
    P5 = 5
    P6 = 6
    P7 = 7
    P8 = 8
    P9 = 9
    P10 = 10
    P11 = 11
    P12 = 12
    P13 = 13
    P14 = 14
    P15 = 15
    UNKNOWN = 32

    def __str__(self) -> str:  # P8 而非 PState.P8
        return f"P{self.value}" if self is not PState.UNKNOWN else "P?"


class ThrottleReason(IntFlag):
    """时钟降频原因位掩码（``NVML_CLOCKS_EVENT_REASON_*``）。"""

    NONE = 0
    GPU_IDLE = 0x1
    APPLICATIONS_CLOCKS_SETTING = 0x2
    SW_POWER_CAP = 0x4
    HW_SLOWDOWN = 0x8
    SYNC_BOOST = 0x10
    SW_THERMAL_SLOWDOWN = 0x20
    HW_THERMAL_SLOWDOWN = 0x40
    HW_POWER_BRAKE_SLOWDOWN = 0x80
    DISPLAY_CLOCK_SETTING = 0x100
    BOARD_LIMIT = 0x200
    RELIABILITY = 0x400

    @classmethod
    def known_bits(cls) -> int:
        """所有已定义位之和，用于过滤新驱动可能引入的未知位。"""
        return sum(int(member) for member in cls)

    @classmethod
    def from_bits(cls, bits: int) -> ThrottleReason:
        """把驱动返回的整数掩码转成枚举；未知位会被丢弃但保留在 ``ThrottleInfo.bits``。"""
        return cls(bits & cls.known_bits())

    @property
    def names(self) -> tuple[str, ...]:
        """展开为可读名称元组，``NONE`` 返回空元组。"""
        return tuple(member.name for member in type(self) if member and member & self)

    def __str__(self) -> str:
        return "|".join(self.names) or "NONE"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------- 换算
_UNITS: Mapping[str, int] = {
    "B": 1,
    "KiB": 1024,
    "MiB": 1024**2,
    "GiB": 1024**3,
    "TiB": 1024**4,
}


def bytes_to(value: int | None, unit: str = "GiB", *, digits: int = 2) -> float | None:
    """字节换算到指定单位，``None`` 透传。"""
    if value is None:
        return None
    return round(value / _UNITS[unit], digits)


def cuda_version_str(version: int | None) -> str | None:
    """``13040`` → ``"13.4"``（NVML 用 major*1000 + minor*10 编码）。"""
    if not version:
        return None
    return f"{version // 1000}.{(version % 1000) // 10}"


def _round(value: float | None, digits: int = 1) -> float | None:
    return None if value is None else round(value, digits)


# ---------------------------------------------------------------- 值对象
@dataclass(frozen=True, slots=True)
class DriverInfo:
    """驱动与 NVML 版本（整机级别）。"""

    driver_version: str
    nvml_version: str
    cuda_version: int | None
    gpu_count: int
    timestamp: float = field(default_factory=time.time)

    @property
    def cuda(self) -> str | None:
        return cuda_version_str(self.cuda_version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "driver_version": self.driver_version,
            "nvml_version": self.nvml_version,
            "cuda_version": self.cuda,
            "cuda_version_code": self.cuda_version,
            "gpu_count": self.gpu_count,
        }

    def __str__(self) -> str:
        return (
            f"驱动 {self.driver_version} | CUDA {self.cuda or '?'} | "
            f"NVML {self.nvml_version} | {self.gpu_count} 块 GPU"
        )


@dataclass(frozen=True, slots=True)
class MemoryUsage:
    """显存占用（字节）。"""

    total: int
    used: int
    free: int
    reserved: int | None = None

    @property
    def percent(self) -> float:
        return round(self.used / self.total * 100, 1) if self.total else 0.0

    def to_dict(self, *, unit: str = "GiB") -> dict[str, Any]:
        return {
            "total_bytes": self.total,
            "used_bytes": self.used,
            "free_bytes": self.free,
            "reserved_bytes": self.reserved,
            "total": bytes_to(self.total, unit),
            "used": bytes_to(self.used, unit),
            "free": bytes_to(self.free, unit),
            "unit": unit,
            "percent": self.percent,
        }

    def __str__(self) -> str:
        return f"{bytes_to(self.used)}/{bytes_to(self.total)} GiB（{self.percent}%）"


@dataclass(frozen=True, slots=True)
class Utilization:
    """各类引擎的繁忙度（0-100，``None`` 表示读不到）。"""

    gpu: int | None = None
    memory: int | None = None
    encoder: int | None = None
    decoder: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu": self.gpu,
            "memory": self.memory,
            "encoder": self.encoder,
            "decoder": self.decoder,
        }


@dataclass(frozen=True, slots=True)
class PowerInfo:
    """功耗（瓦特）。``limit_watts`` 为当前生效上限。"""

    watts: float | None = None
    limit_watts: float | None = None
    default_limit_watts: float | None = None
    min_limit_watts: float | None = None
    max_limit_watts: float | None = None

    @property
    def percent(self) -> float | None:
        """当前功耗占生效上限的比例。"""
        if self.watts is None or not self.limit_watts:
            return None
        return round(self.watts / self.limit_watts * 100, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "watts": self.watts,
            "limit_watts": self.limit_watts,
            "default_limit_watts": self.default_limit_watts,
            "min_limit_watts": self.min_limit_watts,
            "max_limit_watts": self.max_limit_watts,
            "percent": self.percent,
        }


@dataclass(frozen=True, slots=True)
class TemperatureLimits:
    """温度阈值（摄氏度），驱动未提供的为 ``None``。"""

    slowdown: int | None = None
    shutdown: int | None = None
    max_performance: int | None = None
    max_memory: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "slowdown": self.slowdown,
            "shutdown": self.shutdown,
            "max_performance": self.max_performance,
            "max_memory": self.max_memory,
        }


@dataclass(frozen=True, slots=True)
class ClockReading:
    """单个时钟域的当前 / 最大频率（MHz）。"""

    current: int | None = None
    max: int | None = None

    @property
    def percent(self) -> float | None:
        if self.current is None or not self.max:
            return None
        return round(self.current / self.max * 100, 1)

    def to_dict(self) -> dict[str, Any]:
        return {"mhz": self.current, "max_mhz": self.max, "percent": self.percent}


@dataclass(frozen=True, slots=True)
class Clocks:
    """四个时钟域：SM / 显存 / 图形 / 视频编解码。"""

    sm: ClockReading = ClockReading()
    memory: ClockReading = ClockReading()
    graphics: ClockReading = ClockReading()
    video: ClockReading = ClockReading()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sm": self.sm.to_dict(),
            "memory": self.memory.to_dict(),
            "graphics": self.graphics.to_dict(),
            "video": self.video.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PciLink:
    """PCIe 链路状态与吞吐。"""

    bus_id: str | None = None
    generation: int | None = None
    width: int | None = None
    max_generation: int | None = None
    max_width: int | None = None
    tx_kbps: int | None = None
    rx_kbps: int | None = None

    @property
    def degraded(self) -> bool:
        """链路宽度是否未跑满（如 x8 插在 x16 插槽、MUX 限制）。

        只看宽度：PCIe 代数在空闲时会被驱动主动降速（ASPM / 链路省电），
        属正常现象，单看代数会误报。
        """
        if self.width is None or self.max_width is None:
            return False
        return self.width < self.max_width

    def to_dict(self) -> dict[str, Any]:
        return {
            "bus_id": self.bus_id,
            "generation": self.generation,
            "width": self.width,
            "max_generation": self.max_generation,
            "max_width": self.max_width,
            "tx_kbps": self.tx_kbps,
            "rx_kbps": self.rx_kbps,
            "degraded": self.degraded,
        }


@dataclass(frozen=True, slots=True)
class ThrottleInfo:
    """降频原因。``bits`` 保留驱动的原始掩码，便于排查新版本位定义。"""

    bits: int
    active: ThrottleReason
    supported: ThrottleReason

    @property
    def throttled(self) -> bool:
        """是否处于"被限制"状态（单纯空闲不算）。"""
        return bool(self.active & ~ThrottleReason.GPU_IDLE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "hex": hex(self.bits),
            "active": list(self.active.names),
            "supported": list(self.supported.names),
            "throttled": self.throttled,
        }

    def __str__(self) -> str:
        return ",".join(self.active.names) or "NONE"


@dataclass(frozen=True, slots=True)
class GPUInfo:
    """GPU 静态身份信息（型号、板卡、能力）。"""

    index: int
    uuid: str
    name: str
    brand: Brand = Brand.UNKNOWN
    architecture: Architecture = Architecture.UNKNOWN
    pci_bus_id: str | None = None
    vbios: str | None = None
    serial: str | None = None
    board_part_number: str | None = None
    memory_total: int | None = None
    memory_bus_width: int | None = None
    cores: int | None = None
    max_pcie_generation: int | None = None
    max_pcie_width: int | None = None
    ecc_enabled: bool | None = None
    mig_mode: tuple[int, int] | None = None
    persistence_mode: bool | None = None
    compute_mode: ComputeMode | None = None
    display_active: bool | None = None
    multi_gpu_board: bool | None = None

    def to_dict(self, *, unit: str = "GiB") -> dict[str, Any]:
        return {
            "index": self.index,
            "uuid": self.uuid,
            "name": self.name,
            "brand": self.brand.value,
            "architecture": self.architecture.value,
            "pci_bus_id": self.pci_bus_id,
            "vbios": self.vbios,
            "serial": self.serial,
            "board_part_number": self.board_part_number,
            "memory_total_bytes": self.memory_total,
            "memory_total": bytes_to(self.memory_total, unit),
            "memory_unit": unit,
            "memory_bus_width": self.memory_bus_width,
            "cores": self.cores,
            "max_pcie_generation": self.max_pcie_generation,
            "max_pcie_width": self.max_pcie_width,
            "ecc_enabled": self.ecc_enabled,
            "mig_mode": list(self.mig_mode) if self.mig_mode else None,
            "persistence_mode": self.persistence_mode,
            "compute_mode": self.compute_mode.value if self.compute_mode else None,
            "display_active": self.display_active,
            "multi_gpu_board": self.multi_gpu_board,
        }

    def __str__(self) -> str:
        memory = f"{bytes_to(self.memory_total)} GiB" if self.memory_total else "? GiB"
        return (
            f"[{self.index}] {self.name} | {memory} | {self.architecture} | {self.uuid}"
        )


@dataclass(frozen=True, slots=True)
class GPUStatus:
    """某一时刻的运行状态快照。"""

    index: int
    uuid: str
    name: str
    timestamp: float = field(default_factory=time.time)
    temperature: float | None = None
    fan_percent: int | None = None
    power: PowerInfo = PowerInfo()
    pstate: PState = PState.UNKNOWN
    utilization: Utilization = Utilization()
    memory: MemoryUsage | None = None
    bar1: MemoryUsage | None = None
    clocks: Clocks = Clocks()
    pci: PciLink = PciLink()
    throttle: ThrottleInfo | None = None
    limits: TemperatureLimits = TemperatureLimits()

    @property
    def is_idle(self) -> bool:
        """显卡基本空闲：占用率低、功耗低、处于省电 P-state。"""
        util = self.utilization.gpu
        return util is not None and util <= 5 and self.pstate >= PState.P6

    def to_dict(self, *, unit: str = "GiB") -> dict[str, Any]:
        return {
            "index": self.index,
            "uuid": self.uuid,
            "name": self.name,
            "timestamp": self.timestamp,
            "temperature_c": self.temperature,
            "fan_percent": self.fan_percent,
            "power": self.power.to_dict(),
            "pstate": int(self.pstate),
            "pstate_label": str(self.pstate),
            "utilization": self.utilization.to_dict(),
            "memory": self.memory.to_dict(unit=unit) if self.memory else None,
            "bar1": self.bar1.to_dict(unit=unit) if self.bar1 else None,
            "clocks": self.clocks.to_dict(),
            "pci": self.pci.to_dict(),
            "throttle": self.throttle.to_dict() if self.throttle else None,
            "limits": self.limits.to_dict(),
            "idle": self.is_idle,
        }

    def __str__(self) -> str:
        parts = [f"[{self.index}] {self.name}"]
        if self.utilization.gpu is not None:
            parts.append(f"占用 {self.utilization.gpu}%")
        if self.memory is not None:
            parts.append(f"显存 {self.memory}")
        if self.temperature is not None:
            parts.append(f"{self.temperature:.0f}°C")
        if self.power.watts is not None:
            parts.append(f"{self.power.watts:.0f}W")
        return " | ".join(parts)


@dataclass(frozen=True, slots=True)
class GPUProcess:
    """占用 GPU 的进程。``used_memory`` 为 ``None`` 表示驱动未提供（如 Windows WDDM）。"""

    pid: int
    used_memory: int | None = None
    name: str | None = None
    kind: ProcessKind = ProcessKind.COMPUTE
    gpu_index: int | None = None

    def to_dict(self, *, unit: str = "MiB") -> dict[str, Any]:
        return {
            "pid": self.pid,
            "name": self.name,
            "kind": self.kind.value,
            "gpu_index": self.gpu_index,
            "used_memory_bytes": self.used_memory,
            "used_memory": bytes_to(self.used_memory, unit),
            "unit": unit,
        }

    def __str__(self) -> str:
        memory = (
            f"{bytes_to(self.used_memory, 'MiB', digits=0):.0f} MiB"
            if self.used_memory is not None
            else "-"
        )
        return f"pid={self.pid:<7} {self.name or '?':<28} {memory:<10} {self.kind}"


@dataclass(frozen=True, slots=True)
class TopologyMatrix:
    """GPU 之间的连接拓扑矩阵（对称）。"""

    uuids: tuple[str, ...]
    names: tuple[str, ...]
    levels: tuple[tuple[TopologyLevel, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "uuids": list(self.uuids),
            "names": list(self.names),
            "matrix": [[level.value for level in row] for row in self.levels],
        }

    def __str__(self) -> str:
        rows = ["  ".join(f"{level.value:<12}" for level in row) for row in self.levels]
        return "\n".join(rows)


@dataclass(frozen=True, slots=True)
class HealthIssue:
    """单条健康检查结论。"""

    level: Severity
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level.value, "code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class HealthReport:
    """健康检查汇总。``level`` 取所有 issue 中最严重的一级。"""

    index: int
    uuid: str
    name: str
    level: Severity = Severity.INFO
    issues: tuple[HealthIssue, ...] = ()
    status: GPUStatus | None = None

    @staticmethod
    def worst(issues: Iterable[HealthIssue]) -> Severity:
        """取一组问题中最严重的一级，空集合视为 ``INFO``。"""
        order = {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}
        return max(
            (issue.level for issue in issues),
            key=lambda level: order[level],
            default=Severity.INFO,
        )

    @property
    def healthy(self) -> bool:
        return self.level is not Severity.CRITICAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "uuid": self.uuid,
            "name": self.name,
            "level": self.level.value,
            "healthy": self.healthy,
            "issues": [issue.to_dict() for issue in self.issues],
            "status": self.status.to_dict() if self.status else None,
        }

    def __str__(self) -> str:
        body = "；".join(issue.message for issue in self.issues) or "各项指标正常"
        return f"[{self.index}] {self.name} → {self.level.value}: {body}"
