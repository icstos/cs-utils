"""读取层：把 NVML 的原始返回值翻译成 :mod:`cs.nvidia.models` 的数据类。

这一层只做"翻译"，不做生命周期管理，也不抛平台相关的异常：
读不到的指标一律 ``None``（风扇、ECC、MIG 在消费级卡上普遍缺失）。

拆成独立模块的好处是门面（``core.py``）只剩编排逻辑，
而这里是纯函数集合，便于单测和替换（例如换成 ``nvidia-smi`` 解析实现）。
"""

from __future__ import annotations

from typing import Any

from .bindings import as_text, ensure, optional
from .models import (
    Architecture,
    Brand,
    ClockReading,
    Clocks,
    ComputeMode,
    GPUInfo,
    GPUProcess,
    GPUStatus,
    HealthIssue,
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
)

__all__ = [
    "diagnose",
    "read_info",
    "read_memory",
    "read_processes",
    "read_status",
    "read_throttle",
    "read_topology",
    "read_utilization",
]

# NVML 枚举值 → Python 枚举。这些数值由 NVML ABI 固定，不随 pynvml 版本变化
_BRANDS: dict[int, Brand] = {
    0: Brand.UNKNOWN,
    1: Brand.QUADRO,
    2: Brand.TESLA,
    3: Brand.NVS,
    4: Brand.GRID,
    5: Brand.GEFORCE,
    6: Brand.TITAN,
    7: Brand.VAPPS,
    8: Brand.VPC,
    9: Brand.VCS,
    10: Brand.VWS,
    11: Brand.CLOUD_GAMING,
    12: Brand.QUADRO_RTX,
    13: Brand.RTX,
    14: Brand.NVIDIA,
    15: Brand.GEFORCE_RTX,
    16: Brand.TITAN_RTX,
    17: Brand.DLA,
    18: Brand.VGAMEDEV,
    19: Brand.NPU,
}
_ARCHITECTURES: dict[int, Architecture] = {
    2: Architecture.KEPLER,
    3: Architecture.MAXWELL,
    4: Architecture.PASCAL,
    5: Architecture.VOLTA,
    6: Architecture.TURING,
    7: Architecture.AMPERE,
    8: Architecture.ADA,
    9: Architecture.HOPPER,
    10: Architecture.BLACKWELL,
    11: Architecture.DLA,
    12: Architecture.DLA2,
    13: Architecture.RUBIN,
    15: Architecture.NPU3,
}
_COMPUTE_MODES: dict[int, ComputeMode] = {
    0: ComputeMode.DEFAULT,
    1: ComputeMode.EXCLUSIVE_THREAD,
    2: ComputeMode.PROHIBITED,
    3: ComputeMode.EXCLUSIVE_PROCESS,
}
_TOPOLOGY: dict[int, TopologyLevel] = {
    0: TopologyLevel.INTERNAL,
    10: TopologyLevel.SINGLE,
    20: TopologyLevel.MULTIPLE,
    30: TopologyLevel.HOST_BRIDGE,
    40: TopologyLevel.NODE,
    50: TopologyLevel.SYSTEM,
}
# 温度传感器 / 阈值枚举（NVML ABI 固定值）
_SENSOR_GPU = 0
_THRESHOLD_SHUTDOWN = 0
_THRESHOLD_SLOWDOWN = 1
_THRESHOLD_MAX_MEMORY = 2
_THRESHOLD_MAX_PERF = 3
# 时钟域
_CLOCK_GRAPHICS = 0
_CLOCK_SM = 1
_CLOCK_MEMORY = 2
_CLOCK_VIDEO = 3
# PCIe 吞吐计数器
_PCIE_TX = 0
_PCIE_RX = 1


def _enum(mapping: dict[int, Any], value: Any, default: Any) -> Any:
    return mapping.get(int(value), default) if value is not None else default


def _clock(module: Any, device: Any, clock_type: int) -> ClockReading:
    """读取单个时钟域的当前值 / 最大值（MHz），缺失时该字段为 ``None``。"""
    current = optional(
        lambda: module.nvmlDeviceGetClockInfo(device, clock_type),
        feature="时钟频率",
    )
    maximum = optional(
        lambda: module.nvmlDeviceGetMaxClockInfo(device, clock_type),
        feature="最大时钟频率",
    )
    return ClockReading(
        current=int(current) if current is not None else None,
        max=int(maximum) if maximum is not None else None,
    )


def read_memory(device: Any) -> MemoryUsage | None:
    """显存占用（字节）。"""
    module = ensure()
    info = optional(lambda: module.nvmlDeviceGetMemoryInfo(device), feature="显存占用")
    if info is None:
        return None
    return MemoryUsage(
        total=int(info.total),
        used=int(info.used),
        free=int(info.free),
        reserved=int(getattr(info, "reserved", 0) or 0) or None,
    )


def _bar1(module: Any, device: Any) -> MemoryUsage | None:
    info = optional(
        lambda: module.nvmlDeviceGetBAR1MemoryInfo(device), feature="BAR1 显存"
    )
    if info is None:
        return None
    return MemoryUsage(
        total=int(info.bar1Total), used=int(info.bar1Used), free=int(info.bar1Free)
    )


def _fan(module: Any, device: Any) -> int | None:
    """风扇转速百分比。消费级笔记本通常不支持，退化为 ``None``。"""
    speed = optional(lambda: module.nvmlDeviceGetFanSpeed(device), feature="风扇转速")
    if speed is not None:
        return int(speed)
    counter = getattr(module, "nvmlDeviceGetNumFans", None)
    per_fan = getattr(module, "nvmlDeviceGetFanSpeed_v2", None)
    if counter is None or per_fan is None:
        return None
    count = optional(lambda: counter(device), feature="风扇数量") or 0
    speeds = [
        optional(lambda index=index: per_fan(device, index), feature="风扇转速")
        for index in range(int(count))
    ]
    valid = [int(item) for item in speeds if item is not None]
    return max(valid) if valid else None


def _power(module: Any, device: Any) -> PowerInfo:
    """功耗与上限。笔记本上 ``PowerManagementLimit`` 常不支持，
    退化为 ``EnforcedPowerLimit``（动态上限），两者都拿不到则为 ``None``。"""
    milliwatts = optional(
        lambda: module.nvmlDeviceGetPowerUsage(device), feature="功耗"
    )
    limit = optional(
        lambda: module.nvmlDeviceGetEnforcedPowerLimit(device), feature="功耗上限"
    )
    if limit is None:
        limit = optional(
            lambda: module.nvmlDeviceGetPowerManagementLimit(device), feature="功耗上限"
        )
    default_limit = optional(
        lambda: module.nvmlDeviceGetPowerManagementDefaultLimit(device),
        feature="默认功耗上限",
    )
    constraints = optional(
        lambda: module.nvmlDeviceGetPowerManagementLimitConstraints(device),
        feature="功耗上限区间",
    )
    low, high = (
        (int(constraints[0]), int(constraints[1])) if constraints else (None, None)
    )

    def watt(value: int | None) -> float | None:
        return round(value / 1000, 2) if value is not None else None

    return PowerInfo(
        watts=watt(milliwatts),
        limit_watts=watt(limit),
        default_limit_watts=watt(default_limit),
        min_limit_watts=watt(low),
        max_limit_watts=watt(high),
    )


def read_utilization(device: Any) -> Utilization:
    """GPU / 显存 / 编解码引擎占用率。"""
    module = ensure()
    rates = optional(
        lambda: module.nvmlDeviceGetUtilizationRates(device), feature="占用率"
    )
    encoder = optional(
        lambda: module.nvmlDeviceGetEncoderUtilization(device), feature="编码器占用"
    )
    decoder = optional(
        lambda: module.nvmlDeviceGetDecoderUtilization(device), feature="解码器占用"
    )
    return Utilization(
        gpu=int(rates.gpu) if rates is not None else None,
        memory=int(rates.memory) if rates is not None else None,
        encoder=int(encoder[0]) if encoder else None,
        decoder=int(decoder[0]) if decoder else None,
    )


def _pci(module: Any, device: Any) -> PciLink:
    info = optional(lambda: module.nvmlDeviceGetPciInfo(device), feature="PCI 信息")
    return PciLink(
        bus_id=as_text(getattr(info, "busId", None)),
        generation=optional(
            lambda: module.nvmlDeviceGetCurrPcieLinkGeneration(device),
            feature="PCIe 代数",
        ),
        width=optional(
            lambda: module.nvmlDeviceGetCurrPcieLinkWidth(device), feature="PCIe 宽度"
        ),
        max_generation=optional(
            lambda: module.nvmlDeviceGetMaxPcieLinkGeneration(device),
            feature="PCIe 最大代数",
        ),
        max_width=optional(
            lambda: module.nvmlDeviceGetMaxPcieLinkWidth(device),
            feature="PCIe 最大宽度",
        ),
        tx_kbps=optional(
            lambda: module.nvmlDeviceGetPcieThroughput(device, _PCIE_TX),
            feature="PCIe 发送吞吐",
        ),
        rx_kbps=optional(
            lambda: module.nvmlDeviceGetPcieThroughput(device, _PCIE_RX),
            feature="PCIe 接收吞吐",
        ),
    )


def read_throttle(device: Any) -> ThrottleInfo | None:
    """当前 / 支持的降频原因位掩码。"""
    module = ensure()
    current = optional(
        lambda: (
            module.nvmlDeviceGetCurrentClocksEventReasons(device)
            if hasattr(module, "nvmlDeviceGetCurrentClocksEventReasons")
            else module.nvmlDeviceGetCurrentClocksThrottleReasons(device)
        ),
        feature="降频原因",
    )
    if current is None:
        return None
    supported = optional(
        lambda: (
            module.nvmlDeviceGetSupportedClocksEventReasons(device)
            if hasattr(module, "nvmlDeviceGetSupportedClocksEventReasons")
            else module.nvmlDeviceGetSupportedClocksThrottleReasons(device)
        ),
        feature="支持的降频原因",
    )
    bits = int(current)
    return ThrottleInfo(
        bits=bits,
        active=ThrottleReason.from_bits(bits),
        supported=ThrottleReason.from_bits(int(supported or 0)),
    )


def _temperature_limits(module: Any, device: Any) -> TemperatureLimits:
    def threshold(kind: int) -> int | None:
        value = optional(
            lambda: module.nvmlDeviceGetTemperatureThreshold(device, kind),
            feature="温度阈值",
        )
        return int(value) if value is not None else None

    return TemperatureLimits(
        slowdown=threshold(_THRESHOLD_SLOWDOWN),
        shutdown=threshold(_THRESHOLD_SHUTDOWN),
        max_performance=threshold(_THRESHOLD_MAX_PERF),
        max_memory=threshold(_THRESHOLD_MAX_MEMORY),
    )


def _process_names(pids: set[int]) -> dict[int, str]:
    """用 psutil 补进程名；psutil 不可用时静默跳过。"""
    if not pids:
        return {}
    try:
        import psutil
    except ImportError:  # pragma: no cover
        return {}
    names: dict[int, str] = {}
    for pid in pids:
        try:
            names[pid] = psutil.Process(pid).name()
        except Exception:  # noqa: BLE001, S112 - 进程可能已退出 / 无权限
            continue
    return names


def read_processes(device: Any, index: int) -> tuple[GPUProcess, ...]:
    """采集占用该卡的进程，按 pid 去重（同一进程可能同时出现在多个列表里）。"""
    module = ensure()
    sources = (
        (ProcessKind.COMPUTE, "nvmlDeviceGetComputeRunningProcesses"),
        (ProcessKind.GRAPHICS, "nvmlDeviceGetGraphicsRunningProcesses"),
        (ProcessKind.MPS, "nvmlDeviceGetMPSComputeRunningProcesses"),
    )
    collected: dict[int, GPUProcess] = {}
    for kind, name in sources:
        reader = getattr(module, name, None)
        if reader is None:  # 旧版 NVML 没有 MPS 接口
            continue
        items = optional(
            lambda reader=reader: reader(device), default=[], feature=f"{kind}进程"
        )
        for item in items or ():
            pid = int(item.pid)
            used = getattr(item, "usedGpuMemory", None)
            entry = GPUProcess(
                pid=pid,
                used_memory=int(used) if used is not None else None,
                kind=kind,
                gpu_index=index,
            )
            previous = collected.get(pid)
            # 同一 pid 优先保留显存已知的记录（Windows WDDM 常常给不出显存）
            if previous is None or (previous.used_memory is None and entry.used_memory):
                collected[pid] = entry
    names = _process_names(set(collected))
    if not names:
        return tuple(collected.values())
    return tuple(
        GPUProcess(
            pid=proc.pid,
            used_memory=proc.used_memory,
            name=names.get(proc.pid),
            kind=proc.kind,
            gpu_index=proc.gpu_index,
        )
        for proc in collected.values()
    )


def read_info(index: int, device: Any) -> GPUInfo:
    """静态身份信息（型号、架构、显存总量、VBIOS、ECC/MIG…）。"""
    module = ensure()

    def get(name: str) -> Any:
        function = getattr(module, name, None)
        if function is None:  # 旧版 NVML 缺接口
            return None
        return optional(lambda: function(device), feature=name)

    memory = read_memory(device)
    ecc = get("nvmlDeviceGetEccMode")
    mig = get("nvmlDeviceGetMigMode")
    persistence = get("nvmlDeviceGetPersistenceMode")
    compute_mode = get("nvmlDeviceGetComputeMode")
    display = get("nvmlDeviceGetDisplayActive")
    multi_gpu = get("nvmlDeviceGetMultiGpuBoard")
    pci = get("nvmlDeviceGetPciInfo")

    return GPUInfo(
        index=index,
        uuid=as_text(get("nvmlDeviceGetUUID")) or f"index:{index}",
        name=as_text(get("nvmlDeviceGetName")) or "Unknown GPU",
        brand=_enum(_BRANDS, get("nvmlDeviceGetBrand"), Brand.UNKNOWN),
        architecture=_enum(
            _ARCHITECTURES, get("nvmlDeviceGetArchitecture"), Architecture.UNKNOWN
        ),
        pci_bus_id=as_text(getattr(pci, "busId", None)),
        vbios=as_text(get("nvmlDeviceGetVbiosVersion")),
        serial=as_text(get("nvmlDeviceGetSerial")),
        board_part_number=as_text(get("nvmlDeviceGetBoardPartNumber")),
        memory_total=memory.total if memory else None,
        memory_bus_width=get("nvmlDeviceGetMemoryBusWidth"),
        cores=get("nvmlDeviceGetNumGpuCores"),
        max_pcie_generation=get("nvmlDeviceGetMaxPcieLinkGeneration"),
        max_pcie_width=get("nvmlDeviceGetMaxPcieLinkWidth"),
        ecc_enabled=bool(ecc[0]) if ecc else None,
        mig_mode=(int(mig[0]), int(mig[1])) if mig else None,
        persistence_mode=bool(persistence) if persistence is not None else None,
        compute_mode=_enum(_COMPUTE_MODES, compute_mode, None),
        display_active=bool(display) if display is not None else None,
        multi_gpu_board=bool(multi_gpu) if multi_gpu is not None else None,
    )


def read_status(device: Any, *, index: int, uuid: str, name: str) -> GPUStatus:
    """一次运行状态快照（温度、功耗、占用、显存、频率、PCIe、降频）。"""
    module = ensure()
    temperature = optional(
        lambda: module.nvmlDeviceGetTemperature(device, _SENSOR_GPU), feature="GPU 温度"
    )
    pstate = optional(
        lambda: module.nvmlDeviceGetPerformanceState(device), feature="性能状态"
    )
    return GPUStatus(
        index=index,
        uuid=uuid,
        name=name,
        temperature=float(temperature) if temperature is not None else None,
        fan_percent=_fan(module, device),
        power=_power(module, device),
        pstate=PState(int(pstate)) if pstate is not None else PState.UNKNOWN,
        utilization=read_utilization(device),
        memory=read_memory(device),
        bar1=_bar1(module, device),
        clocks=Clocks(
            sm=_clock(module, device, _CLOCK_SM),
            memory=_clock(module, device, _CLOCK_MEMORY),
            graphics=_clock(module, device, _CLOCK_GRAPHICS),
            video=_clock(module, device, _CLOCK_VIDEO),
        ),
        pci=_pci(module, device),
        throttle=read_throttle(device),
        limits=_temperature_limits(module, device),
    )


def read_topology(
    devices: list[Any], uuids: list[str], names: list[str]
) -> TopologyMatrix:
    """多卡互联拓扑矩阵（对称，对角线为 ``self``）。"""
    module = ensure()
    levels: list[tuple[TopologyLevel, ...]] = []
    for left_index, left in enumerate(devices):
        row: list[TopologyLevel] = []
        for right_index, right in enumerate(devices):
            if left_index == right_index:
                row.append(TopologyLevel.SELF)
                continue
            value = optional(
                lambda left=left, right=right: (
                    module.nvmlDeviceGetTopologyCommonAncestor(left, right)
                ),
                feature="GPU 拓扑",
            )
            row.append(_TOPOLOGY.get(int(value), TopologyLevel.UNKNOWN))
        levels.append(tuple(row))
    return TopologyMatrix(uuids=tuple(uuids), names=tuple(names), levels=tuple(levels))


def diagnose(status: GPUStatus) -> tuple[HealthIssue, ...]:
    """根据一次快照给出健康结论（纯函数，便于单测）。"""
    issues: list[HealthIssue] = []

    limits = status.limits
    if status.temperature is not None and limits.slowdown:
        if status.temperature >= limits.slowdown:
            issues.append(
                HealthIssue(
                    Severity.CRITICAL,
                    "temperature.slowdown",
                    f"温度 {status.temperature:.0f}°C 已达降频阈值 {limits.slowdown}°C",
                )
            )
        elif status.temperature >= limits.slowdown - 8:
            issues.append(
                HealthIssue(
                    Severity.WARNING,
                    "temperature.high",
                    f"温度 {status.temperature:.0f}°C 接近降频阈值 {limits.slowdown}°C",
                )
            )

    if status.memory is not None:
        percent = status.memory.percent
        if percent >= 98:
            issues.append(
                HealthIssue(
                    Severity.CRITICAL,
                    "memory.exhausted",
                    f"显存占用 {percent}%，即将耗尽",
                )
            )
        elif percent >= 90:
            issues.append(
                HealthIssue(
                    Severity.WARNING,
                    "memory.pressure",
                    f"显存占用 {percent}%，压力偏高",
                )
            )

    throttle = status.throttle
    if throttle is not None and throttle.throttled:
        hardware = throttle.active & (
            ThrottleReason.HW_THERMAL_SLOWDOWN
            | ThrottleReason.HW_POWER_BRAKE_SLOWDOWN
            | ThrottleReason.HW_SLOWDOWN
        )
        issues.append(
            HealthIssue(
                Severity.WARNING if hardware else Severity.INFO,
                "clock.throttle",
                f"当前降频原因: {throttle}",
            )
        )

    if status.pci.degraded:
        issues.append(
            HealthIssue(
                Severity.INFO,
                "pcie.degraded",
                f"PCIe 链路仅 x{status.pci.width}，未跑满 x{status.pci.max_width}",
            )
        )

    return tuple(issues)
