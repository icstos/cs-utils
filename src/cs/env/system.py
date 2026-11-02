"""平台、CPU、内存、磁盘、网络、GPU 的只读采集。

规则：**探测失败不抛异常**。拿不到的信息留空（``None`` / 空元组），
让调用方与模型都能安全地拿到“部分可用”的结果；只有明确的编程错误才抛出。
GPU 走 :mod:`cs.nvidia`（NVML），未安装或非 NVIDIA 机器上返回不可用。
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import subprocess
import sys
import time
from functools import cache
from pathlib import Path

import psutil

from ..my_types import PlatformType
from .models import (
    AVX,
    CPUInfo,
    DiskInfo,
    GPUDevice,
    GPURuntime,
    MemoryInfo,
    NetworkInfo,
    PlatformInfo,
    Snapshot,
    Throughput,
)

__all__ = [
    "cpu_avx",
    "cpu_info",
    "cpu_model",
    "cpu_vendor",
    "disk_io",
    "disk_usage",
    "disks",
    "distro",
    "gpu_processes",
    "gpu_runtime",
    "is_elevated",
    "memory_info",
    "network_info",
    "platform_info",
    "platform_kind",
    "snapshot",
]

#: 通过环境变量强制指定指令集，如 ``CS_ENV_AVX=avx2``
AVX_ENV = "CS_ENV_AVX"


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _capture(argv: list[str], timeout: float = 3.0) -> str:
    """执行探测命令，失败一律返回空串。"""
    try:
        completed = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.decode("utf-8", "replace")


def _windows_value(field_name: str) -> str:
    """读注册表里的 CPU 描述（比已废弃的 wmic 稳定且不走子进程）。"""
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            value, _ = winreg.QueryValueEx(key, field_name)
            return str(value).strip()
    except (ImportError, OSError):
        return ""


# ---------------------------------------------------------------- 平台


def platform_kind() -> PlatformType:
    """当前运行平台；Android 先于 Linux 判断。"""
    if os.environ.get("ANDROID_ARGUMENT") or os.environ.get("ANDROID_ROOT"):
        return PlatformType.ANDROID
    match sys.platform:
        case "win32" | "cygwin" | "msys":
            return PlatformType.WINDOWS
        case "linux":
            return PlatformType.LINUX
        case "darwin":
            return PlatformType.DARWIN
        case _:
            return PlatformType.OTHER


@cache
def distro() -> str:
    """发行版名称，如 ``Ubuntu 24.04.1 LTS`` / ``Windows 11`` / ``macOS 15.1``。"""
    match platform_kind():
        case PlatformType.LINUX:
            info = {}
            for line in _read_text("/etc/os-release").splitlines():
                key, _, value = line.partition("=")
                info[key.strip()] = value.strip().strip('"')
            if info.get("PRETTY_NAME"):
                return info["PRETTY_NAME"]
            if info.get("NAME"):
                return f"{info['NAME']} {info.get('VERSION_ID', '')}".strip()
            return "Linux"
        case PlatformType.DARWIN:
            return f"macOS {platform.mac_ver()[0] or platform.release()}"
        case PlatformType.WINDOWS:
            return f"Windows {platform.release()}（{platform.version()}）"
        case PlatformType.ANDROID:
            return f"Android {platform.release()}"
        case _:
            return platform.system() or "Unknown"


def is_elevated() -> bool | None:
    """是否具备管理员 / root 权限；无法判断时返回 ``None``。"""
    if os.name == "nt":
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            return None
    try:
        return os.geteuid() == 0
    except AttributeError:
        return None


@cache
def platform_info() -> PlatformInfo:
    """操作系统与 Python 运行时的完整描述（结果缓存）。"""
    return PlatformInfo(
        system=platform.system(),
        kind=platform_kind(),
        release=platform.release(),
        version=platform.version(),
        machine=platform.machine(),
        processor=platform.processor(),
        distro=distro(),
        hostname=platform.node(),
        python_version=platform.python_version(),
        python_executable=sys.executable,
        python_implementation=platform.python_implementation(),
        is_64bit=sys.maxsize > 2**32,
        elevated=is_elevated(),
    )


# ---------------------------------------------------------------- CPU


@cache
def cpu_model() -> str:
    """CPU 型号名；取不到时退回架构名。"""
    match platform_kind():
        case PlatformType.LINUX:
            for line in _read_text("/proc/cpuinfo").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[-1].strip()
        case PlatformType.DARWIN:
            brand = _capture(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
            if brand:
                return brand
            if platform.machine() == "arm64":
                return "Apple Silicon"
        case PlatformType.WINDOWS:
            name = _windows_value("ProcessorNameString")
            if name:
                return name
    return platform.processor().strip() or platform.machine() or "Unknown CPU"


@cache
def cpu_vendor() -> str:
    """CPU 厂商，如 ``GenuineIntel`` / ``AuthenticAMD`` / ``Apple``。"""
    match platform_kind():
        case PlatformType.LINUX:
            for line in _read_text("/proc/cpuinfo").splitlines():
                if line.lower().startswith("vendor_id"):
                    return line.split(":", 1)[-1].strip()
        case PlatformType.DARWIN:
            brand = cpu_model()
            match brand:
                case brand if brand.startswith("Apple"):
                    return "Apple"
                case brand if "AMD" in brand:
                    return "AMD"
                case _:
                    return "Intel"
        case PlatformType.WINDOWS:
            vendor = _windows_value("VendorIdentifier")
            if vendor:
                return vendor
    return "Unknown"


@cache
def cpu_avx() -> AVX:
    """CPU 支持的 AVX 等级。

    优先 ``CS_ENV_AVX``；Linux 读 ``/proc/cpuinfo``、macOS 读 ``sysctl``（准确）；
    Windows 没有稳定的免依赖探测手段，按 x64 目标默认 ``avx2``，可用环境变量覆盖。
    """
    override = os.getenv(AVX_ENV)
    if override:
        return AVX.parse(override)

    match platform_kind():
        case PlatformType.LINUX:
            text = _read_text("/proc/cpuinfo").lower()
        case PlatformType.DARWIN:
            text = " ".join(
                _capture(["sysctl", "-n", key]).lower()
                for key in (
                    "machdep.cpu.features",
                    "machdep.cpu.leaf7_features",
                    "hw.optional.avx2_0",
                )
            )
        case PlatformType.WINDOWS:
            text = "avx2"
        case _:
            text = ""

    match text:
        case text if "avx512f" in text:
            return AVX.AVX512
        case text if "avx2" in text:
            return AVX.AVX2
        case text if re.search(r"\bavx\b", text):
            return AVX.AVX
        case _:
            return AVX.BASIC


def cpu_info(*, interval: float | None = 0.1, per_core: bool = False) -> CPUInfo:
    """CPU 状态；``interval=None`` 表示不阻塞（复用上次采样）。"""
    cores: tuple[float, ...] = ()
    if per_core:
        cores = tuple(psutil.cpu_percent(interval=interval, percpu=True))
        percent = round(sum(cores) / len(cores), 1) if cores else 0.0
    else:
        percent = psutil.cpu_percent(interval=interval)

    try:
        freq = psutil.cpu_freq()
    except (NotImplementedError, OSError):
        freq = None
    try:
        load: tuple[float, float, float] | None = os.getloadavg()
    except (AttributeError, OSError):
        load = None

    return CPUInfo(
        model=cpu_model(),
        vendor=cpu_vendor(),
        arch=platform.machine(),
        physical_cores=psutil.cpu_count(logical=False) or 0,
        logical_cores=psutil.cpu_count(logical=True) or 0,
        percent=round(percent, 1),
        per_core=cores,
        freq_mhz=round(freq.current, 1) if freq and freq.current else None,
        freq_max_mhz=round(freq.max, 1) if freq and freq.max else None,
        avx=cpu_avx(),
        load_avg=load,
    )


# ---------------------------------------------------------------- 内存


def memory_info() -> MemoryInfo:
    """物理内存 + 交换分区。"""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return MemoryInfo(
        total=memory.total,
        available=memory.available,
        used=memory.used,
        free=memory.free,
        percent=round(memory.percent, 1),
        cached=getattr(memory, "cached", None),
        swap_total=swap.total,
        swap_used=swap.used,
        swap_percent=round(swap.percent, 1),
    )


# ---------------------------------------------------------------- 磁盘


def disks(*, all: bool = False) -> tuple[DiskInfo, ...]:
    """所有可用的挂载点；``all=False`` 时跳过光驱等无文件系统的设备。"""
    result: list[DiskInfo] = []
    for part in psutil.disk_partitions(all=all):
        if not part.fstype:
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue
        result.append(
            DiskInfo(
                device=part.device,
                mountpoint=part.mountpoint,
                fstype=part.fstype,
                total=usage.total,
                used=usage.used,
                free=usage.free,
                percent=round(usage.percent, 1),
                read_only="ro" in part.opts.split(","),
            )
        )
    return tuple(result)


def _mount_of(target: Path) -> str:
    """找到包含 ``target`` 的挂载点（取最长前缀匹配）。"""
    best = target.anchor or os.sep
    text = os.path.normcase(os.path.abspath(target))
    for part in psutil.disk_partitions(all=True):
        mount = os.path.normcase(os.path.abspath(part.mountpoint))
        if text.startswith(mount) and len(mount) > len(os.path.normcase(best)):
            best = part.mountpoint
    return best


def disk_usage(path: str | Path = ".") -> DiskInfo:
    """指定路径所在分区（或文件）的容量与占用。"""
    target = Path(path)
    usage = psutil.disk_usage(str(target))
    mount = _mount_of(target)
    device, fstype = "", ""
    for part in psutil.disk_partitions(all=True):
        if os.path.normcase(os.path.abspath(part.mountpoint)) == os.path.normcase(
            os.path.abspath(mount)
        ):
            device, fstype = part.device, part.fstype
            break
    return DiskInfo(
        device=device or mount,
        mountpoint=mount,
        fstype=fstype,
        total=usage.total,
        used=usage.used,
        free=usage.free,
        percent=round(usage.percent, 1),
    )


def disk_io(interval: float = 1.0) -> Throughput:
    """磁盘累计读写与采样速率（会阻塞 ``interval`` 秒）。"""
    first = psutil.disk_io_counters()
    if first is None:
        return Throughput()
    time.sleep(interval)
    second = psutil.disk_io_counters() or first
    elapsed = max(interval, 1e-6)
    return Throughput(
        read_bytes=second.read_bytes,
        write_bytes=second.write_bytes,
        read_rate=round((second.read_bytes - first.read_bytes) / elapsed, 1),
        write_rate=round((second.write_bytes - first.write_bytes) / elapsed, 1),
        seconds=interval,
    )


# ---------------------------------------------------------------- 网络


def network_info() -> NetworkInfo:
    """累计收发量与在线网卡。"""
    counters = psutil.net_io_counters()
    interfaces = tuple(
        name for name, stats in psutil.net_if_stats().items() if stats.isup
    )
    if counters is None:
        return NetworkInfo(interfaces=interfaces)
    return NetworkInfo(
        bytes_sent=counters.bytes_sent,
        bytes_recv=counters.bytes_recv,
        packets_sent=counters.packets_sent,
        packets_recv=counters.packets_recv,
        interfaces=interfaces,
    )


# ---------------------------------------------------------------- GPU


def _cuda_text(version: int | None) -> str | None:
    """NVML 的 CUDA 版本是整数（12040），转成 ``12.4``。"""
    if version is None:
        return None
    if isinstance(version, int):
        return f"{version // 1000}.{(version % 1000) // 10}"
    return str(version)


def _gpu_device(status: object) -> GPUDevice:
    memory = getattr(status, "memory", None)
    utilization = getattr(status, "utilization", None)
    power = getattr(status, "power", None)
    return GPUDevice(
        index=getattr(status, "index", 0),
        name=getattr(status, "name", "Unknown GPU"),
        total_memory=getattr(memory, "total", None),
        used_memory=getattr(memory, "used", None),
        free_memory=getattr(memory, "free", None),
        utilization=(
            float(utilization.gpu)
            if utilization is not None and utilization.gpu is not None
            else None
        ),
        temperature=getattr(status, "temperature", None),
        power=getattr(power, "watts", None),
    )


def gpu_runtime() -> GPURuntime:
    """GPU 运行环境；无 NVML / 无显卡时返回不可用而不是抛异常。"""
    try:
        from ..nvidia import NVIDIA
    except ImportError:
        return GPURuntime()

    try:
        if not NVIDIA.available():
            return GPURuntime()
        driver = NVIDIA.driver()
        devices = tuple(_gpu_device(status) for status in NVIDIA.statuses())
    except Exception:  # noqa: BLE001 - 驱动异常不应影响整机采集
        return GPURuntime()

    name = driver.driver_version
    return GPURuntime(
        available=True,
        driver=None if name in (None, "", "unknown") else name,
        cuda=_cuda_text(driver.cuda_version),
        devices=devices,
    )


def gpu_processes() -> dict[int, tuple[int, int | None]]:
    """``pid -> (显卡序号, 显存字节)`` 映射；无 GPU 时为空。"""
    try:
        from ..nvidia import NVIDIA
    except ImportError:
        return {}
    try:
        if not NVIDIA.available():
            return {}
        return {
            item.pid: (item.gpu_index or 0, item.used_memory)
            for item in NVIDIA.all_processes()
        }
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------- 快照


def snapshot(
    *,
    top: int = 5,
    interval: float | None = 0.1,
    with_gpu: bool = True,
) -> Snapshot:
    """一次性采集整机状态。"""
    from .process import processes

    return Snapshot(
        timestamp=time.time(),
        platform=platform_info(),
        cpu=cpu_info(interval=interval),
        memory=memory_info(),
        disks=disks(),
        gpu=gpu_runtime() if with_gpu else GPURuntime(),
        network=network_info(),
        processes=tuple(processes(limit=top, sort="cpu", include_gpu=with_gpu)),
    )
