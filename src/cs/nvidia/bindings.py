"""NVML 绑定层：唯一与 ``pynvml`` 打交道的地方。

职责边界
--------
1. **懒加载** ``pynvml``（``import cs.nvidia`` 不要求已装驱动）；
2. **幂等初始化**：``nvmlInit`` 全局只调一次，进程 fork 后在子进程里自动重来；
3. **异常收敛**：把 ``NVMLError_*`` 映射成 :mod:`cs.nvidia.errors` 的领域异常；
4. **可缺失指标**：笔记本 / 消费级卡大量指标会抛 ``NotSupported``，
   统一走 :func:`optional` 返回 ``None``，而不是让整次调用失败。

调用方不应该直接 ``import pynvml``。
"""

from __future__ import annotations

import atexit
import os
import threading
from collections.abc import Callable
from types import ModuleType
from typing import Any

from .errors import (
    GPUError,
    GPULostError,
    GPUNotFoundError,
    GPUNotSupportedError,
    GPUUnavailableError,
    MissingDependencyError,
)

__all__ = [
    "as_text",
    "device_count",
    "ensure",
    "handle",
    "handle_by_bus",
    "handle_by_uuid",
    "is_initialized",
    "optional",
    "query",
    "shutdown",
]

# NVML 的异常是运行期动态生成的类，按类名匹配比 isinstance 更稳（跨版本友好）
_UNAVAILABLE = frozenset(
    {
        "NVMLError_LibraryNotFound",
        "NVMLError_DriverNotLoaded",
        "NVMLError_Uninitialized",
        "NVMLError_LibRmVersionMismatch",
    }
)
_UNSUPPORTED = frozenset(
    {
        "NVMLError_NotSupported",
        "NVMLError_NoPermission",
        "NVMLError_InsufficientPermissions",
        "NVMLError_NotPermitted",
        "NVMLError_FunctionNotFound",
    }
)
_NOT_FOUND = frozenset({"NVMLError_NotFound", "NVMLError_InvalidArgument"})
_LOST = frozenset({"NVMLError_GpuIsLost"})


class _Session:
    """NVML 的进程级状态（受模块锁保护）。"""

    __slots__ = ("initialized", "module", "pid")

    def __init__(self) -> None:
        self.module: ModuleType | None = None
        self.initialized = False
        self.pid = 0


_lock = threading.RLock()
_session = _Session()
_atexit_registered = False


# ---------------------------------------------------------------- 加载
def _load() -> ModuleType:
    """导入 ``pynvml``；缺失时给出可执行的安装提示。"""
    if _session.module is not None:
        return _session.module
    try:
        import pynvml
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise MissingDependencyError() from exc
    _session.module = pynvml
    return pynvml


def _register_atexit() -> None:
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(shutdown)
        _atexit_registered = True


def ensure() -> ModuleType:
    """确保 NVML 已初始化并返回 ``pynvml`` 模块（线程安全、幂等、fork 安全）。"""
    module = _load()
    pid = os.getpid()
    with _lock:
        if _session.initialized and _session.pid != pid:
            # fork 之后父进程的句柄在子进程里不可用，丢弃状态重新初始化
            _session.initialized = False
        if not _session.initialized:
            try:
                module.nvmlInit()
            except Exception as exc:
                raise _translate(exc, "初始化 NVML") from exc
            _session.initialized = True
            _session.pid = pid
            _register_atexit()
        return module


def shutdown() -> None:
    """释放 NVML 资源；重复调用安全（进程退出时自动执行）。"""
    with _lock:
        if not _session.initialized or _session.module is None:
            return
        try:
            _session.module.nvmlShutdown()
        except Exception:  # noqa: BLE001, S110 - 退出阶段不再抛错
            pass
        finally:
            _session.initialized = False


def is_initialized() -> bool:
    return _session.initialized


# ---------------------------------------------------------------- 异常
def _translate(exc: Exception, context: str) -> GPUError:
    """把 ``NVMLError`` 翻译成领域异常。"""
    name = type(exc).__name__
    if name in _UNAVAILABLE:
        return GPUUnavailableError(f"{context}: {exc}")
    if name in _LOST:
        return GPULostError(context)
    if name in _NOT_FOUND:
        return GPUNotFoundError(context, str(exc))
    if name in _UNSUPPORTED:
        return GPUNotSupportedError(context, str(exc))
    return GPUError(f"{context} 失败: {exc}")


def query[T](func: Callable[[], T], context: str) -> T:
    """必须成功的调用：失败即抛领域异常。"""
    try:
        return func()
    except GPUError:
        raise
    except Exception as exc:
        raise _translate(exc, context) from exc


def optional[T](
    func: Callable[[], T], default: T | None = None, *, feature: str = ""
) -> T | None:
    """可缺失指标的调用：``NotSupported`` / 权限不足 / 不存在时返回 ``default``。

    这是整个模块"在消费级卡上也不炸"的关键：风扇转速、ECC、MIG、
    持久化模式等在很多设备上根本不存在，读到就赚到，读不到给 ``None``。
    """
    try:
        return func()
    except GPUError as exc:
        if isinstance(exc, (GPUNotSupportedError, GPUNotFoundError)):
            return default
        raise
    except Exception as exc:
        translated = _translate(exc, feature or "读取指标")
        if isinstance(translated, (GPUNotSupportedError, GPUNotFoundError)):
            return default
        raise translated from exc


def as_text(value: Any) -> str | None:
    """NVML 的字符串在旧版本是 ``bytes``，统一成去空白的 ``str``。"""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace").strip() or None
    return str(value).strip() or None


# ---------------------------------------------------------------- 句柄
def device_count() -> int:
    """GPU 数量。"""
    module = ensure()
    return int(query(module.nvmlDeviceGetCount, "读取 GPU 数量"))


def handle(index: int) -> Any:
    """按序号取设备句柄。"""
    module = ensure()
    if index < 0:
        raise GPUNotFoundError(index, "GPU 序号必须为非负整数")
    return query(lambda: module.nvmlDeviceGetHandleByIndex(index), f"打开 GPU #{index}")


def handle_by_uuid(uuid: str) -> Any:
    """按 UUID 取设备句柄（``GPU-xxxx``，可省略前缀）。"""
    module = ensure()
    candidate = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
    return query(
        lambda: module.nvmlDeviceGetHandleByUUID(candidate), f"按 UUID 打开 {uuid}"
    )


def handle_by_bus(bus_id: str) -> Any:
    """按 PCI 地址取设备句柄（``00000000:01:00.0``）。"""
    module = ensure()
    return query(
        lambda: module.nvmlDeviceGetHandleByPciBusId(bus_id),
        f"按 PCI 地址打开 {bus_id}",
    )
