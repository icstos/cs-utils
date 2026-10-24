"""``cs.nvidia`` 门面：面向对象的显卡查询接口。

对外只有两个入口::

    from cs.nvidia import NVIDIA

    NVIDIA.driver()              # 驱动 / CUDA 版本
    NVIDIA.list()                # 所有显卡的型号等静态信息
    NVIDIA.status(0)             # 运行状态快照
    NVIDIA.processes(0)          # 占用显卡的进程
    NVIDIA.health(0)             # 健康检查（温度/显存/降频）

    with NVIDIA.open(0) as gpu:  # 句柄式，适合连续操作
        gpu.status()
        for state in gpu.watch(interval=1.0, count=5):
            print(state)

分层
----
``bindings`` 管 NVML 生命周期与异常收敛 → ``readers`` 把原始值翻译成数据类 →
``core``（本模块）做编排与选择器解析 → ``monitor`` 负责采样与呈现。

设计要点
--------
* **无状态**：不长期持有 NVML 指针，每次按 UUID 重新解析，天然线程安全；
* **可缺失指标**：读不到就是 ``None``，绝不 0 值冒充；
* **单位固定**：字节 / 瓦特 / 摄氏度 / 百分比 / MHz。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from . import bindings
from .bindings import as_text, ensure, optional, query
from .errors import (
    GPUError,
    GPUNotFoundError,
    GPUNotSupportedError,
    GPUUnavailableError,
)
from .models import (
    DriverInfo,
    GPUInfo,
    GPUProcess,
    GPUStatus,
    HealthReport,
    MemoryUsage,
    ThrottleInfo,
    TopologyMatrix,
    Utilization,
)
from .readers import (
    diagnose,
    read_info,
    read_memory,
    read_processes,
    read_status,
    read_throttle,
    read_topology,
    read_utilization,
)

if TYPE_CHECKING:  # 仅用于类型标注，避免运行期循环导入
    from .monitor import SampleSeries

__all__ = ["NVIDIA", "GPUHandle", "Selector"]

#: 选择器：序号 / UUID / PCI 地址 / 已解析的句柄 / ``None``（等价 0 号卡）
type Selector = int | str | GPUHandle | None

_ALL_TOKENS = frozenset({"", "all", "*"})


def _resolve(selector: Selector) -> GPUHandle:
    """把序号 / UUID / PCI 地址统一解析成 :class:`GPUHandle`。"""
    if isinstance(selector, GPUHandle):
        return selector

    module = ensure()
    match selector:
        case None:
            index, device = 0, bindings.handle(0)
        case int():
            index, device = selector, bindings.handle(selector)
        case str() if selector.strip().lower() in _ALL_TOKENS:
            raise ValueError("该接口只针对单块显卡，多卡请传具体序号 / UUID / PCI 地址")
        case str() if selector.strip().isdigit():
            index = int(selector.strip())
            device = bindings.handle(index)
        case str() if selector.lower().startswith("pci:") or ":" in selector:
            bus_id = selector[4:] if selector.lower().startswith("pci:") else selector
            device = bindings.handle_by_bus(bus_id.strip())
            index = int(
                query(lambda: module.nvmlDeviceGetIndex(device), "读取 GPU 序号")
            )
        case str():
            device = bindings.handle_by_uuid(selector.strip())
            index = int(
                query(lambda: module.nvmlDeviceGetIndex(device), "读取 GPU 序号")
            )
        case _:
            raise GPUNotFoundError(selector, "不支持的选择器类型")

    uuid = as_text(optional(lambda: module.nvmlDeviceGetUUID(device), feature="UUID"))
    name = as_text(optional(lambda: module.nvmlDeviceGetName(device), feature="型号"))
    return GPUHandle(
        index=index, uuid=uuid or f"index:{index}", name=name or "Unknown GPU"
    )


@dataclass(frozen=True, slots=True)
class GPUHandle:
    """单块 GPU 的操作句柄。

    轻量值对象：内部不缓存 NVML 指针，每次调用都按 UUID 重新解析
    （热插拔或驱动重排序号后依然指向同一块卡）。
    """

    index: int
    uuid: str
    name: str

    # -- 内部：解析出当前有效的 NVML 句柄
    def _device(self) -> Any:
        module = ensure()
        if self.uuid.startswith("GPU-"):
            try:
                return module.nvmlDeviceGetHandleByUUID(self.uuid)
            except Exception:  # noqa: BLE001, S110 - 回退到序号解析
                pass
        return bindings.handle(self.index)

    # -- 只读查询
    def info(self) -> GPUInfo:
        """静态身份信息（型号、架构、显存总量、VBIOS…）。"""
        return read_info(self.index, self._device())

    def status(self) -> GPUStatus:
        """当前运行状态快照。"""
        return read_status(
            self._device(), index=self.index, uuid=self.uuid, name=self.name
        )

    def memory(self) -> MemoryUsage:
        """显存占用；驱动不提供时抛 :class:`GPUNotSupportedError`。"""
        memory = read_memory(self._device())
        if memory is None:
            raise GPUNotSupportedError("显存占用", "设备未提供显存信息")
        return memory

    def utilization(self) -> Utilization:
        """各类引擎占用率。"""
        return read_utilization(self._device())

    def processes(self) -> list[GPUProcess]:
        """占用该卡的进程（compute + graphics + MPS，按 pid 去重）。"""
        return list(read_processes(self._device(), self.index))

    def throttle(self) -> ThrottleInfo:
        """当前降频原因位掩码。"""
        throttle = read_throttle(self._device())
        if throttle is None:
            raise GPUNotSupportedError("降频原因")
        return throttle

    def health(self) -> HealthReport:
        """温度 / 显存 / 降频综合体检。"""
        status = self.status()
        issues = diagnose(status)
        return HealthReport(
            index=self.index,
            uuid=self.uuid,
            name=self.name,
            level=HealthReport.worst(issues),
            issues=issues,
            status=status,
        )

    # -- 采样
    def watch(
        self,
        *,
        interval: float = 1.0,
        count: int | None = None,
        duration: float | None = None,
    ) -> Iterator[GPUStatus]:
        """按间隔持续产出状态快照。

        ``count`` 与 ``duration`` 至少给一个（避免调用方写出死循环），
        两者都给时以先到者为准。
        """
        from .monitor import iter_status

        return iter_status(self, interval=interval, count=count, duration=duration)

    def series(self, *, interval: float = 1.0, count: int = 10) -> SampleSeries:
        """采样固定次数并汇总统计（见 :class:`cs.nvidia.monitor.SampleSeries`）。"""
        from .monitor import sample_status

        return sample_status(self, interval=interval, count=count)

    def raw(self) -> Any:
        """逃生通道：拿到 ``pynvml`` 的原始设备指针。"""
        return self._device()

    def __enter__(self) -> Self:
        ensure()
        return self

    def __exit__(self, *exc_info: object) -> None:
        # 句柄本身不持有资源，NVML 由进程级 atexit 统一释放
        return None

    def __str__(self) -> str:
        return f"[{self.index}] {self.name} ({self.uuid})"


class NVIDIA:
    """整机级别的 GPU 查询入口（全部为静态方法，无状态、可并发）。"""

    __slots__ = ()

    # ---------------------------------------------------------- 生命周期
    @staticmethod
    def available() -> bool:
        """NVML 是否可用且有显卡（不抛异常，适合探测）。"""
        try:
            return bindings.device_count() > 0
        except GPUError:
            return False

    @staticmethod
    def init() -> DriverInfo:
        """显式初始化并返回驱动信息；失败抛 :class:`GPUUnavailableError`。"""
        return NVIDIA.driver()

    @staticmethod
    def shutdown() -> None:
        """释放 NVML（进程退出时也会自动调用）。"""
        bindings.shutdown()

    # ---------------------------------------------------------- 整机信息
    @staticmethod
    def driver() -> DriverInfo:
        """驱动版本、CUDA 版本、NVML 版本、GPU 数量。"""
        module = ensure()
        return DriverInfo(
            driver_version=as_text(
                query(module.nvmlSystemGetDriverVersion, "读取驱动版本")
            )
            or "unknown",
            nvml_version=as_text(
                optional(module.nvmlSystemGetNVMLVersion, feature="NVML 版本")
            )
            or "unknown",
            cuda_version=optional(
                module.nvmlSystemGetCudaDriverVersion, feature="CUDA 版本"
            ),
            gpu_count=bindings.device_count(),
        )

    @staticmethod
    def count() -> int:
        """GPU 数量。"""
        return bindings.device_count()

    # ---------------------------------------------------------- 枚举
    @staticmethod
    def list() -> list[GPUInfo]:
        """所有显卡的静态信息。"""
        module = ensure()
        return [
            read_info(index, device)
            for index, device in enumerate(NVIDIA._devices(module))
        ]

    @staticmethod
    def info(selector: Selector = 0) -> GPUInfo:
        """单卡静态信息。"""
        return _resolve(selector).info()

    @staticmethod
    def gpus() -> list[GPUHandle]:
        """所有显卡的句柄。"""
        module = ensure()
        handles: list[GPUHandle] = []
        for index, device in enumerate(NVIDIA._devices(module)):
            uuid = as_text(
                optional(
                    lambda device=device: module.nvmlDeviceGetUUID(device),
                    feature="UUID",
                )
            )
            name = as_text(
                optional(
                    lambda device=device: module.nvmlDeviceGetName(device),
                    feature="型号",
                )
            )
            handles.append(
                GPUHandle(
                    index=index,
                    uuid=uuid or f"index:{index}",
                    name=name or "Unknown GPU",
                )
            )
        return handles

    @staticmethod
    def gpu(selector: Selector = 0) -> GPUHandle:
        """按序号 / UUID / PCI 地址取句柄。"""
        return _resolve(selector)

    #: ``NVIDIA.open(0)`` 的别名，语义上更贴近"打开一块卡"
    open = gpu

    # ---------------------------------------------------------- 状态
    @staticmethod
    def status(selector: Selector = 0) -> GPUStatus:
        """单卡运行状态。"""
        return _resolve(selector).status()

    @staticmethod
    def statuses() -> list[GPUStatus]:
        """所有卡的运行状态（逐卡采集，时间戳各自记录）。"""
        return [handle.status() for handle in NVIDIA.gpus()]

    @staticmethod
    def processes(selector: Selector = 0) -> list[GPUProcess]:
        """单卡上的进程。"""
        return _resolve(selector).processes()

    @staticmethod
    def all_processes() -> list[GPUProcess]:
        """所有卡上的进程（``gpu_index`` 标明归属）。"""
        result: list[GPUProcess] = []
        for handle in NVIDIA.gpus():
            result.extend(handle.processes())
        return result

    @staticmethod
    def throttle(selector: Selector = 0) -> ThrottleInfo:
        """单卡降频原因。"""
        return _resolve(selector).throttle()

    @staticmethod
    def health(selector: Selector = 0) -> HealthReport:
        """单卡体检。"""
        return _resolve(selector).health()

    @staticmethod
    def health_all() -> list[HealthReport]:
        """所有卡体检。"""
        return [handle.health() for handle in NVIDIA.gpus()]

    @staticmethod
    def topologies() -> TopologyMatrix:
        """多卡互联拓扑矩阵。"""
        handles = NVIDIA.gpus()
        devices = [handle.raw() for handle in handles]
        return read_topology(
            devices,
            [handle.uuid for handle in handles],
            [handle.name for handle in handles],
        )

    # ---------------------------------------------------------- 采样
    @staticmethod
    def watch(
        selector: Selector = 0,
        *,
        interval: float = 1.0,
        count: int | None = None,
        duration: float | None = None,
    ) -> Iterator[GPUStatus]:
        """持续采样（生成器）。``count`` / ``duration`` 至少给一个。"""
        return _resolve(selector).watch(
            interval=interval, count=count, duration=duration
        )

    @staticmethod
    def sample(
        selector: Selector = 0, *, interval: float = 1.0, count: int = 10
    ) -> SampleSeries:
        """采样固定次数并返回带统计的序列。"""
        return _resolve(selector).series(interval=interval, count=count)

    # ---------------------------------------------------------- 呈现
    @staticmethod
    def text(selector: Selector | None = None) -> str:
        """``nvidia-smi`` 风格的纯文本表格；``selector=None`` 渲染全部显卡。"""
        from .monitor import render_table

        if selector is None:
            return render_table(NVIDIA.statuses())
        return render_table([NVIDIA.status(selector)])

    # ---------------------------------------------------------- 内部
    @staticmethod
    def _devices(module: Any) -> list[Any]:
        count = bindings.device_count()
        if count == 0:
            raise GPUUnavailableError("没有检测到任何 NVIDIA GPU")
        return [module.nvmlDeviceGetHandleByIndex(index) for index in range(count)]
