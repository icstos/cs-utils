"""``cs.nvidia`` 的异常体系。

所有异常都继承 :class:`GPUError`，调用方既可以精确捕获子类，
也可以统一用 ``except GPUError`` 兜底；同时尽量复用内建异常语义
（``RuntimeError`` / ``LookupError`` / ``ImportError`` / ``NotImplementedError``），
让不熟悉本模块的调用方也能凭直觉处理。
"""

from __future__ import annotations

__all__ = [
    "GPUError",
    "GPULostError",
    "GPUNotFoundError",
    "GPUNotSupportedError",
    "GPUUnavailableError",
    "MissingDependencyError",
]


class GPUError(Exception):
    """GPU 操作异常基类。"""


class MissingDependencyError(GPUError, ImportError):
    """缺少 ``nvidia-ml-py``（提供 ``pynvml`` 命名空间）。"""

    def __init__(self, package: str = "nvidia-ml-py", *, extra: str = "gpu") -> None:
        self.package = package
        self.extra = extra
        super().__init__(
            f"读取 NVIDIA 显卡信息需要 {package}，请安装：pip install {package}"
        )


class GPUUnavailableError(GPUError, RuntimeError):
    """NVML 不可用：未安装驱动、找不到 NVML 动态库、或初始化失败。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(
            f"NVML 不可用（{reason}）。请确认已安装 NVIDIA 驱动，"
            "或在非 NVIDIA 机器上改用其它方案"
        )


class GPUNotFoundError(GPUError, LookupError):
    """按序号 / UUID / PCI 地址找不到对应 GPU。"""

    def __init__(self, selector: object, reason: str = "未找到匹配的 GPU") -> None:
        self.selector = selector
        self.reason = reason
        super().__init__(f"{selector}：{reason}")


class GPUNotSupportedError(GPUError, NotImplementedError):
    """当前 GPU / 驱动 / 平台不支持该指标（如笔记本上的风扇转速、ECC）。"""

    def __init__(self, feature: str, reason: str = "当前设备不支持") -> None:
        self.feature = feature
        super().__init__(f"{feature}：{reason}")


class GPULostError(GPUError, RuntimeError):
    """GPU 已掉线或被驱动移除（``NVML_ERROR_GPU_IS_LOST``）。"""

    def __init__(self, selector: object) -> None:
        self.selector = selector
        super().__init__(f"GPU 已失去连接: {selector!r}（请重新扫描设备）")
