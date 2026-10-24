"""``cs.pdf`` 的异常体系。

所有异常都继承 :class:`PDFError`，调用方既可以精确捕获子类，
也可以统一用 ``except PDFError`` 兜底；同时尽量复用内建异常语义
（``ValueError`` / ``FileExistsError`` / ``ImportError``），
让不熟悉本模块的调用方也能凭直觉处理。
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "MissingDependencyError",
    "OutputExistsError",
    "PDFError",
    "PDFOpenError",
    "PDFPasswordError",
    "PageSpecError",
]


class PDFError(Exception):
    """PDF 操作异常基类。"""


class PDFOpenError(PDFError, OSError):
    """无法打开或解析 PDF 文件。"""

    def __init__(self, path: Path | str, reason: str) -> None:
        self.path = Path(path)
        self.reason = reason
        super().__init__(f"无法打开 PDF {self.path}: {reason}")


class PDFPasswordError(PDFError):
    """PDF 已加密，且未提供密码或密码错误。"""

    def __init__(self, path: Path | str, reason: str = "需要密码") -> None:
        self.path = Path(path)
        super().__init__(f"{self.path}: {reason}")


class PageSpecError(PDFError, ValueError):
    """页码范围表达式非法，或超出文档实际页数。"""


class OutputExistsError(PDFError, FileExistsError):
    """目标文件已存在，且调用方未允许覆盖。"""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        super().__init__(
            f"目标文件已存在: {self.path}（如需覆盖请设置 overwrite=True）"
        )


class MissingDependencyError(PDFError, ImportError):
    """缺少可选依赖。"""

    def __init__(self, package: str, *, extra: str, purpose: str) -> None:
        self.package = package
        self.extra = extra
        self.purpose = purpose
        super().__init__(
            f"{purpose} 需要额外依赖 {package!r}，请安装：pip install 'cs-utils[{extra}]'"
        )
