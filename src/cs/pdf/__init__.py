"""``cs.pdf`` —— PDF 常用操作 + FastMCP 服务。

对外只暴露一个门面 :class:`PDF`（以及只读句柄 :class:`PDFHandle`）::

    from cs.pdf import PDF

    PDF.merge(["a.pdf", "b.pdf"], "merged.pdf")

    with PDF.open("a.pdf") as doc:
        doc.info().page_count

MCP 相关能力按需加载，未安装 ``fastmcp`` 时仍可正常使用类方法::

    from cs.pdf import create_server
    create_server().run()
"""

from __future__ import annotations

from typing import Any

from .core import PDF, ImageFormat, PDFHandle
from .errors import (
    MissingDependencyError,
    OutputExistsError,
    PageSpecError,
    PDFError,
    PDFOpenError,
    PDFPasswordError,
)
from .models import PageInfo, PDFInfo
from .pages import PageSelection, PageSpec

__all__ = [
    "PDF",
    "ImageFormat",
    "MissingDependencyError",
    "OutputExistsError",
    "PDFError",
    "PDFHandle",
    "PDFInfo",
    "PDFOpenError",
    "PDFPasswordError",
    "PageInfo",
    "PageSelection",
    "PageSpec",
    "PageSpecError",
    "create_server",
]


def create_server(*args: Any, **kwargs: Any) -> Any:
    """创建 FastMCP 服务器实例（惰性导入 ``fastmcp``）。"""
    from .mcp_server import create_server as _create_server

    return _create_server(*args, **kwargs)


def __getattr__(name: str) -> Any:
    """``cs.pdf.mcp`` 惰性暴露，避免可选依赖在 import 期被拉起。"""
    if name == "mcp":
        from .mcp_server import mcp as server

        return server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
