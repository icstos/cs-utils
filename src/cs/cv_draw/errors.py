"""绘制模块的异常层级。

与 :mod:`cs.pdf` 一致：领域异常统一继承 :class:`DrawError`，
MCP 层再把它们转换成 ``ToolError``，避免 traceback 泄漏给模型。
"""

from __future__ import annotations

__all__ = ["DrawError", "ImageError", "ShapeError"]


class DrawError(Exception):
    """绘制相关错误的基类。"""


class ImageError(DrawError):
    """图像读写、解码或格式问题。"""


class ShapeError(DrawError):
    """图元规格或几何参数不合法。"""
