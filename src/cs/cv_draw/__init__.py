"""图像绘制工具：在图像上绘制点、线、框、多边形、圆、文字、掩码与网格。

两种调用方式：

- 类方法 / 函数式：:class:`Canvas`（链式）或 :func:`render`（一次性）；
- MCP：``python -m cs.cv_draw serve``，工具见 :mod:`cs.cv_draw.mcp_server`。

>>> from cs.cv_draw import Canvas, Style, imread
>>> Canvas(imread("in.jpg"), style=Style(thickness=2)).box(box).save("out.jpg")
"""

from __future__ import annotations

from .core import (
    Canvas,
    ImageInfo,
    image_info,
    imread,
    imread_gray,
    imwrite,
    render,
    resolve_color,
    to_bgr,
)
from .errors import DrawError, ImageError, ShapeError
from .models import Anno, ColorLike, Shape, Stroke, Style, Text, parse_shape

__all__ = [
    "Anno",
    "Canvas",
    "ColorLike",
    "DrawError",
    "ImageError",
    "ImageInfo",
    "Shape",
    "ShapeError",
    "Stroke",
    "Style",
    "Text",
    "image_info",
    "imread",
    "imread_gray",
    "imwrite",
    "parse_shape",
    "render",
    "resolve_color",
    "to_bgr",
]
