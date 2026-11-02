"""把 :mod:`cs.cv_draw` 暴露为 MCP 工具（FastMCP ≥ 4）。

启动方式::

    python -m cs.cv_draw serve                 # stdio，供 MCP 客户端拉起
    python -m cs.cv_draw serve --transport http --port 8000
    fastmcp run src/cs/cv_draw/mcp_server.py:mcp

约定：工具名统一 ``cv_draw_*``；成功返回 ``{"ok": true, ...}``，
业务异常统一转换成 ``ToolError``，避免把 traceback 泄露给模型。
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from ..cv import Point
from .core import Canvas, image_info, imread, imread_gray
from .errors import DrawError
from .models import Style, parse_shape

__all__ = ["create_server", "mcp"]

_SERVER_NAME = "cs-cv-draw"
_INSTRUCTIONS = """\
本服务器在本地图像上绘制常见计算机视觉图元并另存为新文件：点、线段、矩形框、
多边形、圆、文字，以及掩码着色与辅助网格。支持实线/虚线/点线、半透明填充、
自动配色与中文文字。

使用建议：
1. 先调用 cv_draw_info 确认图像尺寸，坐标以左上角为原点、单位为像素；
2. 图元用 JSON 描述，必填 kind（point/line/box/polygon/circle/text），
   label 为自定义文字，category_idx 决定调色板取色（0 起）；
3. 所有工具都写入新文件，默认不覆盖已存在文件（需显式 overwrite=true）；
4. 建议传绝对路径（相对路径按服务进程的工作目录解析），输出目录会自动创建。
"""


def _guard[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """把领域异常转换成 MCP 可读错误。"""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except (DrawError, OSError, ValueError) as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


def _done(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def create_server(
    name: str = _SERVER_NAME, *, instructions: str = _INSTRUCTIONS
) -> FastMCP:
    """构造一个独立的服务器实例（便于测试或多实例部署）。"""
    server: FastMCP = FastMCP(name, instructions=instructions)

    # ---------------------------------------------------------- 只读
    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def cv_draw_info(source: str) -> dict[str, Any]:
        """读取图像基本信息：宽高、通道数、dtype、是否灰度（不修改文件）。"""
        return _done(**image_info(source).to_dict())

    # ---------------------------------------------------------- 绘制
    @server.tool()
    @_guard
    def cv_draw_shapes(
        source: str,
        output: str,
        shapes: list[dict[str, Any]],
        style: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """在图像上绘制一批图元并另存为新文件，返回路径与图元数量。

        shapes 每项的 kind 与字段：
        - point:   {"x": 10, "y": 20}
        - line:    {"points": [[x1, y1], [x2, y2]]}
        - box:     {"x1": 10, "y1": 20, "x2": 110, "y2": 90}
        - polygon: {"points": [[x1, y1], [x2, y2], [x3, y3], ...]}
        - circle:  {"cx": 50, "cy": 50, "r": 20}
        - text:    {"text": "缺陷", "x": 10, "y": 30, "size": 18}
        公共可选字段：label（自定义标签文字）、conf（置信度）、category_idx（取色）。
        style 可选字段：color、thickness、stroke(solid/dashed/dotted)、fill、alpha、
        label、label_bg、font_size。
        """
        canvas = Canvas(imread(source), style=Style.from_dict(style))
        annos = [parse_shape(item) for item in shapes]
        canvas.draw_all(annos)
        path = canvas.save(output, overwrite=overwrite)
        return _done(
            output=str(path),
            count=len(annos),
            width=canvas.width,
            height=canvas.height,
        )

    @server.tool()
    @_guard
    def cv_draw_text(
        source: str,
        output: str,
        text: str,
        x: float,
        y: float,
        size: int = 0,
        color: str | list[int] | int | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """在 (x, y)（文字左上角）处写一行文字；中文会自动使用系统字体。"""
        canvas = Canvas(imread(source))
        canvas.text(text, Point(x, y), size=size, style=Style(color=color))
        path = canvas.save(output, overwrite=overwrite)
        return _done(output=str(path))

    @server.tool()
    @_guard
    def cv_draw_mask(
        source: str,
        output: str,
        mask: str,
        color: str | list[int] | int | None = None,
        alpha: float = 0.35,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """把掩码图（非零像素）以半透明颜色叠加到原图，用于分割/热区可视化。"""
        canvas = Canvas(imread(source))
        canvas.mask(imread_gray(mask), color=color, alpha=alpha)
        path = canvas.save(output, overwrite=overwrite)
        return _done(output=str(path), alpha=alpha)

    @server.tool()
    @_guard
    def cv_draw_grid(
        source: str,
        output: str,
        cols: int = 10,
        rows: int = 10,
        color: str | list[int] | int | None = None,
        thickness: int = 1,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """绘制等距辅助网格，便于人工核对坐标。"""
        canvas = Canvas(imread(source))
        canvas.grid(cols, rows, style=Style(color=color, thickness=thickness))
        path = canvas.save(output, overwrite=overwrite)
        return _done(output=str(path), cols=cols, rows=rows)

    return server


mcp = create_server()
