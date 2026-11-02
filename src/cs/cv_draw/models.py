"""绘制相关的数据模型：样式、文字图元、JSON 规格解析。

本模块只描述“要画什么”，不碰像素；真正的绘制在 :mod:`cs.cv_draw.core`。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Any

from ..cv import Box, Circle, Line, Point, Polygon
from .errors import ShapeError

__all__ = [
    "Anno",
    "ColorLike",
    "Shape",
    "Stroke",
    "Style",
    "Text",
    "parse_shape",
]

type ColorLike = tuple[int, int, int] | list[int] | str | int
"""颜色写法：RGB 三元组、``#rrggbb`` 字符串，或调色板下标。"""


class Stroke(StrEnum):
    """线型。"""

    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"


@dataclass(slots=True, frozen=True)
class Style:
    """一次绘制使用的样式。

    字段取 ``None`` / ``0`` 时表示“自适应”：线宽按图像尺寸推算，
    颜色按图元的 ``category_idx`` 从调色板取，字号按线宽推算。
    """

    color: ColorLike | None = None
    thickness: int | None = None
    stroke: Stroke = Stroke.SOLID
    fill: bool = False
    alpha: float = 0.25
    label: bool = True
    label_bg: bool = False
    font_size: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None) -> Style:
        """从 MCP/CLI 的 JSON 对象构造样式，未知字段直接报错。"""
        payload = dict(data or {})
        if unknown := set(payload) - {item.name for item in fields(cls)}:
            raise ShapeError(f"未知样式字段：{', '.join(sorted(unknown))}")
        if (stroke := payload.get("stroke")) is not None:
            try:
                payload["stroke"] = Stroke(stroke)
            except ValueError:
                raise ShapeError(
                    f"stroke 只能是 {[item.value for item in Stroke]}"
                ) from None
        if isinstance(color := payload.get("color"), list):
            if len(color) != 3:
                raise ShapeError("color 列表需为 [r, g, b]")
            payload["color"] = tuple(color)
        if (alpha := payload.get("alpha")) is not None:
            if not 0.0 <= float(alpha) <= 1.0:
                raise ShapeError("alpha 需在 0~1 之间")
            payload["alpha"] = float(alpha)
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        data = {item.name: getattr(self, item.name) for item in fields(self)}
        data["stroke"] = self.stroke.value
        return data


@dataclass(slots=True)
class Text:
    """文字图元，``point`` 为文字左上角（像素）。"""

    text: str
    point: Point = field(default_factory=Point)
    size: int = 0


@dataclass(slots=True)
class Anno:
    """图元 + 标签文字；标签优先于“类别名: 置信度”的自动生成结果。"""

    shape: Shape
    label: str = ""


type Shape = Point | Line | Box | Polygon | Circle | Text | Anno
"""可绘制的一切：几何图元、文字，以及带标签的包装。"""


# ---------------------------------------------------------------- 规格解析


def parse_shape(spec: Mapping[str, Any]) -> Anno:
    """把 JSON 规格转成 :class:`Anno`。

    必填 ``kind``（point/line/box/polygon/circle/text，缺省时按字段推断），
    可选 ``label``、``conf``、``category_idx``（决定调色板取色）。
    """
    if not isinstance(spec, Mapping):
        raise ShapeError(f"图元规格必须是对象，收到 {type(spec).__name__}")
    kind = str(spec.get("kind") or _guess_kind(spec)).lower()
    conf = _num(spec, "conf", default=-1.0)
    category_idx = int(_num(spec, "category_idx", default=0))

    match kind:
        case "point":
            shape = Point(
                x=_num(spec, "x"),
                y=_num(spec, "y"),
                category_idx=category_idx,
                conf=conf,
            )
        case "line":
            start, end = _points(spec, minimum=2)[:2]
            shape = Line(start=start, end=end, category_idx=category_idx, conf=conf)
        case "box":
            x1, y1, x2, y2 = (_num(spec, key) for key in ("x1", "y1", "x2", "y2"))
            shape = Box(
                start=Point(x1, y1),
                end=Point(x2, y2),
                category_idx=category_idx,
                conf=conf,
            )
        case "polygon":
            shape = Polygon(
                points=_points(spec, minimum=3), category_idx=category_idx, conf=conf
            )
        case "circle":
            cx, cy, radius = (_num(spec, key) for key in ("cx", "cy", "r"))
            if radius <= 0:
                raise ShapeError("圆半径 r 必须为正数")
            shape = Circle(
                center=Point(cx, cy),
                radius=radius,
                category_idx=category_idx,
                conf=conf,
            )
        case "text":
            content = str(spec.get("text", "")).strip()
            if not content:
                raise ShapeError("文字图元的 text 不能为空")
            shape = Text(
                text=content,
                point=Point(_num(spec, "x"), _num(spec, "y")),
                size=int(_num(spec, "size", default=0)),
            )
        case _:
            raise ShapeError(f"未知图元类型：{kind}")

    return Anno(shape=shape, label=str(spec.get("label", "")))


def _guess_kind(spec: Mapping[str, Any]) -> str:
    """字段齐全时省略 ``kind`` 也能识别；两点默认按线段、三点及以上按多边形。"""
    keys = set(spec)
    if {"x1", "y1", "x2", "y2"} <= keys:
        return "box"
    if {"cx", "cy", "r"} <= keys:
        return "circle"
    if isinstance(points := spec.get("points"), (list, tuple)):
        return "line" if len(points) == 2 else "polygon"
    if "text" in keys:
        return "text"
    if {"x", "y"} <= keys:
        return "point"
    raise ShapeError("无法从字段推断图元类型，请显式提供 kind")


def _num(spec: Mapping[str, Any], key: str, *, default: float | None = None) -> float:
    if (value := spec.get(key, default)) is None:
        raise ShapeError(f"缺少字段 {key}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ShapeError(f"字段 {key} 需为数值，收到 {value!r}") from None


def _points(spec: Mapping[str, Any], *, minimum: int) -> list[Point]:
    raw = spec.get("points")
    if not isinstance(raw, (list, tuple)) or len(raw) < minimum:
        raise ShapeError(f"points 至少需要 {minimum} 个 [x, y] 点")
    if not all(isinstance(item, (list, tuple)) and len(item) >= 2 for item in raw):
        raise ShapeError("points 需为 [[x, y], ...] 形式")
    return [Point(float(item[0]), float(item[1])) for item in raw]
