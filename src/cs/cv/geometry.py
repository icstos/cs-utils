"""几何图元：点、线段、矩形框、旋转框、多边形、圆、姿态关键点。

设计约定：

- 全部是 ``frozen`` dataclass —— 变换（``offset`` / ``scale`` / ``simplify``）返回**新**
  对象，原对象不变，因此可以安全地在多处共享（例如 ``Box`` 复用内外两层 ``Point``）；
- 可哈希、可比较，能直接当 dict/set 的键；
- 派生量（外接框、Shapely 对象、圆周采样点）一律 ``cached_property``，
  对象不可变，缓存永不失效，不需要手工清理；
- 坐标系与图像一致：原点在左上角，y 轴向下，单位为像素。

每个图元都支持 YOLO 文本行的双向转换（``to_yolo`` / ``from_yolo``）：

===========  ==========================================
图元         格式（数值均按图像宽高归一化）
===========  ==========================================
Point        ``cls x y [conf]``
Line         ``cls x1 y1 x2 y2 [conf]``
Box          ``cls cx cy w h [conf]``
Obb          ``cls x1 y1 x2 y2 x3 y3 x4 y4 [conf]``
Polygon      ``cls x1 y1 x2 y2 ... [conf]``
Circle       ``cls cx cy r [conf]``（半径按短边归一化）
Pose         ``cls x1 y1 v1 x2 y2 v2 ...``
===========  ==========================================
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import cached_property
from itertools import pairwise
from typing import Any, Self

import shapely

from ..my_types import ImgSize

__all__ = [
    "CIRCLE_SAMPLES",
    "POINT_RADIUS",
    "Bounds",
    "Box",
    "Circle",
    "Geometry",
    "Line",
    "Obb",
    "Point",
    "Polygon",
    "Pose",
]

#: 绘制点时使用的默认半径（也参与 ``Point`` 的外接框计算）
POINT_RADIUS = 4

#: 圆周采样点数：取 4 的倍数以保证包含上下左右四个极点，
#: 这样采样点的外接框与圆的真实外接框完全一致。
CIRCLE_SAMPLES = 64


def _num(value: float) -> str:
    """紧凑的浮点写法：``12.5`` 而不是 ``12.500000``。"""
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


# ============================================================
# 公共基类
# ============================================================


class Bounds:
    """基于 ``points`` 的外接框与通用几何操作（mixin，不产生字段）。"""

    __slots__ = ()

    points: Sequence[Point]
    obj: shapely.Geometry
    category_idx: int
    conf: float

    @cached_property
    def x_min(self) -> float:
        return min(p.x for p in self.points)

    @cached_property
    def y_min(self) -> float:
        return min(p.y for p in self.points)

    @cached_property
    def x_max(self) -> float:
        return max(p.x for p in self.points)

    @cached_property
    def y_max(self) -> float:
        return max(p.y for p in self.points)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def x_center(self) -> float:
        return (self.x_min + self.x_max) / 2

    @property
    def y_center(self) -> float:
        return (self.y_min + self.y_max) / 2

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    # ---------------------------------------------------------- 校验与裁剪
    def validate(self, img_size: ImgSize) -> bool:
        """外接框是否完全落在图像内。"""
        return not (
            self.x_min < 0
            or self.y_min < 0
            or self.x_max > img_size.width
            or self.y_max > img_size.height
        )

    def clamp(self, img_size: ImgSize) -> Self:
        """平移回图像范围内（不改变尺寸）。"""
        dx = -self.x_min if self.x_min < 0 else min(0.0, img_size.width - self.x_max)
        dy = -self.y_min if self.y_min < 0 else min(0.0, img_size.height - self.y_max)
        return self.offset(dx, dy) if dx or dy else self

    # ---------------------------------------------------------- 几何运算
    def distance(self, other: Bounds) -> float:
        """到另一个图元的最短距离（相交或内含时为 0）。"""
        return shapely.distance(self.obj, other.obj)

    def to_box(self) -> Box:
        """转成轴对齐外接框。"""
        return Box(
            Point(self.x_min, self.y_min),
            Point(self.x_max, self.y_max),
            category_idx=self.category_idx,
            conf=self.conf,
        )

    # 由子类实现：变换、序列化
    def offset(self, delta_x: float = 0.0, delta_y: float = 0.0) -> Self:
        raise NotImplementedError

    def scale(self, scale_x: float = 1.0, scale_y: float = 1.0) -> Self:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


# ============================================================
# 点
# ============================================================


@dataclass(frozen=True)
class Point(Bounds):
    """二维点。"""

    x: float = -1.0
    y: float = -1.0
    category_idx: int = 0
    conf: float = -1.0
    create_time: datetime = field(
        default_factory=datetime.now, compare=False, repr=False
    )

    # --- 以点半径为边长的外接框（比逐点求 min/max 快） ---
    @property
    def x_min(self) -> float:
        return self.x - POINT_RADIUS

    @property
    def y_min(self) -> float:
        return self.y - POINT_RADIUS

    @property
    def x_max(self) -> float:
        return self.x + POINT_RADIUS

    @property
    def y_max(self) -> float:
        return self.y + POINT_RADIUS

    @property
    def width(self) -> float:
        return 2 * POINT_RADIUS

    @property
    def height(self) -> float:
        return 2 * POINT_RADIUS

    # --- 常用快捷属性 ---
    @property
    def x_int(self) -> int:
        return int(self.x)

    @property
    def y_int(self) -> int:
        return int(self.y)

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def xy_int(self) -> tuple[int, int]:
        return (self.x_int, self.y_int)

    @cached_property
    def obj(self) -> shapely.Point:
        return shapely.Point(self.x, self.y)

    @cached_property
    def box(self) -> Box:
        return Box(Point(self.x_min, self.y_min), Point(self.x_max, self.y_max))

    def __repr__(self) -> str:
        return f"({_num(self.x)}, {_num(self.y)})"

    # --- 变换 ---
    def offset(self, delta_x: float = 0.0, delta_y: float = 0.0) -> Self:
        return replace(self, x=self.x + delta_x, y=self.y + delta_y)

    def scale(self, scale_x: float = 1.0, scale_y: float = 1.0) -> Self:
        return replace(self, x=self.x * scale_x, y=self.y * scale_y)

    # --- 判断 ---
    def is_in(self, other: Bounds) -> bool:
        """是否落在 ``other`` 的外接框内。"""
        return (
            other.x_min <= self.x <= other.x_max
            and other.y_min <= self.y <= other.y_max
        )

    def to_yolo(self, img_size: ImgSize, *, with_conf: bool = False) -> str:
        tokens = [
            str(self.category_idx),
            _num(self.x / img_size.width),
            _num(self.y / img_size.height),
        ]
        if with_conf:
            tokens.append(_num(self.conf))
        return " ".join(tokens)

    @classmethod
    def from_yolo(cls, line: str, img_size: ImgSize) -> Self:
        values = [float(v) for v in line.split()]
        return cls(
            x=values[1] * img_size.width,
            y=values[2] * img_size.height,
            category_idx=int(values[0]),
            conf=values[3] if len(values) > 3 else -1.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "point",
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "cat": self.category_idx,
            "conf": round(self.conf, 4),
        }

    def to_sql_model(self, img: Any, session: Any) -> Any:
        """转为分类标注实体。"""
        from ..entity_model import TCategory, TClassifyAnno, select

        category = session.exec(
            select(TCategory).where(TCategory.idx == self.category_idx)
        ).one()
        return TClassifyAnno(conf=self.conf, img=img, category=category)


# ============================================================
# 线段
# ============================================================


@dataclass(frozen=True)
class Line(Bounds):
    """线段，端点自动规范为“左上 → 右下”。

    坐标使用图像坐标系（y 轴向下），因此正角度表示向右下方倾斜。
    """

    start: Point = field(default_factory=Point)
    end: Point = field(default_factory=Point)
    category_idx: int = 0
    conf: float = -1.0
    create_time: datetime = field(
        default_factory=datetime.now, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if (self.start.x, self.start.y) > (self.end.x, self.end.y):
            start, end = self.end, self.start
            object.__setattr__(self, "start", start)
            object.__setattr__(self, "end", end)

    # --- 顶点 ---
    @property
    def points(self) -> tuple[Point, Point]:
        return (self.start, self.end)

    # --- 几何量 ---
    @property
    def delta_x(self) -> float:
        return self.end.x - self.start.x

    @property
    def delta_y(self) -> float:
        return self.end.y - self.start.y

    @property
    def length(self) -> float:
        return math.hypot(self.delta_x, self.delta_y)

    @property
    def angle(self) -> float:
        """与 x 轴正方向的夹角（度），取值落在 [-90, 90]。"""
        return math.degrees(math.atan2(self.delta_y, self.delta_x))

    @property
    def slope(self) -> float:
        """斜率 dy/dx；竖直线为 ``inf``，水平线为 0。"""
        return self.delta_y / self.delta_x if self.delta_x else float("inf")

    @property
    def center(self) -> Point:
        """线段中点。"""
        return Point(self.x_center, self.y_center)

    @property
    def area(self) -> float:
        """线段没有面积，恒为 0（与 Box/Polygon 保持同一接口）。"""
        return 0.0

    @property
    def is_horizontal(self) -> bool:
        return math.isclose(self.delta_y, 0.0, abs_tol=1e-9)

    @property
    def is_vertical(self) -> bool:
        return math.isclose(self.delta_x, 0.0, abs_tol=1e-9)

    @cached_property
    def obj(self) -> shapely.LineString:
        return shapely.LineString([self.start.xy, self.end.xy])

    def __repr__(self) -> str:
        return f"Line({self.start!r} -> {self.end!r}, conf={self.conf:.2f}, cat={self.category_idx})"

    # --- 变换 ---
    def offset(self, delta_x: float = 0.0, delta_y: float = 0.0) -> Self:
        return replace(
            self,
            start=self.start.offset(delta_x, delta_y),
            end=self.end.offset(delta_x, delta_y),
        )

    def scale(self, scale_x: float = 1.0, scale_y: float = 1.0) -> Self:
        return replace(
            self,
            start=self.start.scale(scale_x, scale_y),
            end=self.end.scale(scale_x, scale_y),
        )

    # --- 判断 ---
    def is_in(self, other: Bounds) -> bool:
        """整条线段是否落在 ``other`` 的外接框内。"""
        return (
            other.x_min <= self.x_min
            and self.x_max <= other.x_max
            and other.y_min <= self.y_min
            and self.y_max <= other.y_max
        )

    # --- 序列化 ---
    def to_yolo(self, img_size: ImgSize, *, with_conf: bool = False) -> str:
        tokens = [
            str(self.category_idx),
            _num(self.start.x / img_size.width),
            _num(self.start.y / img_size.height),
            _num(self.end.x / img_size.width),
            _num(self.end.y / img_size.height),
        ]
        if with_conf:
            tokens.append(_num(self.conf))
        return " ".join(tokens)

    @classmethod
    def from_yolo(cls, line: str, img_size: ImgSize) -> Self:
        values = [float(v) for v in line.split()]
        return cls(
            start=Point(values[1] * img_size.width, values[2] * img_size.height),
            end=Point(values[3] * img_size.width, values[4] * img_size.height),
            category_idx=int(values[0]),
            conf=values[5] if len(values) > 5 else -1.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "line",
            "x1": round(self.start.x, 3),
            "y1": round(self.start.y, 3),
            "x2": round(self.end.x, 3),
            "y2": round(self.end.y, 3),
            "cat": self.category_idx,
            "conf": round(self.conf, 4),
        }


# ============================================================
# 矩形框
# ============================================================


@dataclass(frozen=True)
class Box(Bounds):
    """轴对齐矩形框，对角两点自动规范为左上 / 右下。"""

    start: Point = field(default_factory=Point)
    end: Point = field(default_factory=Point)
    category_idx: int = 0
    conf: float = -1.0
    create_time: datetime = field(
        default_factory=datetime.now, compare=False, repr=False
    )
    #: 跟踪 ID（ultralytics 的 ``boxes.id``），非跟踪场景为 -1
    track_id: int = -1

    def __post_init__(self) -> None:
        x1, y1 = min(self.start.x, self.end.x), min(self.start.y, self.end.y)
        x2, y2 = max(self.start.x, self.end.x), max(self.start.y, self.end.y)
        if (x1, y1, x2, y2) != (self.start.x, self.start.y, self.end.x, self.end.y):
            object.__setattr__(self, "start", replace(self.start, x=x1, y=y1))
            object.__setattr__(self, "end", replace(self.end, x=x2, y=y2))

    # --- 边界（直接取端点，避免遍历） ---
    @property
    def x_min(self) -> float:
        return self.start.x

    @property
    def y_min(self) -> float:
        return self.start.y

    @property
    def x_max(self) -> float:
        return self.end.x

    @property
    def y_max(self) -> float:
        return self.end.y

    @property
    def center(self) -> Point:
        """框的中心点。"""
        return Point(self.x_center, self.y_center)

    @property
    def x_min_int(self) -> int:
        return int(self.x_min)

    @property
    def y_min_int(self) -> int:
        return int(self.y_min)

    @property
    def x_max_int(self) -> int:
        return int(self.x_max)

    @property
    def y_max_int(self) -> int:
        return int(self.y_max)

    # --- 面积与形状 ---
    @cached_property
    def area(self) -> float:
        return self.width * self.height

    @property
    def long_axis(self) -> float:
        return max(self.width, self.height)

    @property
    def short_axis(self) -> float:
        return min(self.width, self.height)

    @cached_property
    def axis_ratio(self) -> float:
        return self.long_axis / self.short_axis if self.short_axis else float("inf")

    # --- 顶点与 Shapely 对象 ---
    @cached_property
    def points(self) -> tuple[Point, Point, Point, Point]:
        return (
            Point(self.x_min, self.y_min),
            Point(self.x_max, self.y_min),
            Point(self.x_max, self.y_max),
            Point(self.x_min, self.y_max),
        )

    @cached_property
    def obj(self) -> shapely.Polygon:
        return shapely.box(self.x_min, self.y_min, self.x_max, self.y_max)

    def __repr__(self) -> str:
        return (
            f"Box([{self.x_min_int}, {self.y_min_int}, {self.x_max_int}, {self.y_max_int}], "
            f"conf={self.conf:.2f}, cat={self.category_idx})"
        )

    # --- 变换 ---
    def offset(self, delta_x: float = 0.0, delta_y: float = 0.0) -> Self:
        return replace(
            self,
            start=self.start.offset(delta_x, delta_y),
            end=self.end.offset(delta_x, delta_y),
        )

    def scale(self, scale_x: float = 1.0, scale_y: float = 1.0) -> Self:
        return replace(
            self,
            start=self.start.scale(scale_x, scale_y),
            end=self.end.scale(scale_x, scale_y),
        )

    def resize(self, width: float, height: float) -> Self:
        """以左上角为锚点改为指定尺寸。"""
        return Box(
            self.start,
            Point(self.x_min + width, self.y_min + height),
            self.category_idx,
            self.conf,
        )

    # --- 交并比 ---
    def iou(self, other: Box) -> float:
        """交并比（IoU）。"""
        cross = _intersection_area(self, other)
        return cross / (self.area + other.area - cross) if cross > 0 else 0.0

    def iou_max(self, other: Box) -> float:
        """交集面积 / 两者中较大的面积（小目标被大框包住时更有区分度）。"""
        cross = _intersection_area(self, other)
        if cross <= 0 or self.area <= 0 or other.area <= 0:
            return 0.0
        return max(cross / self.area, cross / other.area)

    # --- 图像区域 ---
    def crop(self, image: Any, expand: int = 0) -> Any:
        """裁剪出该框对应的图像区域（越界部分自动截断）。"""
        import numpy as np

        x1 = max(self.x_min_int - expand, 0)
        y1 = max(self.y_min_int - expand, 0)
        x2 = min(self.x_max_int + expand, image.shape[1])
        y2 = min(self.y_max_int + expand, image.shape[0])
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, *image.shape[2:]), image.dtype)
        return image[y1:y2, x1:x2]

    # --- 序列化 ---
    def to_yolo(self, img_size: ImgSize, *, with_conf: bool = False) -> str:
        tokens = [
            str(self.category_idx),
            _num(self.x_center / img_size.width),
            _num(self.y_center / img_size.height),
            _num(self.width / img_size.width),
            _num(self.height / img_size.height),
        ]
        if with_conf:
            tokens.append(_num(self.conf))
        return " ".join(tokens)

    @classmethod
    def from_yolo(cls, line: str, img_size: ImgSize) -> Self:
        values = [float(v) for v in line.split()]
        width, height = values[3] * img_size.width, values[4] * img_size.height
        cx, cy = values[1] * img_size.width, values[2] * img_size.height
        return cls(
            start=Point(cx - width / 2, cy - height / 2),
            end=Point(cx + width / 2, cy + height / 2),
            category_idx=int(values[0]),
            conf=values[5] if len(values) > 5 else -1.0,
        )

    @classmethod
    def from_xyxy(
        cls, x1: float, y1: float, x2: float, y2: float, **kwargs: Any
    ) -> Self:
        """由左上 / 右下角坐标构造（ultralytics 的 ``xyxy`` 布局）。"""
        return cls(Point(x1, y1), Point(x2, y2), **kwargs)

    @classmethod
    def bounding(cls, objs: Sequence[Bounds], **kwargs: Any) -> Self:
        """一组图元的公共外接框。"""
        if not objs:
            raise ValueError("objs 不能为空")
        return cls(
            Point(min(o.x_min for o in objs), min(o.y_min for o in objs)),
            Point(max(o.x_max for o in objs), max(o.y_max for o in objs)),
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "kind": "box",
            "x1": self.x_min_int,
            "y1": self.y_min_int,
            "x2": self.x_max_int,
            "y2": self.y_max_int,
            "cat": self.category_idx,
            "conf": round(self.conf, 4),
        }
        if self.track_id >= 0:
            data["track_id"] = self.track_id
        return data

    def to_sql_model(self, img: Any, session: Any) -> Any:
        """转为检测标注实体。"""
        from ..entity_model import TCategory, TDetectAnno, select

        category = session.exec(
            select(TCategory).where(TCategory.idx == self.category_idx)
        ).one()
        return TDetectAnno(
            x=self.x_min,
            y=self.y_min,
            width=self.width,
            height=self.height,
            conf=self.conf,
            img=img,
            category=category,
        )


def _intersection_area(left: Box, right: Box) -> float:
    width = min(left.x_max, right.x_max) - max(left.x_min, right.x_min)
    height = min(left.y_max, right.y_max) - max(left.y_min, right.y_min)
    return width * height if width > 0 and height > 0 else 0.0


# ============================================================
# 旋转框
# ============================================================


@dataclass(frozen=True)
class Obb(Bounds):
    """旋转矩形框（YOLO OBB 任务）。

    ``size`` 为 ``(宽, 高)``（未旋转时的边长），``angle`` 为绕中心顺时针角度（度）。
    """

    center: Point = field(default_factory=Point)
    size: tuple[float, float] = (0.0, 0.0)
    angle: float = 0.0
    category_idx: int = 0
    conf: float = -1.0
    create_time: datetime = field(
        default_factory=datetime.now, compare=False, repr=False
    )

    @property
    def width(self) -> float:
        return self.size[0]

    @property
    def height(self) -> float:
        return self.size[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @cached_property
    def points(self) -> tuple[Point, Point, Point, Point]:
        """四个角点，自左上开始顺时针。"""
        radians = math.radians(self.angle)
        cos_a, sin_a = math.cos(radians), math.sin(radians)
        half_w, half_h = self.width / 2, self.height / 2
        return tuple(
            Point(
                x=self.center.x + dx * cos_a - dy * sin_a,
                y=self.center.y + dx * sin_a + dy * cos_a,
            )
            for dx, dy in (
                (-half_w, -half_h),
                (half_w, -half_h),
                (half_w, half_h),
                (-half_w, half_h),
            )
        )

    @cached_property
    def obj(self) -> shapely.Polygon:
        return shapely.Polygon([p.xy for p in self.points])

    def __repr__(self) -> str:
        return (
            f"Obb(center={self.center!r}, size=({_num(self.width)}, {_num(self.height)}), "
            f"angle={_num(self.angle)}, conf={self.conf:.2f}, cat={self.category_idx})"
        )

    # --- 变换 ---
    def offset(self, delta_x: float = 0.0, delta_y: float = 0.0) -> Self:
        return replace(self, center=self.center.offset(delta_x, delta_y))

    def scale(self, scale_x: float = 1.0, scale_y: float = 1.0) -> Self:
        """收缩放：非等比缩放会把矩形变成平行四边形，此处按均值近似。"""
        return replace(
            self,
            center=self.center.scale(scale_x, scale_y),
            size=(
                self.width * scale_x,
                self.height * scale_y,
            ),
        )

    # --- 转换 ---
    def to_polygon(self) -> Polygon:
        return Polygon(self.points, self.category_idx, self.conf)

    def to_yolo(self, img_size: ImgSize, *, with_conf: bool = False) -> str:
        tokens = [str(self.category_idx)]
        for point in self.points:
            tokens.extend(
                (_num(point.x / img_size.width), _num(point.y / img_size.height))
            )
        if with_conf:
            tokens.append(_num(self.conf))
        return " ".join(tokens)

    @classmethod
    def from_yolo(cls, line: str, img_size: ImgSize) -> Self:
        values = [float(v) for v in line.split()]
        corners = [
            Point(values[i] * img_size.width, values[i + 1] * img_size.height)
            for i in range(1, min(len(values), 9), 2)
        ]
        if len(corners) < 4:
            raise ValueError(f"OBB 需要 4 个角点：{line!r}")
        return cls.from_corners(
            corners,
            category_idx=int(values[0]),
            conf=values[9] if len(values) > 9 else -1.0,
        )

    @classmethod
    def from_corners(cls, corners: Sequence[Point], **kwargs: Any) -> Self:
        """由四个角点（顺时针）反算中心、尺寸与角度。"""
        if len(corners) != 4:
            raise ValueError(f"需要 4 个角点，收到 {len(corners)} 个")
        x = sum(p.x for p in corners) / 4
        y = sum(p.y for p in corners) / 4
        width = math.dist(corners[0].xy, corners[1].xy)
        height = math.dist(corners[1].xy, corners[2].xy)
        angle = math.degrees(
            math.atan2(corners[1].y - corners[0].y, corners[1].x - corners[0].x)
        )
        return cls(Point(x, y), (width, height), angle, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "obb",
            "cx": round(self.center.x, 3),
            "cy": round(self.center.y, 3),
            "w": round(self.width, 3),
            "h": round(self.height, 3),
            "angle": round(self.angle, 2),
            "cat": self.category_idx,
            "conf": round(self.conf, 4),
        }


# ============================================================
# 多边形
# ============================================================


@dataclass(frozen=True)
class Polygon(Bounds):
    """由顶点序列构成的多边形（分割标注）。"""

    points: tuple[Point, ...] = ()
    category_idx: int = 0
    conf: float = -1.0
    create_time: datetime = field(
        default_factory=datetime.now, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.points, tuple):
            object.__setattr__(self, "points", tuple(self.points))

    @cached_property
    def obj(self) -> shapely.Polygon:
        return shapely.Polygon([p.xy for p in self.points])

    @cached_property
    def area(self) -> float:
        """鞋带公式求面积。"""
        n = len(self.points)
        if n < 3:
            return 0.0
        total = 0.0
        for index in range(n):
            current, following = self.points[index], self.points[(index + 1) % n]
            total += current.x * following.y - following.x * current.y
        return abs(total) / 2

    @property
    def perimeter(self) -> float:
        n = len(self.points)
        if n < 2:
            return 0.0
        return sum(
            math.dist(self.points[i].xy, self.points[(i + 1) % n].xy) for i in range(n)
        )

    @cached_property
    def centroid(self) -> Point:
        if not self.points:
            return Point()
        return Point(
            sum(p.x for p in self.points) / len(self.points),
            sum(p.y for p in self.points) / len(self.points),
        )

    def __repr__(self) -> str:
        return f"Polygon({len(self.points)} pts, conf={self.conf:.2f}, cat={self.category_idx})"

    # --- 变换 ---
    def offset(self, delta_x: float = 0.0, delta_y: float = 0.0) -> Self:
        return replace(
            self, points=tuple(p.offset(delta_x, delta_y) for p in self.points)
        )

    def scale(self, scale_x: float = 1.0, scale_y: float = 1.0) -> Self:
        return replace(
            self, points=tuple(p.scale(scale_x, scale_y) for p in self.points)
        )

    # --- 简化（掩码轮廓点通常非常密集） ---
    def simplify(
        self,
        *,
        epsilon_ratio: float = 0.002,
        min_vertices: int = 3,
        max_vertices: int = 100,
        min_dist: float = 2.0,
    ) -> Self:
        """RDP 简化：在保留形状特征的前提下减少顶点数。

        ``epsilon_ratio`` 是相对轮廓周长的误差阈值，越大越简化。
        """
        import cv2
        import numpy as np

        if len(self.points) < 4:
            return self

        contour = np.array([p.xy for p in self.points], dtype=np.float32).reshape(
            -1, 1, 2
        )
        epsilon = max(epsilon_ratio * cv2.arcLength(contour, True), 1e-6)

        def reduce(value: float) -> np.ndarray:
            return cv2.approxPolyDP(contour, value, True).reshape(-1, 2)

        simplified = reduce(epsilon)
        while len(simplified) < min_vertices and epsilon > 1e-6:
            epsilon *= 0.5
            simplified = reduce(epsilon)
        while len(simplified) > max_vertices:
            epsilon *= 1.2
            simplified = reduce(epsilon)

        points = [Point(float(x), float(y)) for x, y in simplified]
        if min_dist > 0:
            points = [points[0]] + [
                point
                for previous, point in pairwise(points)
                if math.dist(previous.xy, point.xy) >= min_dist
            ]
        return replace(self, points=tuple(points)) if len(points) >= 3 else self

    # --- 序列化 ---
    def to_yolo(self, img_size: ImgSize, *, with_conf: bool = False) -> str:
        tokens = [str(self.category_idx)]
        for point in self.points:
            tokens.extend(
                (_num(point.x / img_size.width), _num(point.y / img_size.height))
            )
        if with_conf:
            tokens.append(_num(self.conf))
        return " ".join(tokens)

    @classmethod
    def from_yolo(cls, line: str, img_size: ImgSize) -> Self:
        values = [float(v) for v in line.split()]
        coords, conf = values[1:], -1.0
        if len(coords) % 2:  # 坐标成对出现，多出来的那个是置信度
            coords, conf = coords[:-1], coords[-1]
        return cls(
            points=tuple(
                Point(coords[i] * img_size.width, coords[i + 1] * img_size.height)
                for i in range(0, len(coords), 2)
            ),
            category_idx=int(values[0]),
            conf=conf,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "polygon",
            "points": [[round(p.x, 2), round(p.y, 2)] for p in self.points],
            "cat": self.category_idx,
            "conf": round(self.conf, 4),
        }

    def to_sql_model(self, img: Any, session: Any) -> Any:
        """转为分割标注实体（顶点序列以 pickle 存储，与旧版一致）。"""
        import pickle

        from ..entity_model import TCategory, TSegmentAnno, select

        category = session.exec(
            select(TCategory).where(TCategory.idx == self.category_idx)
        ).one()
        return TSegmentAnno(
            points=pickle.dumps(self.points),
            conf=self.conf,
            img=img,
            category=category,
        )


# ============================================================
# 圆
# ============================================================


@dataclass(frozen=True)
class Circle(Bounds):
    """由圆心与半径确定的圆。"""

    center: Point = field(default_factory=Point)
    radius: float = 0.0
    category_idx: int = 0
    conf: float = -1.0
    create_time: datetime = field(
        default_factory=datetime.now, compare=False, repr=False
    )

    # --- 边界（解析式，无需采样） ---
    @property
    def x_min(self) -> float:
        return self.center.x - self.radius

    @property
    def y_min(self) -> float:
        return self.center.y - self.radius

    @property
    def x_max(self) -> float:
        return self.center.x + self.radius

    @property
    def y_max(self) -> float:
        return self.center.y + self.radius

    @property
    def width(self) -> float:
        return 2 * self.radius

    @property
    def height(self) -> float:
        return 2 * self.radius

    @property
    def diameter(self) -> float:
        return 2 * self.radius

    @property
    def area(self) -> float:
        return math.pi * self.radius**2

    @property
    def circumference(self) -> float:
        return 2 * math.pi * self.radius

    @cached_property
    def points(self) -> tuple[Point, ...]:
        """圆周采样点（默认 64 个，自 x 轴正方向顺时针排列）。"""
        return tuple(
            Point(
                x=self.center.x
                + self.radius * math.cos(2 * math.pi * i / CIRCLE_SAMPLES),
                y=self.center.y
                + self.radius * math.sin(2 * math.pi * i / CIRCLE_SAMPLES),
            )
            for i in range(CIRCLE_SAMPLES)
        )

    @cached_property
    def obj(self) -> shapely.Polygon:
        return shapely.Point(self.center.xy).buffer(
            self.radius, quad_segs=CIRCLE_SAMPLES // 4
        )

    def __repr__(self) -> str:
        return (
            f"Circle(center={self.center!r}, r={_num(self.radius)}, "
            f"conf={self.conf:.2f}, cat={self.category_idx})"
        )

    # --- 变换 ---
    def offset(self, delta_x: float = 0.0, delta_y: float = 0.0) -> Self:
        return replace(self, center=self.center.offset(delta_x, delta_y))

    def scale(self, scale_x: float = 1.0, scale_y: float = 1.0) -> Self:
        """非等比缩放会把圆变成椭圆，半径按均值近似（等比缩放时精确）。"""
        return replace(
            self,
            center=self.center.scale(scale_x, scale_y),
            radius=self.radius * (scale_x + scale_y) / 2,
        )

    # --- 判断 ---
    def contains(self, point: Point, tolerance: float = 1e-9) -> bool:
        """点是否位于圆内或圆周上。"""
        return self.center.distance(point) <= self.radius + tolerance

    def distance_to_boundary(self, point: Point) -> float:
        """点到圆周的距离（点在圆内同样返回正值）。"""
        return abs(self.center.distance(point) - self.radius)

    # --- 序列化 ---
    def to_yolo(self, img_size: ImgSize, *, with_conf: bool = False) -> str:
        short_side = min(img_size.width, img_size.height)
        tokens = [
            str(self.category_idx),
            _num(self.center.x / img_size.width),
            _num(self.center.y / img_size.height),
            _num(self.radius / short_side),
        ]
        if with_conf:
            tokens.append(_num(self.conf))
        return " ".join(tokens)

    @classmethod
    def from_yolo(cls, line: str, img_size: ImgSize) -> Self:
        values = [float(v) for v in line.split()]
        return cls(
            center=Point(values[1] * img_size.width, values[2] * img_size.height),
            radius=values[3] * min(img_size.width, img_size.height),
            category_idx=int(values[0]),
            conf=values[4] if len(values) > 4 else -1.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "circle",
            "cx": round(self.center.x, 3),
            "cy": round(self.center.y, 3),
            "r": round(self.radius, 3),
            "cat": self.category_idx,
            "conf": round(self.conf, 4),
        }


# ============================================================
# 姿态关键点
# ============================================================


@dataclass(frozen=True)
class Pose(Bounds):
    """一个实例的关键点集合（姿态估计）。

    ``points`` 中每个 ``Point.conf`` 存该关键点的置信度 / 可见性，``box`` 为可选的实例框。
    """

    points: tuple[Point, ...] = ()
    category_idx: int = 0
    conf: float = -1.0
    box: Box | None = None
    create_time: datetime = field(
        default_factory=datetime.now, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.points, tuple):
            object.__setattr__(self, "points", tuple(self.points))

    @cached_property
    def _frame(self) -> tuple[float, float, float, float]:
        if self.box is not None:
            return self.box.xyxy
        if not self.points:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            min(p.x for p in self.points),
            min(p.y for p in self.points),
            max(p.x for p in self.points),
            max(p.y for p in self.points),
        )

    @property
    def x_min(self) -> float:
        return self._frame[0]

    @property
    def y_min(self) -> float:
        return self._frame[1]

    @property
    def x_max(self) -> float:
        return self._frame[2]

    @property
    def y_max(self) -> float:
        return self._frame[3]

    @cached_property
    def obj(self) -> shapely.MultiPoint:
        return (
            shapely.MultiPoint([p.xy for p in self.points])
            if self.points
            else shapely.MultiPoint()
        )

    @property
    def keypoints(self) -> tuple[Point, ...]:
        """语义化别名（``points`` 同时用于外接框计算，容易被误解）。"""
        return self.points

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return f"Pose({len(self.points)} kpts, conf={self.conf:.2f}, cat={self.category_idx})"

    # --- 变换 ---
    def offset(self, delta_x: float = 0.0, delta_y: float = 0.0) -> Self:
        return replace(
            self,
            points=tuple(p.offset(delta_x, delta_y) for p in self.points),
            box=self.box.offset(delta_x, delta_y) if self.box else None,
        )

    def scale(self, scale_x: float = 1.0, scale_y: float = 1.0) -> Self:
        return replace(
            self,
            points=tuple(p.scale(scale_x, scale_y) for p in self.points),
            box=self.box.scale(scale_x, scale_y) if self.box else None,
        )

    def to_polygon(self) -> Polygon:
        """关键点无法构成多边形，这里返回外接框多边形（便于统一绘制/存储）。"""
        return Polygon(
            self.box.points if self.box else self.to_box().points,
            self.category_idx,
            self.conf,
        )

    # --- 序列化 ---
    def to_yolo(self, img_size: ImgSize) -> str:
        """姿态行：``cls x1 y1 v1 ...``，``v`` 由关键点置信度换算。"""
        tokens = [str(self.category_idx)]
        for point in self.points:
            tokens.extend(
                (
                    _num(point.x / img_size.width),
                    _num(point.y / img_size.height),
                    str(_visibility(point.conf)),
                )
            )
        return " ".join(tokens)

    @classmethod
    def from_yolo(cls, line: str, img_size: ImgSize) -> Self:
        values = [float(v) for v in line.split()]
        points = tuple(
            Point(
                values[i] * img_size.width,
                values[i + 1] * img_size.height,
                conf=values[i + 2] / 2,
            )
            for i in range(1, len(values) - 2, 3)
        )
        return cls(points=points, category_idx=int(values[0]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "pose",
            "points": [
                [round(p.x, 2), round(p.y, 2), round(p.conf, 3)] for p in self.points
            ],
            "box": self.box.to_dict() if self.box else None,
            "cat": self.category_idx,
            "conf": round(self.conf, 4),
        }


def _visibility(conf: float) -> int:
    """置信度 → YOLO 可见性标记（0 未标注 / 1 遮挡 / 2 可见）。

    ``from_yolo`` 用 ``v / 2`` 反算置信度，这里的边界与之一致，保证往返稳定。
    """
    if conf <= 0:
        return 0
    return 1 if conf <= 0.5 else 2


type Geometry = Point | Line | Box | Obb | Polygon | Circle | Pose
"""可参与标注、绘制与序列化的几何图元。"""
