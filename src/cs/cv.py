"""计算机视觉标注数据结构。

提供 Point、Box、Polygon、Line、Circle、Category、AnnoImg、AnnoDataset 等类型。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, override

import cv2
import numpy as np
import shapely

from .color import Color
from .my_types import ImgSize, TaskType

if TYPE_CHECKING:
    from collections.abc import Sequence

POINT_RADIUS = 4
# 圆周采样点数量：需为 4 的倍数，以保证取到上下左右四个极点，
# 这样由采样点得到的外接框与圆的真实外接框完全一致。
CIRCLE_SAMPLE_NUM = 64


# ============================================================
# 类别
# ============================================================


@dataclass
class Category:
    """类别名称列表，支持按下标或名称查找。"""

    names: list[str] = field(default_factory=list)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def __getitem__(self, idx: int) -> str:
        return self.names[idx]

    def __len__(self) -> int:
        return len(self.names)


# ============================================================
# 几何工具 Mixin
# ============================================================


class _GeometryBounds:
    """基于 ``points`` 提供几何边界计算能力的 Mixin。"""

    points: Sequence["Point"]

    @property
    def x_min(self) -> float:
        return min(p.x for p in self.points)

    @property
    def y_min(self) -> float:
        return min(p.y for p in self.points)

    @property
    def x_max(self) -> float:
        return max(p.x for p in self.points)

    @property
    def y_max(self) -> float:
        return max(p.y for p in self.points)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def validate(self, img_size: ImgSize) -> bool:
        return not (
            self.x_min < 0
            or self.y_min < 0
            or self.x_max > img_size.width
            or self.y_max > img_size.height
        )

    def limit_obj(self, size: ImgSize) -> None:  # pyright: ignore[reportUnannotatedBase]
        """将对象裁剪到图像边界之内。"""
        dx = dy = 0.0
        if self.x_min < 0:
            dx = -self.x_min
        if self.y_min < 0:
            dy = -self.y_min
        if self.x_max > size.width:
            dx = size.width - self.x_max if dx == 0 else dx
        if self.y_max > size.height:
            dy = size.height - self.y_max if dy == 0 else dy
        if dx or dy:
            self.offset(dx, dy)  # type: ignore[attr-defined]


# ============================================================
# 点
# ============================================================


@dataclass
class Point(_GeometryBounds):
    """二维点，包含类别下标、置信度与时间戳。"""

    x: float = -1.0
    y: float = -1.0
    category_idx: int = 0
    conf: float = -1.0
    create_time: datetime = field(default_factory=datetime.now)

    # --- 为提升性能而重写 Mixin 的边界计算 ---
    @property
    @override
    def x_min(self) -> float:
        return self.x - POINT_RADIUS

    @property
    @override
    def y_min(self) -> float:
        return self.y - POINT_RADIUS

    @property
    @override
    def x_max(self) -> float:
        return self.x + POINT_RADIUS

    @property
    @override
    def y_max(self) -> float:
        return self.y + POINT_RADIUS

    @property
    @override
    def width(self) -> float:
        return 2 * POINT_RADIUS

    @property
    @override
    def height(self) -> float:
        return 2 * POINT_RADIUS

    # --- 快速转整数 ---
    @property
    def x_int(self) -> int:
        return int(self.x)

    @property
    def y_int(self) -> int:
        return int(self.y)

    # --- 打印表示 ---
    def __repr__(self) -> str:
        return f"({self.x},{self.y})"

    # --- 惰性构建的几何对象 ---
    @cached_property
    def points(self) -> list[Point]:
        r = POINT_RADIUS
        return [
            Point(self.x - r, self.y - r),
            Point(self.x + r, self.y - r),
            Point(self.x + r, self.y + r),
            Point(self.x - r, self.y + r),
        ]

    @cached_property
    def obj(self) -> shapely.Point:
        return shapely.Point(self.x, self.y)

    @cached_property
    def box(self) -> Box:
        return Box(
            start=Point(self.x_min, self.y_min), end=Point(self.x_max, self.y_max)
        )

    # --- 格式转换 ---
    def to_str(self, img_size: ImgSize, with_conf: bool = False) -> str:
        line = (
            f"{self.category_idx} {self.x / img_size.width} {self.y / img_size.height}"
        )
        return f"{line} {self.conf}\n" if with_conf else f"{line}\n"

    def to_sql_model(self, img, session):
        from .entity_model import TCategory, TClassifyAnno, select

        t_category = session.exec(
            select(TCategory).where(TCategory.idx == self.category_idx)
        ).one()
        return TClassifyAnno(conf=self.conf, img=img, category=t_category)

    # --- 操作（支持链式调用） ---
    def offset(self, delta_x: float = 0.0, delta_y: float = 0.0) -> Point:
        """返回按给定位移平移后的新点。"""
        return replace(self, x=self.x + delta_x, y=self.y + delta_y)

    def scale(self, scale_x: float = 1.0, scale_y: float = 1.0) -> Point:
        """返回按给定比例缩放坐标后的新点。"""
        return replace(self, x=self.x * scale_x, y=self.y * scale_y)

    def distance(self, point: Point) -> float:
        return shapely.distance(self.obj, point.obj)

    def is_in_obj(self, obj: Point | Box | Polygon | Line | Circle) -> bool:
        """判断该点是否落在 ``obj`` 的外接框内。"""
        return obj.x_min <= self.x <= obj.x_max and obj.y_min <= self.y <= obj.y_max

    # --- Flet Canvas 绘制 ---
    def canvas_text(
        self, scale=1.0, left=0, top=0, category: Category | None = None, data=None
    ):
        import flet as ft
        from flet import canvas

        value = category[self.category_idx] if category else "category"
        return canvas.Text(
            self.x_min * scale + left,
            self.y_min * scale + top,
            value,
            ft.TextStyle(
                weight=ft.FontWeight.BOLD, size=14, color=Color[self.category_idx].hex
            ),
            text_align=ft.TextAlign.LEFT,
            data=data,
        )

    def canvas_stroke(
        self, scale=1.0, left=0, top=0, data=None, radius: int = 3, stroke_width=1
    ):
        from flet import canvas

        from .flet import stroke_paint

        return canvas.Circle(
            self.x * scale + left,
            self.y * scale + top,
            radius=radius,
            paint=stroke_paint(color_idx=self.category_idx, stroke_width=stroke_width),
            data=data,
        )

    def canvas_fill(self, scale=1.0, left=0, top=0, data=None):
        from flet import canvas

        from .flet import fill_paint

        return canvas.Circle(
            self.x * scale + left,
            self.y * scale + top,
            radius=3,
            paint=fill_paint(color_idx=self.category_idx),
            data=data,
        )


# ============================================================
# 矩形框
# ============================================================


@dataclass
class Box(_GeometryBounds):
    """由对角两点定义的轴对齐矩形框
    （会自动排序为左上 → 右下）。
    """

    start: Point = field(default_factory=Point)
    end: Point = field(default_factory=Point)
    category_idx: int = 0
    conf: float = -1.0
    create_time: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        self.sort_point()

    # --- 为提升性能而重写 Mixin 的边界计算 ---
    @property
    @override
    def x_min(self) -> float:
        return self.start.x

    @property
    @override
    def y_min(self) -> float:
        return self.start.y

    @property
    @override
    def x_max(self) -> float:
        return self.end.x

    @property
    @override
    def y_max(self) -> float:
        return self.end.y

    # --- 整数快捷属性 ---
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

    @property
    def x_center(self) -> float:
        return self.x_min + self.width / 2

    @property
    def y_center(self) -> float:
        return self.y_min + self.height / 2

    # --- 面积与长宽比 ---
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

    # --- XYXY / IOU 数据 ---
    @property
    def xyxy(self) -> list[float]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    @property
    def iou_data(self) -> list[float]:
        return self.xyxy

    # --- 顶点列表 / Shapely 对象 ---
    @cached_property
    def points(self) -> list[Point]:
        return [
            Point(self.x_min, self.y_min),
            Point(self.x_max, self.y_min),
            Point(self.x_max, self.y_max),
            Point(self.x_min, self.y_max),
        ]

    @cached_property
    def obj(self) -> shapely.Polygon:
        return shapely.box(self.x_min, self.y_min, self.x_max, self.y_max)

    # --- 打印表示 ---
    def __repr__(self) -> str:
        return f"[{self.x_min_int}, {self.y_min_int}, {self.x_max_int}, {self.y_max_int}, {self.conf:.2f}, {self.category_idx}]"

    # --- 内部排序 ---
    def sort_point(self) -> None:
        if self.start.x > self.end.x:
            self.start.x, self.end.x = self.end.x, self.start.x
        if self.start.y > self.end.y:
            self.start.y, self.end.y = self.end.y, self.start.y

    # --- 操作（返回 self 以支持链式调用） ---
    def offset(self, delta_x: float = 0.0, delta_y: float = 0.0) -> Box:
        """原地平移矩形框并返回 self。"""
        self.start.x += delta_x
        self.start.y += delta_y
        self.end.x += delta_x
        self.end.y += delta_y
        return self

    def scale(self, scale_x: float = 1.0, scale_y: float = 1.0) -> Box:
        """原地缩放矩形框并返回 self。"""
        self.start.x *= scale_x
        self.start.y *= scale_y
        self.end.x *= scale_x
        self.end.y *= scale_y
        return self

    # --- 距离与 IOU ---
    def distance(self, box: Box) -> float:
        return shapely.distance(self.obj, box.obj)

    def iou(self, box: Box) -> float:
        """计算交并比（IoU）。"""
        left = max(self.x_min, box.x_min)
        top = max(self.y_min, box.y_min)
        right = min(self.x_max, box.x_max)
        bottom = min(self.y_max, box.y_max)
        cross = max(right - left, 0) * max(bottom - top, 0)
        union = self.area + box.area - cross
        return cross / union if cross > 0 and union > 0 else 0.0

    def iou_max(self, box: Box) -> float:
        """计算交集 / max(self.area, box.area)。"""
        left = max(self.x_min, box.x_min)
        top = max(self.y_min, box.y_min)
        right = min(self.x_max, box.x_max)
        bottom = min(self.y_max, box.y_max)
        cross = max(right - left, 0) * max(bottom - top, 0)
        if cross <= 0 or self.area <= 0 or box.area <= 0:
            return 0.0
        return max(cross / self.area, cross / box.area)

    # --- 裁剪区域 ---
    def get_box_crop_cv2(
        self, img_cv2: np.ndarray, expand_size: int = 50
    ) -> np.ndarray:
        y1, y2 = int(self.y_min - expand_size), int(self.y_max + expand_size)
        x1, x2 = int(self.x_min - expand_size), int(self.x_max + expand_size)
        crop = img_cv2[y1:y2, x1:x2]
        return cv2.rectangle(
            crop,
            (expand_size, expand_size),
            (int(self.width + expand_size), int(self.height + expand_size)),
            (255, 0, 0),
            1,
        )

    def get_region_data(self, img: np.ndarray, flag=1) -> float | None:
        """返回该区域的统计值（目前仅支持最大灰度值）。"""
        roi = img[int(self.y_min) : int(self.y_max), int(self.x_min) : int(self.x_max)]
        return cv2.minMaxLoc(roi)[1] if roi.size else None

    # --- 灰度统计（torch / np） ---
    def _roi_torch(self, img_torch):
        return img_torch[
            self.y_min_int : self.y_max_int, self.x_min_int : self.x_max_int
        ]

    def get_max_grayscale(self, img_torch) -> float:
        return self._roi_torch(img_torch).max().item()

    def get_min_grayscale(self, img_torch) -> float:
        return self._roi_torch(img_torch).min().item()

    def get_diff_grayscale(self, img_torch) -> float:
        roi = self._roi_torch(img_torch)
        return (roi.max() - roi.min()).item()

    def get_avg_grayscale(self, img_torch) -> float:
        return self._roi_torch(img_torch).mean().item()

    def _count_roi_np(self, img_cv2: np.ndarray, op) -> int:
        roi = img_cv2[self.y_min_int : self.y_max_int, self.x_min_int : self.x_max_int]
        return int(op(roi).sum())

    def gt_grayscale_nums(self, value, img_cv2: np.ndarray) -> int:
        return self._count_roi_np(img_cv2, lambda r: r > value)

    def ge_grayscale_nums(self, value, img_cv2: np.ndarray) -> int:
        return self._count_roi_np(img_cv2, lambda r: r >= value)

    def le_grayscale_nums(self, value, img_cv2: np.ndarray) -> int:
        return self._count_roi_np(img_cv2, lambda r: r <= value)

    def lt_grayscale_nums(self, value, img_cv2: np.ndarray) -> int:
        return self._count_roi_np(img_cv2, lambda r: r < value)

    # --- 格式转换 ---
    def load_yolo_txt_line(
        self, img_width: int, img_height: int, line_str: str
    ) -> None:
        yolo_box = [float(i) for i in line_str.split()]
        box_w = yolo_box[3] * img_width
        box_h = yolo_box[4] * img_height
        x1 = int(yolo_box[1] * img_width - box_w / 2)
        y1 = int(yolo_box[2] * img_height - box_h / 2)
        self.start = Point(x=x1, y=y1)
        self.end = Point(x=int(x1 + box_w), y=int(y1 + box_h))
        self.category_idx = int(yolo_box[0])
        self.sort_point()

    def to_str(self, img_size: ImgSize, with_conf: bool = False) -> str:
        line = (
            f"{self.category_idx} "
            f"{self.x_center / img_size.width} "
            f"{self.y_center / img_size.height} "
            f"{self.width / img_size.width} "
            f"{self.height / img_size.height}"
        )
        return f"{line} {self.conf}\n" if with_conf else f"{line}\n"

    def to_json(self) -> dict:
        return {
            "x1": int(self.start.x),
            "y1": int(self.start.y),
            "x2": int(self.end.x),
            "y2": int(self.end.y),
            "category_idx": self.category_idx,
            "conf": f"{self.conf:.3f}",
        }

    def to_sql_model(self, img, session):
        from .entity_model import TCategory, TDetectAnno, select

        t_category = session.exec(
            select(TCategory).where(TCategory.idx == self.category_idx)
        ).one()
        return TDetectAnno(
            x=self.x_min,
            y=self.y_min,
            width=self.width,
            height=self.height,
            conf=self.conf,
            img=img,
            category=t_category,
        )

    # --- 静态工具方法 ---
    @staticmethod
    def bounding_box(objs: list[Point | Box | Polygon | Line | Circle]) -> list[int]:
        return [
            int(f([getattr(o, attr) for o in objs]))
            for f, attr in [
                (min, "x_min"),
                (min, "y_min"),
                (max, "x_max"),
                (max, "y_max"),
            ]
        ]

    # --- Flet Canvas 绘制 ---
    def canvas_text(
        self, scale=1.0, left=0, top=0, category: Category | None = None, data=None
    ):
        import flet as ft
        from flet import canvas

        value = category[self.category_idx] if category else "category"
        return canvas.Text(
            self.x_min * scale + left,
            self.y_min * scale + top,
            value,
            ft.TextStyle(
                weight=ft.FontWeight.BOLD, size=14, color=Color[self.category_idx].hex
            ),
            text_align=ft.TextAlign.LEFT,
            data=data,
        )

    def canvas_stroke(self, scale=1.0, left=0, top=0, data=None):
        from flet import canvas

        from .flet import stroke_paint

        return canvas.Rect(
            self.x_min * scale + left,
            self.y_min * scale + top,
            self.width * scale,
            self.height * scale,
            paint=stroke_paint(color_idx=self.category_idx),
            data=data,
        )

    def canvas_fill(self, scale=1.0, left=0, top=0, data=None):
        from flet import canvas

        from .flet import fill_paint

        return canvas.Rect(
            self.x_min * scale + left,
            self.y_min * scale + top,
            self.width * scale,
            self.height * scale,
            paint=fill_paint(color_idx=self.category_idx),
            data=data,
        )


# ============================================================
# 多边形
# ============================================================


@dataclass
class Polygon(_GeometryBounds):
    """由一组顶点定义的多边形标注。"""

    points: list[Point] = field(default_factory=list)
    category_idx: int = 0
    conf: float = -1.0
    create_time: datetime = field(default_factory=datetime.now)

    # --- Shapely 对象 ---
    @cached_property
    def obj(self) -> shapely.Polygon:
        return shapely.Polygon([(p.x, p.y) for p in self.points])

    # --- 面积（鞋带公式） ---
    @cached_property
    def area(self) -> float:
        n = len(self.points)
        if n < 3:
            return 0.0
        pts = self.points
        s = pts[-1].x * pts[0].y - pts[0].x * pts[-1].y
        for i in range(n - 1):
            s += pts[i].x * pts[i + 1].y - pts[i + 1].x * pts[i].y
        return abs(s) / 2.0

    # --- 格式转换 ---
    def load_yolo_txt_line(
        self, img_width: int, img_height: int, line_str: str
    ) -> None:
        parts = line_str.split()
        self.category_idx = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        self.points = [
            Point(x=coords[i] * img_width, y=coords[i + 1] * img_height)
            for i in range(0, len(coords), 2)
        ]

    def to_str(self, img_size: ImgSize) -> str:
        tokens = [str(self.category_idx)]
        for p in self.points:
            tokens.append(f"{p.x / img_size.width}")
            tokens.append(f"{p.y / img_size.height}")
        return " ".join(tokens) + "\n"

    def to_sql_model(self, img, session):
        import pickle

        from .entity_model import TCategory, TSegmentAnno, select

        t_category = session.exec(
            select(TCategory).where(TCategory.idx == self.category_idx)
        ).one()
        return TSegmentAnno(
            points=pickle.dumps(self.points),
            conf=self.conf,
            img=img,
            category=t_category,
        )

    # --- 操作（返回 self） ---
    def offset(self, delta_x: float = 0.0, delta_y: float = 0.0) -> Polygon:
        for p in self.points:
            p.x += delta_x
            p.y += delta_y
        return self

    def scale(self, scale_x: float = 1.0, scale_y: float = 1.0) -> Polygon:
        for p in self.points:
            p.x *= scale_x
            p.y *= scale_y
        return self

    def plot(self, img_cv2: np.ndarray) -> None:
        """占位方法 —— 实际绘制由 ``cv_draw`` 模块负责。"""
        pass

    # --- Flet Canvas 绘制 ---
    def canvas_text(
        self, scale=1.0, left=0, top=0, category: Category | None = None, data=None
    ):
        import flet as ft
        from flet import canvas

        value = category[self.category_idx] if category else "category"
        return canvas.Text(
            self.x_min * scale + left,
            self.y_min * scale + top,
            value,
            ft.TextStyle(
                weight=ft.FontWeight.BOLD, size=14, color=Color[self.category_idx].hex
            ),
            text_align=ft.TextAlign.LEFT,
            data=data,
        )

    def canvas_stroke(self, scale=1.0, left=0, top=0, data=None):
        from flet import canvas

        from .flet import stroke_paint

        paint = stroke_paint(color_idx=self.category_idx)
        path = canvas.Path(elements=[], paint=paint)
        if self.points:
            path.elements.append(
                canvas.Path.MoveTo(
                    self.points[0].x * scale + left, self.points[0].y * scale + top
                )
            )
            for p in self.points[1:]:
                path.elements.append(
                    canvas.Path.LineTo(p.x * scale + left, p.y * scale + top)
                )
            path.elements.append(canvas.Path.Close())
        return path

    def canvas_fill(self, scale=1.0, left=0, top=0, data=None):
        from flet import canvas

        from .flet import fill_paint

        paint = fill_paint(color_idx=self.category_idx)
        path = canvas.Path(elements=[], paint=paint)
        if self.points:
            path.elements.append(
                canvas.Path.MoveTo(
                    self.points[0].x * scale + left, self.points[0].y * scale + top
                )
            )
            for p in self.points[1:]:
                path.elements.append(
                    canvas.Path.LineTo(p.x * scale + left, p.y * scale + top)
                )
            path.elements.append(canvas.Path.Close())
        return path


# ============================================================
# 线段
# ============================================================


@dataclass
class Line(_GeometryBounds):
    """由起点与终点确定的线段（自动规范为从左上指向右下）。

    坐标使用图像坐标系（y 轴向下），因此 ``angle`` 的顺时针方向为正。
    """

    start: Point = field(default_factory=Point)
    end: Point = field(default_factory=Point)
    category_idx: int = 0
    conf: float = -1.0
    create_time: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        self.sort_point()

    # --- 顶点列表 ---
    @property
    def points(self) -> list[Point]:
        return [self.start, self.end]

    # --- 几何量 ---
    @property
    def delta_x(self) -> float:
        return self.end.x - self.start.x

    @property
    def delta_y(self) -> float:
        return self.end.y - self.start.y

    @property
    def length(self) -> float:
        """线段长度。"""
        return math.hypot(self.delta_x, self.delta_y)

    @property
    def angle(self) -> float:
        """与 x 轴正方向的夹角（角度制）。

        端点已规整为从左上指向右下，故取值落在 [-90, 90]；
        y 轴向下，正角度表示向右下方倾斜。
        """
        return math.degrees(math.atan2(self.delta_y, self.delta_x))

    @property
    def slope(self) -> float:
        """斜率 dy/dx；竖直线返回 inf，水平线返回 0。"""
        return self.delta_y / self.delta_x if self.delta_x else float("inf")

    @property
    def center(self) -> Point:
        """线段中点。"""
        return Point(
            x=(self.start.x + self.end.x) / 2, y=(self.start.y + self.end.y) / 2
        )

    @property
    def x_center(self) -> float:
        return self.center.x

    @property
    def y_center(self) -> float:
        return self.center.y

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

    # --- Shapely 对象 ---
    @cached_property
    def obj(self) -> shapely.LineString:
        return shapely.LineString(
            [(self.start.x, self.start.y), (self.end.x, self.end.y)]
        )

    # --- 打印表示 ---
    def __repr__(self) -> str:
        return f"Line({self.start!r} -> {self.end!r}, conf={self.conf:.2f}, category_idx={self.category_idx})"

    # --- 内部排序 ---
    def sort_point(self) -> None:
        """规范端点顺序：保证 start 位于 end 的左上方。"""
        if (self.start.x, self.start.y) > (self.end.x, self.end.y):
            self.start, self.end = self.end, self.start

    # --- 操作（返回 self 以支持链式调用） ---
    def offset(self, delta_x: float = 0.0, delta_y: float = 0.0) -> Line:
        """原地平移线段并返回 self。"""
        self.start.x += delta_x
        self.start.y += delta_y
        self.end.x += delta_x
        self.end.y += delta_y
        self.__dict__.pop("obj", None)
        return self

    def scale(self, scale_x: float = 1.0, scale_y: float = 1.0) -> Line:
        """原地缩放线段（相对坐标原点）并返回 self。"""
        self.start.x *= scale_x
        self.start.y *= scale_y
        self.end.x *= scale_x
        self.end.y *= scale_y
        self.__dict__.pop("obj", None)
        return self

    # --- 距离 ---
    def distance(self, obj: Point | Box | Polygon | Line | Circle) -> float:
        """到另一个几何对象的最短距离（相交时为 0）。"""
        return shapely.distance(self.obj, obj.obj)

    def is_in_obj(self, obj: Point | Box | Polygon | Line | Circle) -> bool:
        """判断整条线段是否落在 ``obj`` 的外接框内。"""
        return (
            obj.x_min <= self.x_min
            and self.x_max <= obj.x_max
            and obj.y_min <= self.y_min
            and self.y_max <= obj.y_max
        )

    # --- 格式转换 ---
    def load_yolo_txt_line(
        self, img_width: int, img_height: int, line_str: str
    ) -> None:
        """解析 ``类别 x1 y1 x2 y2 [conf]`` 格式的归一化文本行。"""
        parts = [float(v) for v in line_str.split()]
        self.category_idx = int(parts[0])
        self.start = Point(x=parts[1] * img_width, y=parts[2] * img_height)
        self.end = Point(x=parts[3] * img_width, y=parts[4] * img_height)
        if len(parts) > 5:
            self.conf = parts[5]
        self.sort_point()

    def to_str(self, img_size: ImgSize, with_conf: bool = False) -> str:
        line = (
            f"{self.category_idx} "
            f"{self.start.x / img_size.width} "
            f"{self.start.y / img_size.height} "
            f"{self.end.x / img_size.width} "
            f"{self.end.y / img_size.height}"
        )
        return f"{line} {self.conf}\n" if with_conf else f"{line}\n"

    def to_json(self) -> dict:
        return {
            "x1": int(self.start.x),
            "y1": int(self.start.y),
            "x2": int(self.end.x),
            "y2": int(self.end.y),
            "category_idx": self.category_idx,
            "conf": f"{self.conf:.3f}",
        }

    def to_sql_model(self, img, session):
        """转为分割表实体：线段以 2 个点存储。

        注意：``TSegmentAnno.anno`` 会按多边形还原，线段因此被还原为退化多边形。
        """
        import pickle

        from .entity_model import TCategory, TSegmentAnno, select

        t_category = session.exec(
            select(TCategory).where(TCategory.idx == self.category_idx)
        ).one()
        return TSegmentAnno(
            points=pickle.dumps(self.points),
            conf=self.conf,
            img=img,
            category=t_category,
        )

    # --- Flet Canvas 绘制（线段无填充，仅提供文字与描边） ---
    def canvas_text(
        self, scale=1.0, left=0, top=0, category: Category | None = None, data=None
    ):
        import flet as ft
        from flet import canvas

        value = category[self.category_idx] if category else "category"
        return canvas.Text(
            self.x_min * scale + left,
            self.y_min * scale + top,
            value,
            ft.TextStyle(
                weight=ft.FontWeight.BOLD, size=14, color=Color[self.category_idx].hex
            ),
            text_align=ft.TextAlign.LEFT,
            data=data,
        )

    def canvas_stroke(self, scale=1.0, left=0, top=0, data=None, stroke_width=1):
        from flet import canvas

        from .flet import stroke_paint

        return canvas.Line(
            self.start.x * scale + left,
            self.start.y * scale + top,
            self.end.x * scale + left,
            self.end.y * scale + top,
            paint=stroke_paint(color_idx=self.category_idx, stroke_width=stroke_width),
            data=data,
        )


# ============================================================
# 圆
# ============================================================


@dataclass
class Circle(_GeometryBounds):
    """由圆心与半径确定的圆。

    ``center`` 自身的 ``category_idx`` / ``conf`` 不参与圆的语义，
    类别与置信度以 ``Circle`` 上的字段为准。
    """

    center: Point = field(default_factory=Point)
    radius: float = 0.0
    category_idx: int = 0
    conf: float = -1.0
    create_time: datetime = field(default_factory=datetime.now)

    # --- 缓存失效（圆心或半径变化后需重算） ---
    def _invalidate_cache(self) -> None:
        for key in ("points", "obj"):
            self.__dict__.pop(key, None)

    # --- 为提升性能而重写 Mixin 的边界计算 ---
    @property
    @override
    def x_min(self) -> float:
        return self.center.x - self.radius

    @property
    @override
    def y_min(self) -> float:
        return self.center.y - self.radius

    @property
    @override
    def x_max(self) -> float:
        return self.center.x + self.radius

    @property
    @override
    def y_max(self) -> float:
        return self.center.y + self.radius

    @property
    @override
    def width(self) -> float:
        return 2 * self.radius

    @property
    @override
    def height(self) -> float:
        return 2 * self.radius

    @property
    def x_center(self) -> float:
        return self.center.x

    @property
    def y_center(self) -> float:
        return self.center.y

    # --- 几何量 ---
    @property
    def diameter(self) -> float:
        return self.radius * 2

    @property
    def area(self) -> float:
        """圆面积 πr²。"""
        return math.pi * self.radius**2

    @property
    def circumference(self) -> float:
        """圆周长 2πr。"""
        return 2 * math.pi * self.radius

    # --- 圆周采样点 / Shapely 对象 ---
    @cached_property
    def points(self) -> list[Point]:
        """圆周采样点（默认 64 个，自 x 轴正方向按顺时针排列）。"""
        n = CIRCLE_SAMPLE_NUM
        cx, cy, r = self.center.x, self.center.y, self.radius
        return [
            Point(
                x=cx + r * math.cos(2 * math.pi * i / n),
                y=cy + r * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]

    @cached_property
    def obj(self) -> shapely.Polygon:
        return shapely.Point(self.center.x, self.center.y).buffer(
            self.radius, quad_segs=CIRCLE_SAMPLE_NUM // 4
        )

    # --- 打印表示 ---
    def __repr__(self) -> str:
        return f"Circle(center={self.center!r}, r={self.radius}, conf={self.conf:.2f}, category_idx={self.category_idx})"

    # --- 操作（返回 self 以支持链式调用） ---
    def offset(self, delta_x: float = 0.0, delta_y: float = 0.0) -> Circle:
        """原地平移圆并返回 self。"""
        self.center.x += delta_x
        self.center.y += delta_y
        self._invalidate_cache()
        return self

    def scale(self, scale_x: float = 1.0, scale_y: float = 1.0) -> Circle:
        """原地缩放圆并返回 self。

        圆心按 (scale_x, scale_y) 缩放；半径取两者的平均值 ——
        严格意义下非等比缩放会把圆变成椭圆，此处以均值近似（等比缩放时精确）。
        """
        self.center.x *= scale_x
        self.center.y *= scale_y
        self.radius *= (scale_x + scale_y) / 2
        self._invalidate_cache()
        return self

    # --- 距离与包含判断 ---
    def distance(self, obj: Point | Box | Polygon | Line | Circle) -> float:
        """到另一个几何对象的最短距离（相交/内含时为 0）。"""
        return shapely.distance(self.obj, obj.obj)

    def distance_to_boundary(self, point: Point) -> float:
        """点到圆周的距离（点在圆内时同样返回正值）。"""
        return abs(self.center.distance(point) - self.radius)

    def contains(self, point: Point, tolerance: float = 1e-9) -> bool:
        """判断点是否位于圆内或圆周上。"""
        return self.center.distance(point) <= self.radius + tolerance

    # --- 格式转换 ---
    def load_yolo_txt_line(
        self, img_width: int, img_height: int, line_str: str
    ) -> None:
        """解析 ``类别 cx cy r [conf]`` 格式的归一化文本行。

        半径按短边归一化，以保证非正方形图像上的圆仍是圆。
        """
        parts = [float(v) for v in line_str.split()]
        self.category_idx = int(parts[0])
        self.center = Point(x=parts[1] * img_width, y=parts[2] * img_height)
        self.radius = parts[3] * min(img_width, img_height)
        if len(parts) > 4:
            self.conf = parts[4]
        self._invalidate_cache()

    def to_str(self, img_size: ImgSize, with_conf: bool = False) -> str:
        line = (
            f"{self.category_idx} "
            f"{self.center.x / img_size.width} "
            f"{self.center.y / img_size.height} "
            f"{self.radius / min(img_size.width, img_size.height)}"
        )
        return f"{line} {self.conf}\n" if with_conf else f"{line}\n"

    def to_json(self) -> dict:
        return {
            "cx": int(self.center.x),
            "cy": int(self.center.y),
            "r": int(self.radius),
            "category_idx": self.category_idx,
            "conf": f"{self.conf:.3f}",
        }

    def to_sql_model(self, img, session):
        """转为分割表实体：圆以圆周采样点存储（等价于高精度多边形）。

        由于采样点包含上下左右四个极点，圆心与半径可由其外接框精确还原。
        """
        import pickle

        from .entity_model import TCategory, TSegmentAnno, select

        t_category = session.exec(
            select(TCategory).where(TCategory.idx == self.category_idx)
        ).one()
        return TSegmentAnno(
            points=pickle.dumps(self.points),
            conf=self.conf,
            img=img,
            category=t_category,
        )

    # --- Flet Canvas 绘制 ---
    def canvas_text(
        self, scale=1.0, left=0, top=0, category: Category | None = None, data=None
    ):
        import flet as ft
        from flet import canvas

        value = category[self.category_idx] if category else "category"
        return canvas.Text(
            self.x_min * scale + left,
            self.y_min * scale + top,
            value,
            ft.TextStyle(
                weight=ft.FontWeight.BOLD, size=14, color=Color[self.category_idx].hex
            ),
            text_align=ft.TextAlign.LEFT,
            data=data,
        )

    def canvas_stroke(self, scale=1.0, left=0, top=0, data=None, stroke_width=1):
        from flet import canvas

        from .flet import stroke_paint

        return canvas.Circle(
            self.center.x * scale + left,
            self.center.y * scale + top,
            radius=self.radius * scale,
            paint=stroke_paint(color_idx=self.category_idx, stroke_width=stroke_width),
            data=data,
        )

    def canvas_fill(self, scale=1.0, left=0, top=0, data=None):
        from flet import canvas

        from .flet import fill_paint

        return canvas.Circle(
            self.center.x * scale + left,
            self.center.y * scale + top,
            radius=self.radius * scale,
            paint=fill_paint(color_idx=self.category_idx),
            data=data,
        )


# ============================================================
# AnnoImg —— 单张图像的标注
# ============================================================


@dataclass
class AnnoImg:
    """单张图像的标注集合。"""

    file: Path | None = None
    objs: list[Point | Box | Polygon | Line | Circle] = field(default_factory=list)
    img_size: ImgSize = field(default_factory=lambda: ImgSize(-1, -1))
    task_type: TaskType = TaskType.DETECT

    def load(self, file: Path | None = None) -> None:
        """从 YOLO 格式的标注文件加载标注对象。"""
        if file is not None:
            self.file = file
        if self.file is None:
            return
        content = self.file.read_text(encoding="utf-8")
        for line in content.splitlines():
            if not line.strip():
                continue
            match self.task_type:
                case TaskType.SEGMENT:
                    obj: Box | Polygon = Polygon()
                case _:
                    obj = Box()
            obj.load_yolo_txt_line(
                img_width=self.img_size.width,
                img_height=self.img_size.height,
                line_str=line,
            )
            self.objs.append(obj)

    def to_str(self) -> str:
        """导出为 YOLO 格式的标注字符串。"""
        lines = [
            obj.to_str(self.img_size)
            for obj in self.objs
            if obj.validate(self.img_size)
        ]
        return "\n".join(l.strip() for l in lines)

    def save_txt(self, file_path: Path) -> None:
        file_path.parent.mkdir(exist_ok=True, parents=True)
        file_path.write_text(self.to_str(), encoding="utf-8")

    def filter_(self) -> list:
        """过滤重复框（目前为占位实现：仅做排序）。"""
        self.objs.sort(key=lambda o: (o.x_min, o.y_min))
        return self.objs

    # --- 所有对象的外接框 ---
    @property
    def x_min(self) -> float:
        return min((o.x_min for o in self.objs), default=-1)

    @property
    def y_min(self) -> float:
        return min((o.y_min for o in self.objs), default=-1)

    @property
    def x_max(self) -> float:
        return max((o.x_max for o in self.objs), default=-1)

    @property
    def y_max(self) -> float:
        return max((o.y_max for o in self.objs), default=-1)


# ============================================================
# AnnoDataset —— 数据集级标注
# ============================================================


@dataclass
class AnnoDataset:
    """关联图像目录与标签目录的标注数据集。"""

    img_dir: Path
    anno_dir: Path
    category: Category
    img_paths: list[Path] = field(init=False, default_factory=list)
    anno_paths: list[Path] = field(init=False, default_factory=list)
    anno_dict: dict[str, AnnoImg | None] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.anno_paths = [p for p in self.anno_dir.iterdir() if p.suffix == ".txt"]
        self.anno_dict = self._load_anno_dict()

    def _load_anno_dict(self) -> dict[str, AnnoImg | None]:
        from .img import Img

        result: dict[str, AnnoImg | None] = {}
        for file in self.img_dir.iterdir():
            self.img_paths.append(file)
            anno_file = self.anno_dir / f"{file.stem}.txt"
            result[file.name] = (
                AnnoImg(file=anno_file, img_size=Img.get_img_size(file))
                if anno_file.exists()
                else None
            )
        return result

    @property
    def objs(self) -> list:
        objs: list = []
        for anno in self.anno_dict.values():
            if anno:
                objs.extend(anno.objs)
        return objs

    def save_yolo(self, yolo_root_dir: Path) -> None:
        (yolo_root_dir / "images").mkdir(exist_ok=True, parents=True)
        (yolo_root_dir / "labels").mkdir(exist_ok=True, parents=True)

    @property
    def show_msg_str(self) -> str:
        img_n = len(self.img_paths)
        anno_n = len(self.anno_paths)
        obj_n = len(self.objs)
        return (
            f"\n        - Image count: `{img_n}`\n"
            f"        - Annotation file count: `{anno_n}`\n"
            f"        - Annotation object count: `{obj_n}`"
        )

    def compare(self, other: AnnoDataset) -> None:
        gt_total = len(self.objs)
        if gt_total == 0:
            print("    Ground-truth object count is 0, cannot compare.")
            return

        right = missed = 0
        for img_name, anno in self.anno_dict.items():
            if anno is None:
                continue
            for obj in anno.objs:
                pred_anno = other.anno_dict.get(img_name)
                if pred_anno and any(
                    (isinstance(obj, Box) and isinstance(p, Box) and obj.iou(p) >= 0.45)
                    for p in pred_anno.objs
                ):
                    right += 1
                else:
                    missed += 1

        print(
            f"\n"
            f"    GT objects: {gt_total}\n"
            f"    Predicted objects: {len(other.objs)}\n"
            f"    Detected (in GT): {right}\n"
            f"    Missed (in GT): {missed}\n"
            f"    Recall: {right / gt_total:.2%}\n"
        )
