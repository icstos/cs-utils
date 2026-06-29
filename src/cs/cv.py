"""Computer vision annotation data structures.

Provides Point, Box, Polygon, Category, AnnoImg, and AnnoDataset types.
"""

from __future__ import annotations

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


# ============================================================
# Category
# ============================================================


@dataclass
class Category:
    """Category name list, supports lookup by index or name."""

    names: list[str] = field(default_factory=list)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def __getitem__(self, idx: int) -> str:
        return self.names[idx]

    def __len__(self) -> int:
        return len(self.names)


# ============================================================
# Geometry utility Mixin
# ============================================================


class _GeometryBounds:
    """Mixin providing geometry bounds computation from ``points``."""

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
        """Clamp object within image boundaries."""
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
# Point
# ============================================================


@dataclass
class Point(_GeometryBounds):
    """2D point with category index, confidence, and timestamp."""

    x: float = -1.0
    y: float = -1.0
    category_idx: int = 0
    conf: float = -1.0
    create_time: datetime = field(default_factory=datetime.now)

    # --- Override Mixin bounds for performance ---
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

    # --- Fast int conversion ---
    @property
    def x_int(self) -> int:
        return int(self.x)

    @property
    def y_int(self) -> int:
        return int(self.y)

    # --- Representation ---
    def __repr__(self) -> str:
        return f"({self.x},{self.y})"

    # --- Lazy geometry objects ---
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

    # --- Conversion ---
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

    # --- Operations (chainable) ---
    def offset(self, delta_x: float = 0.0, delta_y: float = 0.0) -> Point:
        """Return a new Point with the given offset."""
        return replace(self, x=self.x + delta_x, y=self.y + delta_y)

    def scale(self, scale_x: float = 1.0, scale_y: float = 1.0) -> Point:
        """Return a new Point with coordinates scaled."""
        return replace(self, x=self.x * scale_x, y=self.y * scale_y)

    def distance(self, point: Point) -> float:
        return shapely.distance(self.obj, point.obj)

    def is_in_obj(self, obj: Point | Box | Polygon) -> bool:
        """Check whether this point lies inside the bounding box of ``obj``."""
        return obj.x_min <= self.x <= obj.x_max and obj.y_min <= self.y <= obj.y_max

    # --- Flet Canvas rendering ---
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
# Box
# ============================================================


@dataclass
class Box(_GeometryBounds):
    """Axis-aligned bounding box defined by two diagonal corners
    (auto-sorted to top-left → bottom-right).
    """

    start: Point = field(default_factory=Point)
    end: Point = field(default_factory=Point)
    category_idx: int = 0
    conf: float = -1.0
    create_time: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        self.sort_point()

    # --- Override Mixin bounds for performance ---
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

    # --- Integer shortcuts ---
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

    # --- Area & axis ratio ---
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

    # --- XYXY / IOU data ---
    @property
    def xyxy(self) -> list[float]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    @property
    def iou_data(self) -> list[float]:
        return self.xyxy

    # --- Point list / Shapely object ---
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

    # --- Representation ---
    def __repr__(self) -> str:
        return f"[{self.x_min_int}, {self.y_min_int}, {self.x_max_int}, {self.y_max_int}, {self.conf:.2f}, {self.category_idx}]"

    # --- Internal sorting ---
    def sort_point(self) -> None:
        if self.start.x > self.end.x:
            self.start.x, self.end.x = self.end.x, self.start.x
        if self.start.y > self.end.y:
            self.start.y, self.end.y = self.end.y, self.start.y

    # --- Operations (returns Self for chaining) ---
    def offset(self, delta_x: float = 0.0, delta_y: float = 0.0) -> Box:
        """Translate the box in-place and return self."""
        self.start.x += delta_x
        self.start.y += delta_y
        self.end.x += delta_x
        self.end.y += delta_y
        return self

    def scale(self, scale_x: float = 1.0, scale_y: float = 1.0) -> Box:
        """Scale the box in-place and return self."""
        self.start.x *= scale_x
        self.start.y *= scale_y
        self.end.x *= scale_x
        self.end.y *= scale_y
        return self

    # --- Distance & IOU ---
    def distance(self, box: Box) -> float:
        return shapely.distance(self.obj, box.obj)

    def iou(self, box: Box) -> float:
        """Compute Intersection over Union."""
        left = max(self.x_min, box.x_min)
        top = max(self.y_min, box.y_min)
        right = min(self.x_max, box.x_max)
        bottom = min(self.y_max, box.y_max)
        cross = max(right - left, 0) * max(bottom - top, 0)
        union = self.area + box.area - cross
        return cross / union if cross > 0 and union > 0 else 0.0

    def iou_max(self, box: Box) -> float:
        """Compute intersection / max(self.area, box.area)."""
        left = max(self.x_min, box.x_min)
        top = max(self.y_min, box.y_min)
        right = min(self.x_max, box.x_max)
        bottom = min(self.y_max, box.y_max)
        cross = max(right - left, 0) * max(bottom - top, 0)
        if cross <= 0 or self.area <= 0 or box.area <= 0:
            return 0.0
        return max(cross / self.area, cross / box.area)

    # --- Crop region ---
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
        """Return a statistic from the region (currently max-grayscale only)."""
        roi = img[int(self.y_min) : int(self.y_max), int(self.x_min) : int(self.x_max)]
        return cv2.minMaxLoc(roi)[1] if roi.size else None

    # --- Grayscale statistics (torch / np) ---
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

    # --- Conversion ---
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

    # --- Static utilities ---
    @staticmethod
    def bounding_box(objs: list[Point | Box | Polygon]) -> list[int]:
        return [
            int(f([getattr(o, attr) for o in objs]))
            for f, attr in [
                (min, "x_min"),
                (min, "y_min"),
                (max, "x_max"),
                (max, "y_max"),
            ]
        ]

    # --- Flet Canvas rendering ---
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
# Polygon
# ============================================================


@dataclass
class Polygon(_GeometryBounds):
    """Polygon annotation defined by a list of vertices."""

    points: list[Point] = field(default_factory=list)
    category_idx: int = 0
    conf: float = -1.0
    create_time: datetime = field(default_factory=datetime.now)

    # --- Shapely ---
    @cached_property
    def obj(self) -> shapely.Polygon:
        return shapely.Polygon([(p.x, p.y) for p in self.points])

    # --- Area (shoelace formula) ---
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

    # --- Conversion ---
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

    # --- Operations (returns Self) ---
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
        """Placeholder — actual drawing is handled by ``cv_draw`` module."""
        pass

    # --- Flet Canvas rendering ---
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
# AnnoImg — single image annotation
# ============================================================


@dataclass
class AnnoImg:
    """Annotation collection for a single image."""

    file: Path | None = None
    objs: list[Point | Box | Polygon] = field(default_factory=list)
    img_size: ImgSize = field(default_factory=lambda: ImgSize(-1, -1))
    task_type: TaskType = TaskType.DETECT

    def load(self, file: Path | None = None) -> None:
        """Load objects from a YOLO-format annotation file."""
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
        """Export to YOLO-format annotation string."""
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
        """Filter duplicate boxes (currently placeholder: sort only)."""
        self.objs.sort(key=lambda o: (o.x_min, o.y_min))
        return self.objs

    # --- Bounding box of all objects ---
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
# AnnoDataset — dataset-level annotation
# ============================================================


@dataclass
class AnnoDataset:
    """Annotation dataset linking an image directory with a label directory."""

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
