"""图像绘制核心：图像读写、颜色解析与 :class:`Canvas` 绘制上下文。

约定：

- 图像一律使用 OpenCV 的 BGR 顺序；
- :class:`Canvas` 默认在**副本**上绘制（不修改调用方数组），``inplace=True`` 时直接改原图；
- 灰度 / 带 alpha 的输入会先转成 BGR —— 灰度图无法原地着色，此时 ``inplace`` 退化为拷贝；
- 线宽、字号、颜色都支持“自适应”，不必每次手写数字。
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Self

import cv2
import numpy as np

from ..color import Color
from ..cv import Box, Category, Circle, Line, Point, Polygon
from .errors import ImageError, ShapeError
from .models import Anno, ColorLike, Shape, Stroke, Style, Text

__all__ = [
    "Canvas",
    "ImageInfo",
    "image_info",
    "imread",
    "imread_gray",
    "imwrite",
    "render",
    "resolve_color",
    "to_bgr",
]

type StrPath = str | Path

# JPEG 需要显式质量参数，其余格式交给编解码器默认值
_JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})

# 按优先级查找可显示中文的字体
_FONT_CANDIDATES = (
    "simhei.ttf",
    "msyh.ttc",
    "msyh.ttf",
    "PingFang.ttc",
    "NotoSansCJK-Regular.ttc",
    "DejaVuSans.ttf",
)


# ============================================================
# 图像读写
# ============================================================


@dataclass(slots=True, frozen=True)
class ImageInfo:
    """图像基本信息。"""

    path: Path
    width: int
    height: int
    channels: int
    dtype: str

    @property
    def is_gray(self) -> bool:
        return self.channels == 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "dtype": self.dtype,
            "gray": self.is_gray,
        }


def imread(source: StrPath, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """读取图像，兼容 Windows 下的中文路径（``cv2.imread`` 不支持）。"""
    path = Path(source)
    if not path.is_file():
        raise ImageError(f"图像不存在：{path}")
    buffer = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(buffer, flags)
    if image is None:
        raise ImageError(f"无法解码图像（格式不支持或文件损坏）：{path}")
    return image


def imwrite(
    target: StrPath,
    image: np.ndarray,
    *,
    overwrite: bool = False,
    quality: int = 95,
) -> Path:
    """写出图像，默认不覆盖已存在文件；同样规避中文路径问题。"""
    path = Path(target)
    if path.exists() and not overwrite:
        raise ImageError(f"文件已存在（需显式 overwrite）：{path}")
    suffix = path.suffix.lower() or ".png"
    path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if suffix in _JPEG_SUFFIXES else []
    ok, buffer = cv2.imencode(suffix, image, params)
    if not ok:
        raise ImageError(f"无法编码为 {suffix}：{path}")
    buffer.tofile(path)
    return path


def image_info(source: StrPath) -> ImageInfo:
    """读取图像尺寸、通道数与 dtype（不修改文件）。"""
    image = imread(source, flags=cv2.IMREAD_UNCHANGED)
    height, width = image.shape[:2]
    return ImageInfo(
        path=Path(source),
        width=width,
        height=height,
        channels=1 if image.ndim == 2 else image.shape[2],
        dtype=str(image.dtype),
    )


def imread_gray(source: StrPath) -> np.ndarray:
    """以单通道方式读取图像，常用于掩码。"""
    return imread(source, flags=cv2.IMREAD_GRAYSCALE)


def to_bgr(image: np.ndarray) -> np.ndarray:
    """统一成 3 通道 BGR；已是 BGR 时原样返回（不拷贝）。"""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ImageError(f"不支持的图像维度：{image.ndim}")
    match image.shape[2]:
        case 3:
            return image
        case 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        case channels:
            raise ImageError(f"不支持的通道数：{channels}")


# ============================================================
# 颜色与文字
# ============================================================


def resolve_color(
    color: ColorLike | None, category_idx: int = 0
) -> tuple[int, int, int]:
    """把多种颜色写法解析为 OpenCV 使用的 BGR 三元组。"""
    match color:
        case None:
            rgb = Color[category_idx % len(Color)].rgb
        case int() as idx:
            rgb = Color[idx % len(Color)].rgb
        case str() as text:
            rgb = Color.hex2rgb(text)
        case (r, g, b):
            rgb = (int(r), int(g), int(b))
        case _:
            raise ShapeError(f"无法识别的颜色：{color!r}")
    r, g, b = rgb
    return (int(b), int(g), int(r))


@dataclass(slots=True, frozen=True)
class _Font:
    """文字度量与绘制：ASCII 走 cv2（快），含非 ASCII 走 Pillow（支持中文）。"""

    size: int
    scale: float
    thickness: int
    face: Any = None

    def measure(self, text: str) -> tuple[int, int]:
        """返回文字外接框的 (宽, 高)。"""
        if text.isascii():
            (width, height), base = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, self.scale, self.thickness
            )
            return width, height + base
        if self.face is None:
            return len(text) * self.size // 2, self.size
        left, top, right, bottom = self.face.getbbox(text)
        return right - left, bottom - top

    def draw(
        self,
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        """在 ``origin``（左上角）处绘制文字，越界部分自动裁掉。"""
        x, y = origin
        if text.isascii():
            height = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, self.scale, self.thickness
            )[0][1]
            cv2.putText(
                image,
                text,
                (x, y + height),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.scale,
                color,
                self.thickness,
                cv2.LINE_AA,
            )
            return

        from PIL import Image, ImageDraw

        if self.face is None:
            raise ShapeError(
                f"未找到可显示非 ASCII 的字体（已尝试 {', '.join(_FONT_CANDIDATES)}）"
            )
        width, height = self.measure(text)
        x2, y2 = min(x + width + 2, image.shape[1]), min(y + height + 4, image.shape[0])
        if x2 <= x or y2 <= y:
            return
        roi = image[y:y2, x:x2]
        pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil).text(
            (0, 0), text, font=self.face, fill=(color[2], color[1], color[0])
        )
        image[y:y2, x:x2] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


@functools.lru_cache(maxsize=16)
def _font(size: int) -> _Font:
    """按字号加载字体；找不到中文字体时退化为 cv2 默认字体。"""
    from PIL import ImageFont

    face = next(
        (
            loaded
            for name in _FONT_CANDIDATES
            if (loaded := _try_font(ImageFont, name, size)) is not None
        ),
        None,
    )
    return _Font(size=size, scale=size / 32, thickness=max(1, size // 16), face=face)


def _try_font(module: Any, name: str, size: int) -> Any:
    try:
        return module.truetype(name, size)
    except OSError:
        return None


def _fit(
    origin: tuple[int, int], width: int, height: int, bounds: tuple[int, int]
) -> tuple[int, int]:
    """把文字左上角收进图像范围内，尽量不画出界。"""
    max_x, max_y = bounds
    return (
        min(max(int(origin[0]), 0), max(max_x - width, 0)),
        min(max(int(origin[1]), 0), max(max_y - height, 0)),
    )


# ============================================================
# 绘制上下文
# ============================================================


@dataclass(slots=True)
class Canvas:
    """图像绘制上下文：一次构造，链式绘制，最后取图或存盘。

    >>> Canvas(imread("in.jpg")).box(box).text("标题", Point(10, 30)).save("out.jpg")
    >>> Canvas(img, inplace=True).circle(Circle(Point(50, 50), 20))  # 直接改原图
    """

    image: np.ndarray
    inplace: bool = False
    style: Style = field(default_factory=Style)
    category: Category | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image, np.ndarray) or self.image.size == 0:
            raise ImageError("image 必须是非空 numpy 数组")
        if self.image.dtype != np.uint8:
            raise ImageError(f"仅支持 uint8 图像，收到 {self.image.dtype}")
        converted = to_bgr(self.image)
        if converted is self.image:
            # 已是 BGR：拷贝语义下复制一份，原图语义下直接沿用
            self.image = self.image if self.inplace else self.image.copy()
        else:
            # 灰度 / BGRA 必须换数组，此时无法原地着色
            self.image = converted

    # ---------------------------------------------------------- 基础属性
    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    def result(self) -> np.ndarray:
        """返回绘制结果（``inplace=True`` 时即原数组）。"""
        return self.image

    def save(
        self, target: StrPath, *, overwrite: bool = False, quality: int = 95
    ) -> Path:
        """把当前结果写到磁盘。"""
        return imwrite(target, self.image, overwrite=overwrite, quality=quality)

    # ---------------------------------------------------------- 内部工具
    def _resolve(self, style: Style | None) -> Style:
        return style or self.style

    def _auto_thickness(self) -> int:
        return max(1, round(0.002 * (self.width + self.height) / 2) + 1)

    def _auto_font(self, style: Style) -> int:
        return max(14, (style.thickness or self._auto_thickness()) * 6)

    def _color(self, style: Style, category_idx: int) -> tuple[int, int, int]:
        return resolve_color(style.color, category_idx)

    def _text_of(self, category_idx: int, conf: float, label: str | None) -> str:
        """显式标签优先，其次“类别名: 置信度”，再次只有置信度。"""
        if label is not None:
            return label
        name = ""
        if self.category is not None and 0 <= category_idx < len(self.category):
            name = self.category[category_idx]
        if conf < 0:
            return name
        return f"{name}: {conf:.2f}" if name else f"{conf:.2f}"

    def _put_text(
        self,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        size: int,
        *,
        background: bool = False,
    ) -> None:
        if not text:
            return
        font = _font(size)
        width, height = font.measure(text)
        x, y = _fit(origin, width, height, (self.width, self.height))
        if background:
            cv2.rectangle(
                self.image, (x - 3, y - 3), (x + width + 3, y + height + 3), color, -1
            )
            color = (255, 255, 255)
        font.draw(self.image, text, (x, y), color)

    def _label_at(
        self,
        text: str,
        anchor: tuple[int, int],
        color: tuple[int, int, int],
        style: Style,
        *,
        above: bool = True,
    ) -> None:
        """在 ``anchor`` 上方（或下方）绘制标签。"""
        if not text or not style.label:
            return
        size = style.font_size or self._auto_font(style)
        _, height = _font(size).measure(text)
        x, y = anchor
        self._put_text(
            text,
            (x, y - height - 2 if above else y + 2),
            color,
            size,
            background=style.label_bg,
        )

    def _blend(
        self,
        bbox: tuple[int, int, int, int],
        paint: Callable[[np.ndarray, int, int], None],
        alpha: float,
    ) -> None:
        """只在 bbox 内做半透明混合，框外像素保持原样。"""
        x, y, width, height = bbox
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + width, self.width), min(y + height, self.height)
        if x2 <= x1 or y2 <= y1:
            return
        roi = self.image[y1:y2, x1:x2]
        overlay = roi.copy()
        paint(overlay, x1, y1)
        self.image[y1:y2, x1:x2] = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)

    def _fill_poly(self, points: np.ndarray, color, style: Style) -> None:
        if not style.fill or len(points) < 3:
            return
        self._blend(
            cv2.boundingRect(points),
            lambda roi, ox, oy: cv2.fillPoly(
                roi, [points], color, cv2.LINE_AA, 0, (-ox, -oy)
            ),
            style.alpha,
        )

    def _fill_circle(
        self, center: tuple[int, int], radius: int, color, style: Style
    ) -> None:
        if not style.fill:
            return
        cx, cy = center
        self._blend(
            (cx - radius, cy - radius, radius * 2, radius * 2),
            lambda roi, ox, oy: cv2.circle(
                roi, (cx - ox, cy - oy), radius, color, -1, cv2.LINE_AA
            ),
            style.alpha,
        )

    def _stroke_path(
        self, points: Sequence[Point], color, style: Style, *, closed: bool = False
    ) -> None:
        """按线型描边折线；虚线与点线沿路径等距采样后绘制。"""
        if len(points) < 2:
            return
        thickness = style.thickness or self._auto_thickness()
        match style.stroke:
            case Stroke.SOLID:
                polyline = np.array([[p.x_int, p.y_int] for p in points], np.int32)
                cv2.polylines(
                    self.image, [polyline], closed, color, thickness, cv2.LINE_AA
                )
            case Stroke.DASHED:
                samples = _sample(points, closed, max(thickness * 4, 4))
                for start, end in pairwise(samples):
                    cv2.line(self.image, start, end, color, thickness, cv2.LINE_AA)
            case Stroke.DOTTED:
                samples = _sample(points, closed, max(thickness * 3, 3))
                for start, _ in pairwise(samples):
                    cv2.circle(self.image, start, thickness, color, -1, cv2.LINE_AA)

    # ---------------------------------------------------------- 图元
    def point(
        self, point: Point, *, style: Style | None = None, label: str | None = None
    ) -> Self:
        """绘制点（默认实心圆点，``fill=False`` 时画空心圈）。"""
        style = self._resolve(style)
        color = self._color(style, point.category_idx)
        radius = style.thickness or 4
        cv2.circle(
            self.image,
            (point.x_int, point.y_int),
            radius,
            color,
            -1 if style.fill else max(1, radius // 2),
            cv2.LINE_AA,
        )
        self._label_at(
            self._text_of(point.category_idx, point.conf, label),
            (point.x_int + radius + 4, point.y_int - radius),
            color,
            style,
            above=False,
        )
        return self

    def points(
        self,
        points: Iterable[Point],
        *,
        style: Style | None = None,
        label: str | None = None,
    ) -> Self:
        for item in points:
            self.point(item, style=style, label=label)
        return self

    def line(
        self, line: Line, *, style: Style | None = None, label: str | None = None
    ) -> Self:
        """绘制线段（默认实线，可改用 dashed/dotted）。"""
        style = self._resolve(style)
        color = self._color(style, line.category_idx)
        self._stroke_path([line.start, line.end], color, style)
        self._label_at(
            self._text_of(line.category_idx, line.conf, label),
            (int(line.x_min), int(line.y_min)),
            color,
            style,
        )
        return self

    def polyline(
        self,
        points: Sequence[Point],
        *,
        closed: bool = False,
        style: Style | None = None,
        label: str | None = None,
    ) -> Self:
        """绘制任意折线；``closed=True`` 时首尾相连。"""
        style = self._resolve(style)
        color = self._color(style, 0)
        self._stroke_path(points, color, style, closed=closed)
        if label and points:
            self._label_at(label, (int(points[0].x), int(points[0].y)), color, style)
        return self

    def box(
        self, box: Box, *, style: Style | None = None, label: str | None = None
    ) -> Self:
        """绘制轴对齐矩形框，标签默认贴在框的左上角外侧。"""
        style = self._resolve(style)
        color = self._color(style, box.category_idx)
        corners = np.array(
            [
                [box.x_min_int, box.y_min_int],
                [box.x_max_int, box.y_min_int],
                [box.x_max_int, box.y_max_int],
                [box.x_min_int, box.y_max_int],
            ],
            np.int32,
        )
        self._fill_poly(corners, color, style)
        cv2.rectangle(
            self.image,
            (box.x_min_int, box.y_min_int),
            (box.x_max_int, box.y_max_int),
            color,
            style.thickness or self._auto_thickness(),
            cv2.LINE_AA,
        )
        self._label_at(
            self._text_of(box.category_idx, box.conf, label),
            (box.x_min_int, box.y_min_int),
            color,
            style,
        )
        return self

    def boxes(
        self,
        boxes: Iterable[Box],
        *,
        style: Style | None = None,
        label: str | None = None,
    ) -> Self:
        for item in boxes:
            self.box(item, style=style, label=label)
        return self

    def polygon(
        self, polygon: Polygon, *, style: Style | None = None, label: str | None = None
    ) -> Self:
        """绘制多边形（分割结果），标签贴在外接框左上角。"""
        style = self._resolve(style)
        color = self._color(style, polygon.category_idx)
        if len(polygon.points) < 2:
            return self
        points = np.array([[p.x_int, p.y_int] for p in polygon.points], np.int32)
        self._fill_poly(points, color, style)
        self._stroke_path(polygon.points, color, style, closed=True)
        self._label_at(
            self._text_of(polygon.category_idx, polygon.conf, label),
            (int(polygon.x_min), int(polygon.y_min)),
            color,
            style,
        )
        return self

    def polygons(
        self,
        polygons: Iterable[Polygon],
        *,
        style: Style | None = None,
        label: str | None = None,
    ) -> Self:
        for item in polygons:
            self.polygon(item, style=style, label=label)
        return self

    def circle(
        self, circle: Circle, *, style: Style | None = None, label: str | None = None
    ) -> Self:
        """绘制圆（可选半透明填充）。"""
        style = self._resolve(style)
        color = self._color(style, circle.category_idx)
        center = (circle.center.x_int, circle.center.y_int)
        radius = int(circle.radius)
        self._fill_circle(center, radius, color, style)
        if style.stroke is Stroke.SOLID:
            cv2.circle(
                self.image,
                center,
                radius,
                color,
                style.thickness or self._auto_thickness(),
                cv2.LINE_AA,
            )
        else:
            # 虚线/点线沿圆周采样点绘制，保证视觉效果与其它图元一致
            self._stroke_path(circle.points, color, style, closed=True)
        self._label_at(
            self._text_of(circle.category_idx, circle.conf, label),
            (int(circle.x_min), int(circle.y_min)),
            color,
            style,
        )
        return self

    def circles(
        self,
        circles: Iterable[Circle],
        *,
        style: Style | None = None,
        label: str | None = None,
    ) -> Self:
        for item in circles:
            self.circle(item, style=style, label=label)
        return self

    def text(
        self,
        text: str,
        point: Point | None = None,
        *,
        size: int = 0,
        style: Style | None = None,
    ) -> Self:
        """绘制文字；含中文时自动启用 Pillow + 系统字体。"""
        style = self._resolve(style)
        origin = point or Point()
        self._put_text(
            text,
            (origin.x_int, origin.y_int),
            self._color(style, 0),
            size or style.font_size or self._auto_font(style),
            background=style.label_bg,
        )
        return self

    # ---------------------------------------------------------- 图像级操作
    def mask(
        self,
        mask: np.ndarray,
        *,
        color: ColorLike | None = None,
        alpha: float = 0.35,
    ) -> Self:
        """把二值掩码区域半透明着色（分割/热区可视化）。"""
        gray = mask if mask.ndim == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if gray.shape != (self.height, self.width):
            gray = cv2.resize(gray, (self.width, self.height), cv2.INTER_NEAREST)
        region = gray > 0
        if not region.any():
            return self
        overlay = self.image.copy()
        overlay[region] = resolve_color(color, 0)
        blended = cv2.addWeighted(self.image, 1 - alpha, overlay, alpha, 0)
        self.image[region] = blended[region]
        return self

    def grid(
        self, cols: int = 10, rows: int = 10, *, style: Style | None = None
    ) -> Self:
        """绘制等距辅助网格。"""
        if cols < 1 or rows < 1:
            raise ShapeError("cols/rows 必须为正整数")
        style = self._resolve(style)
        color = self._color(style, 0)
        thickness = style.thickness or 1
        for index in range(1, cols):
            x = round(self.width * index / cols)
            cv2.line(
                self.image, (x, 0), (x, self.height), color, thickness, cv2.LINE_AA
            )
        for index in range(1, rows):
            y = round(self.height * index / rows)
            cv2.line(self.image, (0, y), (self.width, y), color, thickness, cv2.LINE_AA)
        return self

    # ---------------------------------------------------------- 统一入口
    def draw(
        self, shape: Shape, *, style: Style | None = None, label: str | None = None
    ) -> Self:
        """按类型分派绘制单个图元。"""
        match shape:
            case Anno(shape=inner, label=text):
                return self.draw(inner, style=style, label=text or label)
            case Text(text=content, point=origin, size=size):
                return self.text(content, origin, size=size, style=style)
            case Point():
                return self.point(shape, style=style, label=label)
            case Line():
                return self.line(shape, style=style, label=label)
            case Box():
                return self.box(shape, style=style, label=label)
            case Polygon():
                return self.polygon(shape, style=style, label=label)
            case Circle():
                return self.circle(shape, style=style, label=label)
            case _:
                raise ShapeError(f"不支持的图元类型：{type(shape).__name__}")

    def draw_all(self, shapes: Iterable[Shape], *, style: Style | None = None) -> Self:
        for shape in shapes:
            self.draw(shape, style=style)
        return self

    @classmethod
    def render(
        cls,
        image: np.ndarray,
        shapes: Iterable[Shape],
        *,
        style: Style | None = None,
        inplace: bool = False,
        category: Category | None = None,
    ) -> np.ndarray:
        """一次性绘制多个图元并返回图像。"""
        canvas = cls(image, inplace=inplace, style=style or Style(), category=category)
        canvas.draw_all(shapes)
        return canvas.image


def render(
    image: np.ndarray,
    shapes: Iterable[Shape],
    *,
    style: Style | None = None,
    inplace: bool = False,
    category: Category | None = None,
) -> np.ndarray:
    """模块级快捷方式，等价于 :meth:`Canvas.render`。"""
    return Canvas.render(image, shapes, style=style, inplace=inplace, category=category)


def _sample(
    points: Sequence[Point], closed: bool, step: float
) -> list[tuple[int, int]]:
    """沿折线按 ``step`` 步长等距采样（含拐点），用于虚线/点线。"""
    path = [*points, points[0]] if closed else list(points)
    samples: list[tuple[int, int]] = []
    for start, end in pairwise(path):
        distance = math.hypot(end.x - start.x, end.y - start.y)
        count = max(round(distance / step), 1)
        samples.extend(
            (
                round(start.x + (end.x - start.x) * index / count),
                round(start.y + (end.y - start.y) * index / count),
            )
            for index in range(count)
        )
    return samples
