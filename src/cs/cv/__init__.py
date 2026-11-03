"""计算机视觉工具集：几何图元、图像绘制、YOLO 模型推理与标注数据读写。

三种调用方式：

- 门面 :class:`Cv`（无状态，静态方法）::

    from cs.cv import Cv
    Cv.info("bus.jpg").size          # 图像：宽高/通道/dtype
    Cv.model_info("yolo11n.pt")      # 模型：任务/类别/设备/格式/参数量
    Cv.describe("yolo11n.pt")        # 按后缀自动在两者之间选择
    Cv.predict("yolo11n.pt", "bus.jpg", output_dir="out")
    Cv.draw_file("a.jpg", "b.jpg", [Box(Point(10, 10), Point(90, 60))])

- 领域对象（推荐用于脚本/服务）::

    from cs.cv import Model, Canvas, Point, Box
    with Model("yolo11n.pt") as model:          # 懒加载，退出时释放显存
        for result in model.predict("images/"):
            result.save(f"out/{result.source.stem}.jpg")
    model.train("coco8.yaml", epochs=50, log="train.log")
    model.export(ModelFormat.ONNX)

- MCP：``python -m cs.cv serve``，工具见 :mod:`cs.cv.mcp_server`。

模块划分：

======================  ====================================================
:mod:`cs.cv.geometry`   点 / 线段 / 框 / 旋转框 / 多边形 / 圆 / 姿态关键点
:mod:`cs.cv.models`     任务类型、类别表、分类概率、推理结果、图像信息
:mod:`cs.cv.draw`       :class:`Canvas` 绘制、图像读写、样式与 JSON 规格解析
:mod:`cs.cv.annotation` YOLO 标签文件与数据集扫描、统计、比对
:mod:`cs.cv.infer`      :class:`Model`：加载 / 预测 / 跟踪 / 训练 / 验证 / 导出
:mod:`cs.cv.core`       :class:`Cv` 门面
======================  ====================================================

依赖约定：``opencv-python`` / ``numpy`` / ``shapely`` 为必需依赖；
``ultralytics``（含 torch）与 ``Pillow`` 为可选依赖 —— 只在真正用到
（加载模型 / 绘制中文）时才导入，缺失时抛
:class:`~cs.cv.errors.DependencyError` 或明确的提示。
"""

from __future__ import annotations

from ..my_types import ImgSize
from . import annotation, draw, geometry, infer, models
from .annotation import (
    AnnoDataset,
    AnnoImg,
    CompareReport,
    DatasetSummary,
    image_size,
)
from .core import Cv
from .draw import (
    Anno,
    Canvas,
    ColorLike,
    Shape,
    Stroke,
    Style,
    Text,
    draw_file,
    image_info,
    imread,
    imread_gray,
    imwrite,
    load_shapes,
    parse_shape,
    render,
    resolve_color,
    shapes_from_json,
    to_bgr,
)
from .errors import (
    CVError,
    DependencyError,
    ImageError,
    ModelError,
    ModelNotFoundError,
    ShapeError,
)
from .geometry import (
    CIRCLE_SAMPLES,
    POINT_RADIUS,
    Bounds,
    Box,
    Circle,
    Geometry,
    Line,
    Obb,
    Point,
    Polygon,
    Pose,
)
from .infer import (
    FORMATS,
    FormatInfo,
    Metrics,
    Model,
    ModelFormat,
    ModelInfo,
    TrainResult,
    devices,
    format_of,
    load,
    predict,
    resolve_device,
)
from .models import Category, ImageInfo, JsonModel, Probs, Result, Task

#: 子模块（``from cs.cv import draw`` 等价于导入绘制模块）
submodules = (annotation, draw, geometry, infer, models)

__all__ = [
    "CIRCLE_SAMPLES",
    "FORMATS",
    "POINT_RADIUS",
    "Anno",
    "AnnoDataset",
    "AnnoImg",
    "Bounds",
    "Box",
    "CVError",
    "Canvas",
    "Category",
    "Circle",
    "ColorLike",
    "CompareReport",
    "Cv",
    "DatasetSummary",
    "DependencyError",
    "FormatInfo",
    "Geometry",
    "ImageError",
    "ImageInfo",
    "ImgSize",
    "JsonModel",
    "Line",
    "Metrics",
    "Model",
    "ModelError",
    "ModelFormat",
    "ModelInfo",
    "ModelNotFoundError",
    "Obb",
    "Point",
    "Polygon",
    "Pose",
    "Probs",
    "Result",
    "Shape",
    "ShapeError",
    "Stroke",
    "Style",
    "Task",
    "Text",
    "TrainResult",
    "devices",
    "draw",
    "draw_file",
    "format_of",
    "geometry",
    "image_info",
    "image_size",
    "imread",
    "imread_gray",
    "imwrite",
    "infer",
    "load",
    "load_shapes",
    "models",
    "parse_shape",
    "predict",
    "render",
    "resolve_color",
    "resolve_device",
    "shapes_from_json",
    "submodules",
    "to_bgr",
]
