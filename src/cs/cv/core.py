"""``Cv`` 门面：把绘制、推理、标注数据集三组能力收进一个类。

方法全部是静态方法（无状态），既方便脚本调用，也方便 MCP 层薄封装：

>>> from cs.cv import Cv
>>> Cv.info("bus.jpg").size
ImgSize(width=810, height=1080)
>>> Cv.predict("yolo11n.pt", "bus.jpg", output_dir="out")[0].boxes[0].conf
0.8712
>>> Cv.formats()[2].format
<ModelFormat.ONNX: 'onnx'>
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import infer
from .annotation import AnnoDataset
from .draw import Style, draw_file, image_info, parse_shape, render
from .infer import FORMATS, FormatInfo, Model, ModelFormat, ModelInfo, format_of
from .models import Category, ImageInfo, Result, Task

__all__ = ["Cv"]

type StrPath = str | Path
type ShapeSpec = Mapping[str, Any]


class Cv:
    """计算机视觉门面。"""

    __slots__ = ()

    #: 枚举与样式类挂在门面上，便于 ``Cv.Task.DETECT`` / ``Cv.Style(fill=True)``
    Task = Task
    Format = ModelFormat
    Style = Style

    # ---------------------------------------------------------- 信息
    @staticmethod
    def info(source: StrPath) -> ImageInfo:
        """图像宽高、通道数与 dtype。"""
        return image_info(source)

    @staticmethod
    def model_info(source: StrPath, task: Task | str | None = None) -> ModelInfo:
        """模型概要：任务、类别表、设备、格式与参数量（会加载权重）。"""
        with Model(source, task=task) as model:
            return model.info()

    @staticmethod
    def describe(source: StrPath) -> ImageInfo | ModelInfo:
        """按后缀自动判断：权重文件返回模型概要，其余当作图像。"""
        if format_of(source) is not None:
            return Cv.model_info(source)
        return image_info(source)

    @staticmethod
    def formats() -> list[FormatInfo]:
        """可用的模型导出格式。"""
        return list(FORMATS)

    @staticmethod
    def devices() -> list[str]:
        """当前机器可用的计算设备。"""
        return infer.devices()

    # ---------------------------------------------------------- 绘制
    @staticmethod
    def draw(
        image: np.ndarray,
        shapes: Iterable[Any],
        *,
        style: Style | None = None,
        inplace: bool = False,
        category: Category | None = None,
    ) -> np.ndarray:
        """在图像上绘制一批图元（默认不改动入参）。"""
        return render(image, shapes, style=style, inplace=inplace, category=category)

    @staticmethod
    def draw_specs(
        image: np.ndarray,
        specs: Iterable[ShapeSpec],
        *,
        style: Style | None = None,
        inplace: bool = False,
        category: Category | None = None,
    ) -> np.ndarray:
        """同上，但图元用 JSON 规格描述（MCP / CLI 场景）。"""
        return render(
            image,
            [parse_shape(spec) for spec in specs],
            style=style,
            inplace=inplace,
            category=category,
        )

    @staticmethod
    def draw_file(
        source: StrPath,
        target: StrPath,
        shapes: Iterable[Any],
        *,
        style: Style | None = None,
        category: Category | None = None,
        overwrite: bool = False,
    ) -> Path:
        """读图 → 绘制 → 存盘。"""
        return draw_file(
            source,
            target,
            shapes,
            style=style,
            category=category,
            overwrite=overwrite,
        )

    # ---------------------------------------------------------- 推理
    @staticmethod
    def load(
        model: StrPath = "yolo11n.pt",
        *,
        task: Task | str | None = None,
        device: str | None = None,
        imgsz: int = 640,
        verbose: bool = False,
    ) -> Model:
        """加载模型（返回已就绪的 :class:`~cs.cv.infer.Model`）。"""
        return Model(
            path=model, task=task, device=device, imgsz=imgsz, verbose=verbose
        ).load()

    @staticmethod
    def predict(
        model: Model | StrPath,
        source: Any,
        *,
        task: Task | str | None = None,
        device: str | None = None,
        output_dir: StrPath | None = None,
        style: Style | None = None,
        overwrite: bool = False,
        **options: Any,
    ) -> list[Result]:
        """一次性推理；给出 ``output_dir`` 时同时把渲染结果存盘。"""
        runner = (
            model
            if isinstance(model, Model)
            else Model(model, task=task, device=device)
        )
        return runner.predict(
            source, output_dir=output_dir, style=style, overwrite=overwrite, **options
        )

    @staticmethod
    def track(
        model: Model | StrPath,
        source: Any,
        *,
        task: Task | str | None = None,
        device: str | None = None,
        output_dir: StrPath | None = None,
        **options: Any,
    ) -> list[Result]:
        """视频 / 流的多目标跟踪。"""
        runner = (
            model
            if isinstance(model, Model)
            else Model(model, task=task, device=device)
        )
        return runner.track(source, output_dir=output_dir, **options)

    @staticmethod
    def export(
        model: StrPath,
        fmt: ModelFormat | str = ModelFormat.ONNX,
        *,
        device: str | None = None,
        **options: Any,
    ) -> Path:
        """导出为部署格式。"""
        return Model(model, device=device).export(fmt, **options)

    # ---------------------------------------------------------- 数据集
    @staticmethod
    def annotations(
        img_dir: StrPath,
        anno_dir: StrPath,
        *,
        task: Task | str = Task.DETECT,
        category: Category | Sequence[str] | None = None,
    ) -> AnnoDataset:
        """扫描 YOLO 图像 / 标签目录。"""
        return AnnoDataset(
            img_dir=Path(img_dir),
            anno_dir=Path(anno_dir),
            task=task,
            category=Category.parse(category),
        )
