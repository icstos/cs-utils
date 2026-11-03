"""把 :mod:`cs.cv` 暴露为 MCP 工具（FastMCP ≥ 4）。

启动方式::

    python -m cs.cv serve                 # stdio，供 MCP 客户端拉起
    python -m cs.cv serve --transport http --port 8000
    fastmcp run src/cs/cv/mcp_server.py:mcp

约定：工具名统一 ``cv_*``；成功返回 ``{"ok": true, ...}``，
业务异常统一转换成 ``ToolError``，避免把 traceback 泄露给模型。
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from .annotation import AnnoDataset
from .core import Cv
from .draw import imread, imread_gray, parse_shape
from .errors import CVError
from .infer import FORMATS, Model, ModelInfo, devices
from .models import ImageInfo, Task

__all__ = ["create_server", "mcp"]

_SERVER_NAME = "cs-cv"
_INSTRUCTIONS = """\
本服务器提供计算机视觉能力：图像信息读取与图元绘制（点/线/框/旋转框/多边形/
圆/关键点/文字/掩码），以及基于 ultralytics YOLO 的分类、检测、分割、姿态、
旋转框与跟踪模型的加载、预测、训练、验证与导出。

使用建议：
1. 先调用 cv_info 确认图像尺寸（传入权重路径则返回模型的类别表与参数量），
   坐标以左上角为原点、单位为像素；
2. cv_draw 的 shapes 用 JSON 描述，必填 kind（point/line/box/obb/polygon/
   circle/pose/text），label 为自定义文字，category_idx 决定调色板取色；
3. 预测类工具传 output_dir 即可把渲染结果存成图片（同名 .jpg）；
4. 训练/导出耗时较长，建议传 log 把 ultralytics 日志落盘后再读取；
5. cv_formats / cv_devices 可用于确认可选导出格式与本机设备。
"""


def _guard[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """把领域异常转换成 MCP 可读错误。"""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except (CVError, OSError, ValueError) as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


def _done(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def create_server(
    name: str = _SERVER_NAME, *, instructions: str = _INSTRUCTIONS
) -> FastMCP:
    """构造一个独立的服务器实例（便于测试或多实例部署）。"""
    server: FastMCP = FastMCP(name, instructions=instructions)

    # ---------------------------------------------------------- 只读信息
    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def cv_info(source: str) -> dict[str, Any]:
        """读取图像或权重的基本信息（不修改文件）。

        - 图像：宽高、通道数、dtype、是否灰度；
        - 权重（.pt/.onnx/.engine/...）：任务、类别表、设备、格式、参数量。
        """
        match Cv.describe(source):
            case ImageInfo() as info:
                return _done(kind="image", **info.to_dict())
            case ModelInfo() as info:
                return _done(kind="model", **info.to_dict())

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def cv_formats() -> dict[str, Any]:
        """列出可用的模型导出格式（参数、产物后缀、CPU/GPU 支持）。"""
        return _done(formats=[item.to_dict() for item in FORMATS])

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def cv_devices() -> dict[str, Any]:
        """列出本机可用的计算设备，如 ["cpu", "cuda:0"]。"""
        return _done(devices=devices())

    # ---------------------------------------------------------- 绘制
    @server.tool()
    @_guard
    def cv_draw(
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
        - obb:     {"cx": 60, "cy": 50, "w": 80, "h": 40, "angle": 30}
        - polygon: {"points": [[x1, y1], [x2, y2], [x3, y3], ...]}
        - circle:  {"cx": 50, "cy": 50, "r": 20}
        - pose:    {"points": [[x, y], ...]}
        - text:    {"text": "缺陷", "x": 10, "y": 30, "size": 18}
        公共可选字段：label（自定义标签文字）、conf（置信度）、category_idx（取色）。
        style 可选字段：color、thickness、stroke(solid/dashed/dotted)、fill、alpha、
        label、label_bg、font_size。
        """
        from .draw import Canvas, Style

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
    def cv_mask(
        source: str,
        output: str,
        mask: str,
        color: str | list[int] | int | None = None,
        alpha: float = 0.35,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """把掩码图（非零像素）以半透明颜色叠加到原图，用于分割/热区可视化。"""
        from .draw import Canvas

        canvas = Canvas(imread(source))
        canvas.mask(imread_gray(mask), color=color, alpha=alpha)
        path = canvas.save(output, overwrite=overwrite)
        return _done(output=str(path), alpha=alpha)

    # ---------------------------------------------------------- 推理
    @server.tool()
    @_guard
    def cv_predict(
        model: str,
        source: str,
        output_dir: str | None = None,
        task: str | None = None,
        device: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        max_det: int = 300,
        classes: list[int] | None = None,
    ) -> dict[str, Any]:
        """对图片/目录做推理，返回每张图的图元（框/多边形/旋转框/关键点/概率）。

        source 可为图片或目录；output_dir 不为空时把渲染结果存成同名 .jpg。
        classes 为类别下标列表，留空表示全部类别。
        """
        results = Model(model, task=task, device=device, imgsz=imgsz).predict(
            source,
            conf=conf,
            iou=iou,
            max_det=max_det,
            classes=classes,
            output_dir=output_dir,
        )
        return _done(count=len(results), results=[item.to_dict() for item in results])

    @server.tool()
    @_guard
    def cv_track(
        model: str,
        source: str,
        tracker: str = "bytetrack.yaml",
        output_dir: str | None = None,
        device: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> dict[str, Any]:
        """对视频/视频流做多目标跟踪（tracker 可选 bytetrack.yaml / botsort.yaml）。"""
        results = Model(model, task=Task.DETECT, device=device).track(
            source, tracker=tracker, conf=conf, iou=iou, output_dir=output_dir
        )
        return _done(count=len(results), frames=[item.to_dict() for item in results])

    @server.tool()
    @_guard
    def cv_train(
        model: str,
        data: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = -1,
        device: str | None = None,
        project: str = "runs/train",
        name: str = "exp",
        log: str | None = None,
    ) -> dict[str, Any]:
        """训练模型，data 为 YOLO 数据集配置（yaml）；log 给出时日志同时落盘。"""
        result = Model(model, imgsz=imgsz, device=device).train(
            data,
            epochs=epochs,
            batch=batch,
            project=project,
            name=name,
            log=log,
        )
        return _done(**result.to_dict())

    @server.tool()
    @_guard
    def cv_val(
        model: str,
        data: str | None = None,
        split: str = "val",
        imgsz: int = 640,
        device: str | None = None,
    ) -> dict[str, Any]:
        """在验证集上评估，返回 mAP / 准确率等指标。"""
        metrics = Model(model, imgsz=imgsz, device=device).val(data, split=split)
        return _done(**metrics.to_dict())

    @server.tool()
    @_guard
    def cv_export(
        model: str,
        format: str = "onnx",
        imgsz: int = 640,
        half: bool = False,
        int8: bool = False,
        dynamic: bool = False,
        simplify: bool = True,
        device: str | None = None,
    ) -> dict[str, Any]:
        """导出为部署格式（onnx / torchscript / openvino / engine / pt）。"""
        path = Model(model, imgsz=imgsz, device=device).export(
            format,
            half=half,
            int8=int8,
            dynamic=dynamic,
            simplify=simplify,
        )
        return _done(output=str(path))

    # ---------------------------------------------------------- 数据集
    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def cv_annotations(
        img_dir: str, anno_dir: str, task: str = "detect"
    ) -> dict[str, Any]:
        """扫描 YOLO 数据集目录，统计图像/标签/目标数量与类别分布。"""
        dataset = AnnoDataset(img_dir=img_dir, anno_dir=anno_dir, task=task)
        return _done(**dataset.summary().to_dict())

    return server


mcp = create_server()
