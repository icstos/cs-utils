"""命令行入口：``python -m cs.cv ...``。

::

    python -m cs.cv info in.jpg
    python -m cs.cv info yolo11n.pt
    python -m cs.cv draw in.jpg -o out.jpg --shape '{"kind":"box","x1":10,"y1":10,"x2":80,"y2":60}'
    python -m cs.cv mask in.jpg -o out.jpg mask.png --alpha 0.4
    python -m cs.cv predict yolo11n.pt images/ -o out --conf 0.4
    python -m cs.cv track yolo11n.pt video.mp4 -o out
    python -m cs.cv train yolo11n.pt coco8.yaml --epochs 50 --log train.log
    python -m cs.cv val yolo11n.pt --data coco8.yaml
    python -m cs.cv export yolo11n.pt onnx
    python -m cs.cv formats / devices / dataset images labels
    python -m cs.cv serve --transport http --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .annotation import AnnoDataset
from .core import Cv
from .draw import Canvas, Style, imread, imread_gray, parse_shape
from .errors import CVError
from .infer import FORMATS, Model, devices

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cs.cv",
        description="计算机视觉工具：图像绘制、YOLO 推理/训练/验证/导出、数据集统计。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---------------------------------------------------------- 服务
    serve = sub.add_parser("serve", help="以 MCP 服务方式启动")
    serve.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--name", default="cs-cv")

    # ---------------------------------------------------------- 信息
    info = sub.add_parser("info", help="查看图像或模型的基本信息（按后缀自动判断）")
    info.add_argument("source", help="图像路径，或权重文件（.pt/.onnx/.engine/...）")

    sub.add_parser("formats", help="列出可用的导出格式")
    sub.add_parser("devices", help="列出可用的计算设备")

    # ---------------------------------------------------------- 绘制
    draw = sub.add_parser("draw", help="按 JSON 规格绘制一批图元")
    draw.add_argument("source", help="输入图像路径")
    draw.add_argument("-o", "--output", required=True, help="输出图像路径")
    draw.add_argument("--shape", action="append", default=[], help="图元 JSON，可重复")
    draw.add_argument("--shapes-file", help="图元 JSON 文件（数组或 {shapes:[...]}）")
    draw.add_argument("--style", default=None, help="样式 JSON，如 {'thickness':3}")
    draw.add_argument("--overwrite", action="store_true", help="允许覆盖已存在文件")

    mask = sub.add_parser("mask", help="掩码着色")
    mask.add_argument("source", help="输入图像路径")
    mask.add_argument("-o", "--output", required=True, help="输出图像路径")
    mask.add_argument("mask", help="掩码图像路径（非零像素视为目标区域）")
    mask.add_argument("--color", default=None, help="叠加颜色")
    mask.add_argument("--alpha", type=float, default=0.35, help="透明度 0~1")
    mask.add_argument("--overwrite", action="store_true", help="允许覆盖已存在文件")

    # ---------------------------------------------------------- 推理
    predict = sub.add_parser("predict", help="推理（图片/目录），可同时渲染结果")
    predict.add_argument("model", help="权重或模型名（如 yolo11n.pt）")
    predict.add_argument("source", help="输入图片 / 目录")
    predict.add_argument("-o", "--output-dir", default=None, help="渲染结果目录")
    predict.add_argument(
        "--task", default=None, help="任务：detect/segment/pose/obb/classify"
    )
    predict.add_argument("--device", default=None, help="设备，如 cpu / 0")
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--iou", type=float, default=0.45)
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--max-det", type=int, default=300)
    predict.add_argument("--classes", default=None, help="类别下标，逗号分隔")
    predict.add_argument("--overwrite", action="store_true", help="允许覆盖已存在文件")

    track = sub.add_parser("track", help="视频多目标跟踪")
    track.add_argument("model", help="权重或模型名")
    track.add_argument("source", help="视频 / 视频流")
    track.add_argument("-o", "--output-dir", default=None, help="渲染结果目录")
    track.add_argument("--tracker", default="bytetrack.yaml")
    track.add_argument("--device", default=None)
    track.add_argument("--conf", type=float, default=0.25)
    track.add_argument("--iou", type=float, default=0.45)

    # ---------------------------------------------------------- 训练 / 验证 / 导出
    train = sub.add_parser("train", help="训练模型")
    train.add_argument("model", help="初始权重或模型名")
    train.add_argument("data", help="数据集配置 yaml")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch", type=int, default=-1)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--device", default=None)
    train.add_argument("--workers", type=int, default=8)
    train.add_argument("--project", default="runs/train")
    train.add_argument("--name", default="exp")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--log", default=None, help="把 ultralytics 日志写入该文件")

    val = sub.add_parser("val", help="在验证集上评估")
    val.add_argument("model", help="权重或模型名")
    val.add_argument("--data", default=None, help="数据集配置 yaml")
    val.add_argument("--split", default="val")
    val.add_argument("--imgsz", type=int, default=640)
    val.add_argument("--device", default=None)

    export = sub.add_parser("export", help="导出为部署格式")
    export.add_argument("model", help="权重或模型名")
    export.add_argument(
        "format",
        nargs="?",
        default="onnx",
        choices=[item.format.value for item in FORMATS],
        help="导出格式（默认 onnx）",
    )
    export.add_argument("--imgsz", type=int, default=640)
    export.add_argument("--half", action="store_true", help="FP16")
    export.add_argument("--int8", action="store_true", help="INT8 量化")
    export.add_argument("--dynamic", action="store_true", help="动态输入尺寸")
    export.add_argument("--no-simplify", action="store_true", help="不简化 ONNX 计算图")
    export.add_argument("--device", default=None)

    # ---------------------------------------------------------- 数据集
    dataset = sub.add_parser("dataset", help="统计 YOLO 数据集")
    dataset.add_argument("img_dir", help="图像目录")
    dataset.add_argument("anno_dir", help="标签目录")
    dataset.add_argument("--task", default="detect")

    return parser


def _dump(payload: Any) -> str:
    """把模型 / 列表 / 路径统一序列化为 JSON。"""
    match payload:
        case Path():
            return str(payload)
        case list() | tuple() as items:
            return json.dumps(
                [
                    item.to_dict() if hasattr(item, "to_dict") else item
                    for item in items
                ],
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        case _ if hasattr(payload, "to_dict"):
            return json.dumps(
                payload.to_dict(), ensure_ascii=False, indent=2, default=str
            )
        case _:
            return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _style(raw: str | None) -> Style:
    return Style.from_dict(json.loads(raw) if raw else None)


def _load_shapes(args: argparse.Namespace) -> list[dict[str, Any]]:
    """合并 --shape 与 --shapes-file 的图元规格。"""
    shapes: list[dict[str, Any]] = [json.loads(item) for item in args.shape]
    if args.shapes_file:
        payload = json.loads(Path(args.shapes_file).read_text(encoding="utf-8"))
        shapes.extend(payload["shapes"] if isinstance(payload, dict) else payload)
    if not shapes:
        raise CVError("至少需要一个 --shape 或 --shapes-file")
    return shapes


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        match args.command:
            case "serve":
                return _serve(args)
            case "info":
                print(_dump(Cv.describe(args.source)))
            case "formats":
                print(_dump(FORMATS))
            case "devices":
                print(_dump(devices()))
            case "draw":
                canvas = Canvas(imread(args.source), style=_style(args.style))
                canvas.draw_all([parse_shape(item) for item in _load_shapes(args)])
                print(canvas.save(args.output, overwrite=args.overwrite))
            case "mask":
                canvas = Canvas(imread(args.source))
                color = json.loads(args.color) if args.color else None
                canvas.mask(imread_gray(args.mask), color=color, alpha=args.alpha)
                print(canvas.save(args.output, overwrite=args.overwrite))
            case "predict":
                classes = (
                    [int(v) for v in args.classes.split(",")] if args.classes else None
                )
                results = Model(
                    args.model, task=args.task, device=args.device, imgsz=args.imgsz
                ).predict(
                    args.source,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    classes=classes,
                    output_dir=args.output_dir,
                    overwrite=args.overwrite,
                )
                print(_dump([item.to_dict() for item in results]))
            case "track":
                results = Model(args.model, device=args.device).track(
                    args.source,
                    tracker=args.tracker,
                    conf=args.conf,
                    iou=args.iou,
                    output_dir=args.output_dir,
                )
                print(_dump([item.to_dict() for item in results]))
            case "train":
                result = Model(args.model, device=args.device).train(
                    args.data,
                    epochs=args.epochs,
                    batch=args.batch,
                    imgsz=args.imgsz,
                    workers=args.workers,
                    project=args.project,
                    name=args.name,
                    resume=args.resume,
                    log=args.log,
                )
                print(_dump(result))
            case "val":
                metrics = Model(args.model, device=args.device).val(
                    args.data, split=args.split, imgsz=args.imgsz
                )
                print(_dump(metrics))
            case "export":
                path = Model(args.model, device=args.device).export(
                    args.format,
                    imgsz=args.imgsz,
                    half=args.half,
                    int8=args.int8,
                    dynamic=args.dynamic,
                    simplify=not args.no_simplify,
                )
                print(path)
            case "dataset":
                dataset = AnnoDataset(args.img_dir, args.anno_dir, task=args.task)
                print(_dump(dataset.summary()))
    except (CVError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


def _serve(args: argparse.Namespace) -> int:
    from .mcp_server import create_server

    server = create_server(args.name)
    if args.transport == "http":
        server.run(transport="http", host=args.host, port=args.port)
    else:
        # stdio 的 stderr 会被客户端收进日志，关掉横幅更干净
        server.run(transport="stdio", show_banner=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
