"""命令行入口：``python -m cs.cv_draw ...``。

python -m cs.cv_draw info in.jpg
python -m cs.cv_draw draw in.jpg -o out.jpg --shape '{"kind":"box","x1":10,"y1":10,"x2":80,"y2":60,"label":"cat"}'
python -m cs.cv_draw text in.jpg -o out.jpg "缺陷 0.92" 10 30 --size 20
python -m cs.cv_draw mask in.jpg -o out.jpg mask.png --alpha 0.4
python -m cs.cv_draw grid in.jpg -o out.jpg --cols 8 --rows 6
python -m cs.cv_draw serve --transport http --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..cv import Point
from .core import Canvas, image_info, imread, imread_gray
from .errors import DrawError
from .models import Style, parse_shape

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cs.cv_draw",
        description="图像标注绘制：点、线、框、多边形、圆、文字、掩码、网格。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="以 MCP 服务方式启动")
    serve.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--name", default="cs-cv-draw")

    info = sub.add_parser("info", help="查看图像基本信息")
    info.add_argument("source", help="输入图像路径")

    draw = sub.add_parser("draw", help="按 JSON 规格绘制一批图元")
    draw.add_argument("source", help="输入图像路径")
    draw.add_argument("-o", "--output", required=True, help="输出图像路径")
    draw.add_argument(
        "--shape", action="append", default=[], help="图元 JSON，可重复传入"
    )
    draw.add_argument("--shapes-file", help="图元 JSON 文件（数组或 {shapes:[...]}）")
    draw.add_argument("--style", default=None, help="样式 JSON，如 {'thickness':3}")
    draw.add_argument("--overwrite", action="store_true", help="允许覆盖已存在文件")

    text = sub.add_parser("text", help="写一行文字")
    text.add_argument("source", help="输入图像路径")
    text.add_argument("-o", "--output", required=True, help="输出图像路径")
    text.add_argument("content", help="文字内容")
    text.add_argument("x", type=float, help="左上角 x")
    text.add_argument("y", type=float, help="左上角 y")
    text.add_argument("--size", type=int, default=0, help="字号（0=自适应）")
    text.add_argument("--color", default=None, help="颜色，如 #FF0000 或 [255,0,0]")
    text.add_argument("--overwrite", action="store_true", help="允许覆盖已存在文件")

    mask = sub.add_parser("mask", help="掩码着色")
    mask.add_argument("source", help="输入图像路径")
    mask.add_argument("-o", "--output", required=True, help="输出图像路径")
    mask.add_argument("mask", help="掩码图像路径（非零像素视为目标区域）")
    mask.add_argument("--color", default=None, help="叠加颜色")
    mask.add_argument("--alpha", type=float, default=0.35, help="透明度 0~1")
    mask.add_argument("--overwrite", action="store_true", help="允许覆盖已存在文件")

    grid = sub.add_parser("grid", help="绘制辅助网格")
    grid.add_argument("source", help="输入图像路径")
    grid.add_argument("-o", "--output", required=True, help="输出图像路径")
    grid.add_argument("--cols", type=int, default=10)
    grid.add_argument("--rows", type=int, default=10)
    grid.add_argument("--color", default=None, help="网格颜色")
    grid.add_argument("--thickness", type=int, default=1)
    grid.add_argument("--overwrite", action="store_true", help="允许覆盖已存在文件")

    return parser


def _load_shapes(args: argparse.Namespace) -> list[dict[str, Any]]:
    """合并 --shape 与 --shapes-file 的图元规格。"""
    shapes: list[dict[str, Any]] = [json.loads(item) for item in args.shape]
    if args.shapes_file:
        payload = json.loads(Path(args.shapes_file).read_text(encoding="utf-8"))
        shapes.extend(payload["shapes"] if isinstance(payload, dict) else payload)
    if not shapes:
        raise DrawError("至少需要一个 --shape 或 --shapes-file")
    return shapes


def _style(raw: str | None) -> Style:
    return Style.from_dict(json.loads(raw) if raw else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        match args.command:
            case "serve":
                return _serve(args)
            case "info":
                print(json.dumps(image_info(args.source).to_dict(), ensure_ascii=False))
                return 0
            case "draw":
                canvas = Canvas(imread(args.source), style=_style(args.style))
                annos = [parse_shape(item) for item in _load_shapes(args)]
                canvas.draw_all(annos)
                print(canvas.save(args.output, overwrite=args.overwrite))
                return 0
            case "text":
                canvas = Canvas(imread(args.source))
                style = Style(color=json.loads(args.color) if args.color else None)
                canvas.text(
                    args.content, Point(args.x, args.y), size=args.size, style=style
                )
                print(canvas.save(args.output, overwrite=args.overwrite))
                return 0
            case "mask":
                canvas = Canvas(imread(args.source))
                color = json.loads(args.color) if args.color else None
                canvas.mask(imread_gray(args.mask), color=color, alpha=args.alpha)
                print(canvas.save(args.output, overwrite=args.overwrite))
                return 0
            case "grid":
                canvas = Canvas(imread(args.source))
                color = json.loads(args.color) if args.color else None
                canvas.grid(
                    args.cols,
                    args.rows,
                    style=Style(color=color, thickness=args.thickness),
                )
                print(canvas.save(args.output, overwrite=args.overwrite))
                return 0
    except (DrawError, OSError, json.JSONDecodeError) as exc:
        print(f"错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 1


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
