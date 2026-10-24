"""命令行入口：``python -m cs.pdf ...``。

python -m cs.pdf info input.pdf
python -m cs.pdf text input.pdf -p 1-3 > out.txt
python -m cs.pdf split input.pdf -o out --every 10
python -m cs.pdf merge merged.pdf a.pdf b.pdf
python -m cs.pdf render input.pdf -o out --dpi 150
python -m cs.pdf serve --transport http --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .core import PDF, ImageFormat

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cs.pdf",
        description="PDF 常用操作：查看、抽取、拆分、合并、旋转、渲染、加密等。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("path", help="输入 PDF 路径")
    common.add_argument("--password", default=None, help="PDF 打开密码（如已加密）")

    serve = sub.add_parser("serve", help="以 MCP 服务方式启动")
    serve.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--name", default="cs-pdf")

    info = sub.add_parser("info", help="查看文档信息", parents=[common])
    info.add_argument("--json", action="store_true", help="输出 JSON")
    info.add_argument("--no-pages", action="store_true", help="不输出逐页信息")

    text = sub.add_parser("text", help="抽取文本", parents=[common])
    text.add_argument("-p", "--pages", default="all", help="页码范围，如 1-3,5")
    text.add_argument("--layout", action="store_true", help="保留版面排版")
    text.add_argument(
        "-o", "--output", default=None, help="写入文件（默认打印到标准输出）"
    )

    split = sub.add_parser("split", help="拆分文档", parents=[common])
    split.add_argument("-o", "--out-dir", required=True, help="输出目录")
    split.add_argument("--every", type=int, default=None, help="每 N 页一个文件")
    split.add_argument("--pages", default=None, help="按片段拆分，如 1-3,5")
    split.add_argument("--prefix", default=None, help="输出文件名前缀")
    split.add_argument("--overwrite", action="store_true", help="允许覆盖已存在文件")

    merge = sub.add_parser("merge", help="合并多个文档")
    merge.add_argument("output", help="输出 PDF 路径")
    merge.add_argument("inputs", nargs="+", help="按顺序合并的源文件")
    merge.add_argument("--outline", action="store_true", help="为每个源文件生成书签")
    merge.add_argument("--overwrite", action="store_true", help="允许覆盖已存在文件")

    render = sub.add_parser("render", help="渲染为图片", parents=[common])
    render.add_argument("-o", "--out-dir", required=True, help="输出目录")
    render.add_argument("--dpi", type=int, default=200)
    render.add_argument("-p", "--pages", default="all")
    render.add_argument(
        "--format", default=ImageFormat.PNG, choices=("png", "jpg", "webp")
    )
    render.add_argument("--overwrite", action="store_true", help="允许覆盖已存在文件")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    match args.command:
        case "serve":
            from .mcp_server import create_server

            server = create_server(args.name)
            if args.transport == "http":
                server.run(transport="http", host=args.host, port=args.port)
            else:
                # stdio 的 stderr 会被客户端收进日志，关掉横幅更干净
                server.run(transport="stdio", show_banner=False)
            return 0

        case "info":
            info = PDF.info(args.path, password=args.password)
            if args.json:
                print(
                    json.dumps(
                        info.to_dict(include_pages=not args.no_pages),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                print(info)
                for page in () if args.no_pages else info.pages:
                    size = page.size or f"{page.width:.0f}x{page.height:.0f}"
                    print(f"  p{page.number:<4} {size:<8} rotate={page.rotation}")
            return 0

        case "text":
            content = PDF.text(
                args.path, args.pages, layout=args.layout, password=args.password
            )
            if args.output:
                from pathlib import Path

                Path(args.output).write_text(content, encoding="utf-8")
                print(f"已写入 {args.output}（{len(content)} 字符）")
            else:
                print(content)
            return 0

        case "split":
            files = PDF.split(
                args.path,
                args.out_dir,
                pages=args.pages,
                every=args.every,
                prefix=args.prefix,
                password=args.password,
                overwrite=args.overwrite,
            )
            print(f"已生成 {len(files)} 个文件：")
            for item in files:
                print(f"  {item}")
            return 0

        case "merge":
            result = PDF.merge(
                args.inputs, args.output, outline=args.outline, overwrite=args.overwrite
            )
            print(f"已合并 {len(args.inputs)} 个文件 → {result}")
            return 0

        case "render":
            files = PDF.render(
                args.path,
                args.out_dir,
                dpi=args.dpi,
                pages=args.pages,
                fmt=args.format,
                password=args.password,
                overwrite=args.overwrite,
            )
            print(f"已生成 {len(files)} 张图片：")
            for item in files:
                print(f"  {item}")
            return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
