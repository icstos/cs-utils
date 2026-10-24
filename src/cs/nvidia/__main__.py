"""命令行入口：``python -m cs.nvidia ...``。

python -m cs.nvidia                          # 等同 status
python -m cs.nvidia list
python -m cs.nvidia info 0 --json
python -m cs.nvidia status --json
python -m cs.nvidia processes
python -m cs.nvidia watch 0 -i 1 -n 10 --csv gpu.csv
python -m cs.nvidia health
python -m cs.nvidia topology
python -m cs.nvidia serve --transport http --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from . import monitor
from .core import NVIDIA
from .errors import GPUError

__all__ = ["main"]


def _emit(payload: dict[str, Any], *, as_json: bool, text: str) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cs.nvidia",
        description="NVIDIA 显卡查询：型号、运行状态、进程、降频、健康检查。",
    )
    sub = parser.add_subparsers(dest="command")

    def with_gpu(
        name: str, help_text: str, *, default: str = "all"
    ) -> argparse.ArgumentParser:
        item = sub.add_parser(name, help=help_text)
        item.add_argument(
            "gpu", nargs="?", default=default, help="序号 / UUID / PCI 地址"
        )
        item.add_argument("--json", action="store_true", help="输出 JSON")
        return item

    sub.add_parser("driver", help="驱动与 CUDA 版本").add_argument(
        "--json", action="store_true"
    )
    sub.add_parser("list", help="列出所有显卡").add_argument(
        "--json", action="store_true"
    )
    with_gpu("info", "单卡详细信息", default="0")
    with_gpu("status", "运行状态")
    with_gpu("processes", "占用显卡的进程")
    with_gpu("throttle", "降频原因")
    with_gpu("health", "健康体检")

    sub.add_parser("topology", help="多卡互联拓扑").add_argument(
        "--json", action="store_true"
    )

    watch = sub.add_parser("watch", help="采样一段时间并统计")
    watch.add_argument("gpu", nargs="?", default="0")
    watch.add_argument(
        "-i", "--interval", type=float, default=1.0, help="采样间隔（秒）"
    )
    watch.add_argument("-n", "--count", type=int, default=10, help="采样次数")
    watch.add_argument(
        "-t", "--duration", type=float, default=None, help="总时长（秒）"
    )
    watch.add_argument("--csv", default=None, help="同时导出 CSV 到该路径")
    watch.add_argument("--json", action="store_true")

    serve = sub.add_parser("serve", help="以 MCP 服务方式启动")
    serve.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--name", default="cs-nvidia")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = args.command or "status"

    try:
        match command:
            case "serve":
                from .mcp_server import create_server

                server = create_server(args.name)
                if args.transport == "http":
                    server.run(transport="http", host=args.host, port=args.port)
                else:
                    # stdio 的 stderr 会被客户端收进日志，关掉横幅更干净
                    server.run(transport="stdio", show_banner=False)
                return 0

            case "driver":
                info = NVIDIA.driver()
                _emit(info.to_dict(), as_json=args.json, text=str(info))

            case "list":
                infos = NVIDIA.list()
                text = "\n".join(f"{item}" for item in infos)
                _emit(
                    {"count": len(infos), "gpus": [item.to_dict() for item in infos]},
                    as_json=args.json,
                    text=text,
                )

            case "info":
                info = NVIDIA.info(args.gpu)
                body = info.to_dict()
                text = "\n".join(f"{key:<22} {value}" for key, value in body.items())
                _emit(body, as_json=args.json, text=text)

            case "status":
                states = (
                    NVIDIA.statuses()
                    if args.gpu == "all"
                    else [NVIDIA.status(args.gpu)]
                )
                _emit(
                    {"gpus": [state.to_dict() for state in states]},
                    as_json=args.json,
                    text=monitor.render_table(states),
                )

            case "processes":
                processes = (
                    NVIDIA.all_processes()
                    if args.gpu == "all"
                    else NVIDIA.processes(args.gpu)
                )
                text = "\n".join(f"{item}" for item in processes) or "没有进程占用显卡"
                _emit(
                    {"processes": [item.to_dict() for item in processes]},
                    as_json=args.json,
                    text=text,
                )

            case "throttle":
                if args.gpu == "all":
                    rows = [
                        {
                            "index": handle.index,
                            "name": handle.name,
                            **handle.throttle().to_dict(),
                        }
                        for handle in NVIDIA.gpus()
                    ]
                    text = "\n".join(
                        f"[{row['index']}] {row['name']}: {', '.join(row['active']) or 'NONE'}"
                        for row in rows
                    )
                    _emit({"gpus": rows}, as_json=args.json, text=text)
                else:
                    info = NVIDIA.throttle(args.gpu)
                    _emit(info.to_dict(), as_json=args.json, text=f"降频原因: {info}")

            case "health":
                reports = (
                    NVIDIA.health_all()
                    if args.gpu == "all"
                    else [NVIDIA.health(args.gpu)]
                )
                _emit(
                    {"gpus": [report.to_dict() for report in reports]},
                    as_json=args.json,
                    text=monitor.summarize_health(reports),
                )

            case "topology":
                matrix = NVIDIA.topologies()
                _emit(matrix.to_dict(), as_json=args.json, text=str(matrix))

            case "watch":
                handle = NVIDIA.gpu(args.gpu)
                series = monitor.SampleSeries(
                    index=handle.index,
                    uuid=handle.uuid,
                    name=handle.name,
                    interval=args.interval,
                    states=tuple(
                        handle.watch(
                            interval=args.interval,
                            count=args.count,
                            duration=args.duration,
                        )
                    ),
                )
                lines = [str(series)]
                if args.csv:
                    lines.append(f"已写入 {series.to_csv(args.csv)}")
                _emit(
                    series.to_dict(include_states=args.json),
                    as_json=args.json,
                    text="\n".join(lines),
                )

            case _:  # pragma: no cover - argparse 已保证不会到这里
                parser.print_help()
                return 1
    except GPUError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    finally:
        NVIDIA.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
