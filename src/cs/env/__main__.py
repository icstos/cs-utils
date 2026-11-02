"""命令行入口：``python -m cs.env ...``。

python -m cs.env platform
python -m cs.env cpu --per-core
python -m cs.env memory
python -m cs.env disk E:/data
python -m cs.env gpu
python -m cs.env ps --limit 10 --sort memory
python -m cs.env proc 1234
python -m cs.env kill 1234 --force
python -m cs.env run git status --short
python -m cs.env which python
python -m cs.env vars --prefix CS_
python -m cs.env snapshot --top 5
python -m cs.env serve --transport http --port 8000
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .core import Env
from .errors import EnvError
from .models import Model, jsonable

__all__ = ["main"]


def _print_model(model: Model, *, as_json: bool, text: str | None = None) -> None:
    if as_json:
        print(model.to_json(indent=2))
    else:
        print(text if text is not None else str(model))


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS：子命令未显式给 --json 时不覆盖父级已解析的值
    common.add_argument(
        "-j",
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="输出 JSON",
    )

    parser = argparse.ArgumentParser(
        prog="python -m cs.env",
        description="操作系统级环境信息与操作：平台、CPU、内存、磁盘、GPU、"
        "网络、进程、安全命令执行。",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("platform", parents=[common], help="平台与 Python 运行时")

    cpu = sub.add_parser("cpu", parents=[common], help="CPU 信息与占用")
    cpu.add_argument("--per-core", action="store_true", help="输出每个逻辑核")
    cpu.add_argument("--interval", type=float, default=0.1, help="采样窗口（秒）")

    sub.add_parser("memory", parents=[common], help="内存与交换分区")

    disk = sub.add_parser("disk", parents=[common], help="磁盘占用")
    disk.add_argument("path", nargs="?", default=None, help="路径（缺省列出全部）")
    disk.add_argument("--all", action="store_true", help="含虚拟/临时文件系统")

    sub.add_parser("gpu", parents=[common], help="GPU 状态")
    sub.add_parser("net", parents=[common], help="网络收发量")

    io = sub.add_parser("io", parents=[common], help="磁盘读写速率")
    io.add_argument("--interval", type=float, default=1.0, help="采样窗口（秒）")

    ps = sub.add_parser("ps", parents=[common], help="进程列表")
    ps.add_argument("--limit", type=int, default=15, help="条数，0 表示全部")
    ps.add_argument("--sort", default="cpu", help="cpu / memory / pid / name / age")
    ps.add_argument("--name", default=None, help="按进程名子串过滤")

    proc = sub.add_parser("proc", parents=[common], help="进程详情")
    proc.add_argument("pid", type=int)

    kill = sub.add_parser("kill", parents=[common], help="结束进程")
    kill.add_argument("pid", type=int)
    kill.add_argument("--force", action="store_true", help="直接强杀（默认先请求退出）")

    run = sub.add_parser("run", parents=[common], help="安全执行命令（不经 shell）")
    run.add_argument("cmd", nargs=argparse.REMAINDER, help="可执行文件与参数")
    run.add_argument("--cwd", default=None, help="工作目录")
    run.add_argument("--timeout", type=float, default=30.0, help="超时秒数")
    run.add_argument("--stdin", default=None, help="传给子进程的标准输入")

    which = sub.add_parser("which", parents=[common], help="查找可执行文件")
    which.add_argument("name")

    envs = sub.add_parser("vars", parents=[common], help="环境变量")
    envs.add_argument("--prefix", default=None, help="只显示该前缀")
    envs.add_argument("--reveal", action="store_true", help="显示敏感值原文")

    snap = sub.add_parser("snapshot", parents=[common], help="整机快照")
    snap.add_argument("--top", type=int, default=5, help="Top 进程数")
    snap.add_argument("--no-gpu", action="store_true", help="跳过 GPU 采集")

    serve = sub.add_parser("serve", parents=[common], help="以 MCP 服务方式启动")
    serve.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--name", default="cs-env")

    return parser


def _serve(args: argparse.Namespace) -> int:
    from .mcp_server import create_server

    server = create_server(args.name)
    if args.transport == "http":
        server.run(transport="http", host=args.host, port=args.port)
    else:
        # stdio 的 stderr 会被客户端收进日志，关掉横幅更干净
        server.run(transport="stdio", show_banner=False)
    return 0


def _dispatch(args: argparse.Namespace) -> int:
    as_json: bool = getattr(args, "json", False)

    match args.command:
        case "platform":
            _print_model(Env.platform(), as_json=as_json)
        case "cpu":
            info = Env.cpu(interval=args.interval, per_core=args.per_core)
            text = str(info)
            if args.per_core and info.per_core:
                text += "\n- 各核：" + " ".join(f"{v:.0f}%" for v in info.per_core)
            _print_model(info, as_json=as_json, text=text)
        case "memory":
            _print_model(Env.memory(), as_json=as_json)
        case "disk":
            if args.path:
                _print_model(Env.disk_usage(args.path), as_json=as_json)
                return 0
            items = Env.disks(all=args.all)
            if as_json:
                print(jsonable(items))
            else:
                print("\n".join(str(disk) for disk in items) or "没有可用的挂载点")
        case "gpu":
            _print_model(Env.gpus(), as_json=as_json)
        case "net":
            _print_model(Env.network(), as_json=as_json)
        case "io":
            _print_model(Env.disk_io(args.interval), as_json=as_json)
        case "ps":
            limit = args.limit if args.limit > 0 else None
            items = Env.processes(limit=limit, sort=args.sort, name=args.name)
            if as_json:
                print(jsonable(items))
            else:
                for item in items:
                    print(
                        f"{item.pid:>7} {item.name[:24]:<24} "
                        f"CPU {item.cpu_percent:>5.1f}% "
                        f"内存 {item.memory_rss / 1024 / 1024:>9.1f}MB "
                        f"{item.username}"
                    )
        case "proc":
            _print_model(Env.process(args.pid), as_json=as_json)
        case "kill":
            killed = Env.kill(args.pid, force=args.force)
            print(f"{'已结束' if killed else '未结束'}进程 {args.pid}")
        case "run":
            argv = list(args.cmd)
            if argv and argv[0] == "--":
                argv = argv[1:]
            if not argv:
                print("错误：缺少要执行的命令", file=sys.stderr)
                return 2
            result = Env.run(argv, cwd=args.cwd, timeout=args.timeout, stdin=args.stdin)
            if as_json:
                print(result.to_json(indent=2))
            else:
                if result.stdout:
                    print(
                        result.stdout, end="" if result.stdout.endswith("\n") else "\n"
                    )
                if result.stderr:
                    print(result.stderr, end="", file=sys.stderr)
            return 0 if result.ok else result.returncode
        case "which":
            found = Env.which(args.name)
            print(
                found or f"未找到：{args.name}",
                file=sys.stdout if found else sys.stderr,
            )
            return 0 if found else 1
        case "vars":
            values = Env.environ(args.prefix, reveal=args.reveal)
            if as_json:
                print(jsonable(values))
            else:
                for name, value in values.items():
                    print(f"{name}={value}")
        case "snapshot":
            snapshot = Env.snapshot(top=args.top, include_gpu=not args.no_gpu)
            if as_json:
                print(snapshot.to_json(indent=2))
            else:
                print(snapshot.report(top=args.top))
        case "serve":
            return _serve(args)
        case _:
            return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (EnvError, OSError, ValueError) as exc:
        print(f"错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
