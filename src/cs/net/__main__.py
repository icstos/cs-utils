"""命令行入口：``python -m cs.net ...``。

::

    python -m cs.net info
    python -m cs.net interfaces --up-only
    python -m cs.net wifi
    python -m cs.net ip
    python -m cs.net public-ip
    python -m cs.net resolve example.com
    python -m cs.net check 127.0.0.1 8000
    python -m cs.net free 8000
    python -m cs.net free-port --port-range 8000-8100
    python -m cs.net scan 192.168.1.1 22,80,443,8000-8100
    python -m cs.net wait 127.0.0.1 8000 --timeout 30
    python -m cs.net http get https://example.com
    python -m cs.net http post https://httpbin.org/post --json '{"a": 1}'
    python -m cs.net http download https://example.com/a.zip -o out/a.zip
    python -m cs.net static output --port 8000
    python -m cs.net serve --transport http --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .core import Net
from .errors import NetError
from .http import download
from .models import PortRange
from .server import serve_static

__all__ = ["main"]

type Json = dict[str, Any] | list[Any] | str | int | float | bool | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cs.net",
        description="网络工具：本机地址、域名解析、公网 IP、端口探测/扫描、HTTP、静态服务。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---------------------------------------------------------- 服务
    serve = sub.add_parser("serve", help="以 MCP 服务方式启动")
    serve.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--name", default="cs-net")

    # ---------------------------------------------------------- 本机信息
    sub.add_parser("info", help="本机网络概览（主机名、地址、Wi-Fi、网卡）")
    interfaces = sub.add_parser("interfaces", help="列出网卡")
    interfaces.add_argument("--up-only", action="store_true", help="只看已启用的网卡")
    sub.add_parser("wifi", help="当前 Wi-Fi 名称")
    sub.add_parser("ip", help="本机出网 IP")

    public = sub.add_parser("public-ip", help="查询公网出口 IP")
    public.add_argument("--timeout", type=float, default=5.0)

    resolve = sub.add_parser("resolve", help="解析域名")
    resolve.add_argument("host")
    resolve.add_argument("--timeout", type=float, default=5.0)

    # ---------------------------------------------------------- 端口
    check = sub.add_parser("check", help="探测端口是否开放")
    check.add_argument("target", help="主机名、IP，或 host:port")
    check.add_argument("port", nargs="?", type=int, default=None)
    check.add_argument("--timeout", type=float, default=0.5)

    is_free = sub.add_parser("free", help="判断本机端口能否绑定")
    is_free.add_argument("port", type=int)
    is_free.add_argument("--host", default="", help="绑定地址，默认所有接口")

    free_port = sub.add_parser("free-port", help="分配一个空闲端口")
    free_port.add_argument("--host", default="")
    free_port.add_argument("--port-range", default=None, help="候选范围，如 8000-8100")

    scan = sub.add_parser("scan", help="批量扫描端口")
    scan.add_argument("host")
    scan.add_argument("ports", help='端口集合，如 "22,80,8000-8100"')
    scan.add_argument("--timeout", type=float, default=0.3)
    scan.add_argument("--workers", type=int, default=64)
    scan.add_argument("--all", action="store_true", help="同时输出未开放的端口")

    wait = sub.add_parser("wait", help="等待端口开放（或关闭）")
    wait.add_argument("target")
    wait.add_argument("port", nargs="?", type=int, default=None)
    wait.add_argument("--timeout", type=float, default=10.0)
    wait.add_argument("--interval", type=float, default=0.2)
    wait.add_argument("--closed", action="store_true", help="等待端口关闭")

    # ---------------------------------------------------------- HTTP
    http = sub.add_parser("http", help="HTTP 请求 / 下载")
    http_sub = http.add_subparsers(dest="action", required=True)

    http_get = http_sub.add_parser("get", help="GET 请求")
    http_get.add_argument("url")
    http_get.add_argument("--timeout", type=float, default=10.0)
    http_get.add_argument("--limit", type=int, default=2000, help="正文截断长度")
    http_get.add_argument("--full", action="store_true", help="输出完整正文")

    http_post = http_sub.add_parser("post", help="POST 请求")
    http_post.add_argument("url")
    http_post.add_argument("--json", default=None, help="JSON 正文")
    http_post.add_argument("--text", default=None, help="纯文本正文")
    http_post.add_argument("--timeout", type=float, default=10.0)

    http_download = http_sub.add_parser("download", help="下载文件")
    http_download.add_argument("url")
    http_download.add_argument("-o", "--output", required=True)
    http_download.add_argument("--overwrite", action="store_true")
    http_download.add_argument("--timeout", type=float, default=60.0)

    # ---------------------------------------------------------- 静态服务
    static = sub.add_parser("static", help="把目录以 HTTP 共享（Ctrl-C 退出）")
    static.add_argument("directory")
    static.add_argument("--port", type=int, default=8000)
    static.add_argument("--host", default="127.0.0.1")

    return parser


def _dump(payload: Any) -> str:
    """模型 / 列表 / 路径统一序列化成 JSON。"""
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


def _http(args: argparse.Namespace) -> int:
    match args.action:
        case "get":
            response = Net.get(args.url, timeout=args.timeout)
            print(_dump(response.to_dict(full=args.full, limit=args.limit)))
            return 0 if response.ok else 1
        case "post":
            if args.json is not None:
                response = Net.post_json(
                    args.url, json.loads(args.json), timeout=args.timeout
                )
            elif args.text is not None:
                response = Net.request(
                    "POST",
                    args.url,
                    data=args.text.encode("utf-8"),
                    headers={"Content-Type": "text/plain; charset=utf-8"},
                    timeout=args.timeout,
                )
            else:
                print("错误：需要 --json 或 --text", file=sys.stderr)
                return 2
            print(_dump(response.to_dict()))
            return 0 if response.ok else 1
        case _:
            print(
                download(
                    args.url,
                    args.output,
                    overwrite=args.overwrite,
                    timeout=args.timeout,
                )
            )
            return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        match args.command:
            case "serve":
                return _serve(args)
            case "info":
                print(_dump(Net.overview()))
            case "interfaces":
                print(_dump(Net.interfaces(up_only=args.up_only)))
            case "wifi":
                print(Net.wifi() or "")
            case "ip":
                print(Net.primary_ip())
            case "public-ip":
                print(_dump(Net.public_ip(timeout=args.timeout)))
            case "resolve":
                print(_dump(Net.resolve(args.host, timeout=args.timeout)))
            case "check":
                result = Net.check(args.target, args.port, timeout=args.timeout)
                print(_dump(result))
                return 0 if result.is_open else 1
            case "free":
                free = Net.is_free(args.port, host=args.host)
                print(
                    _dump(
                        {
                            "host": args.host or "0.0.0.0",
                            "port": args.port,
                            "free": free,
                        }
                    )
                )
                return 0 if free else 1
            case "free-port":
                candidates = (
                    PortRange.parse(args.port_range) if args.port_range else None
                )
                print(Net.free_port(host=args.host, candidates=candidates))
            case "scan":
                result = Net.scan(
                    args.host, args.ports, timeout=args.timeout, workers=args.workers
                )
                print(_dump(result.to_dict(only_open=not args.all)))
            case "wait":
                result = (
                    Net.wait_closed(
                        args.target,
                        args.port,
                        timeout=args.timeout,
                        interval=args.interval,
                    )
                    if args.closed
                    else Net.wait_open(
                        args.target,
                        args.port,
                        timeout=args.timeout,
                        interval=args.interval,
                    )
                )
                print(_dump(result))
                expected = not args.closed
                return 0 if result.is_open == expected else 1
            case "http":
                return _http(args)
            case "static":
                return _static(args)
    except (NetError, OSError, ValueError) as exc:
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


def _static(args: argparse.Namespace) -> int:
    """先打印 URL 再阻塞，避免用户不知道服务起在哪个端口。"""
    server = serve_static(args.directory, args.port, host=args.host, block=False)
    print(server.url, flush=True)
    server.wait()
    server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
