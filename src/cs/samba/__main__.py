"""命令行入口：``python -m cs.samba ...``。

连接参数既可用命令行，也可用环境变量（``SMB_HOST`` / ``SMB_SHARE`` / ``SMB_USER`` /
``SMB_PASSWORD``），后者更适合把密码排除在进程列表之外::

    export SMB_HOST=nas SMB_SHARE=share SMB_USER=u SMB_PASSWORD=p

    python -m cs.samba            # 等同 info
    python -m cs.samba ls docs
    python -m cs.samba ls docs -l --all
    python -m cs.samba mtime docs/a.pdf
    python -m cs.samba get docs/a.pdf a.pdf
    python -m cs.samba put b.pdf docs/b.pdf
    python -m cs.samba mkdir docs/2026/10
    python -m cs.samba tree docs -d 2
    python -m cs.samba find "*.log" docs
    python -m cs.samba serve --transport http --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from .client import SMBClient
from .errors import SambaConfigError, SambaError
from .models import SMBOptions, human_age, human_size
from .views import render_table, render_tree

__all__ = ["main"]

# smbprotocol 在正常断开时会打 "socket aborted by peer ... treating as EOF"，
# 对命令行用户是纯噪音；只压这一个 logger，不动全局日志配置。
logging.getLogger("smbprotocol").setLevel(logging.ERROR)


def _emit(payload: dict[str, Any], *, as_json: bool, text: str) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(text)


def _conn_parser() -> argparse.ArgumentParser:
    """各子命令共用的连接参数（``add_help=False`` 便于作为 parent）。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-H", "--server", default=None, help="主机名/IP（或 SMB_HOST）")
    parser.add_argument("-S", "--share", default=None, help="共享名（或 SMB_SHARE）")
    parser.add_argument("-u", "--user", default=None, help="用户名（或 SMB_USER）")
    parser.add_argument(
        "-p", "--password", default=None, help="密码（或 SMB_PASSWORD，更安全）"
    )
    parser.add_argument("--domain", default=None, help="NTLM 域")
    parser.add_argument("--port", type=int, default=None, help="端口，默认 445")
    parser.add_argument("--timeout", type=float, default=None, help="超时秒数")
    parser.add_argument(
        "--guest", action="store_true", help="允许 guest/匿名会话（放宽签名校验）"
    )
    parser.add_argument("--no-signing", action="store_true", help="不要求消息签名")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    return parser


def _build_parser() -> argparse.ArgumentParser:
    conn = _conn_parser()
    parser = argparse.ArgumentParser(
        prog="python -m cs.samba",
        description="SMB/Samba 共享操作：列目录、查时间、上传下载、建目录、改名、删除。",
    )
    sub = parser.add_subparsers(dest="command")

    def cmd(name: str, help_text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, parents=[conn], help=help_text)

    cmd("info", help_text="连接信息（方言、签名、登录用户）")

    ls = cmd("ls", help_text="列出目录")
    ls.add_argument("path", nargs="?", default="")
    ls.add_argument("--pattern", default=None, help="文件名 glob，如 '*.pdf'")
    ls.add_argument("-a", "--all", action="store_true", help="包含隐藏项")
    ls.add_argument("--sort", choices=("name", "size", "mtime"), default="name")

    stat = cmd("stat", help_text="查看属性")
    stat.add_argument("path")

    mtime = cmd("mtime", help_text="查看最后更新时间")
    mtime.add_argument("path")

    cat = cmd("cat", help_text="读取文本文件")
    cat.add_argument("path")
    cat.add_argument("--encoding", default="utf-8")
    cat.add_argument(
        "--errors",
        choices=("strict", "replace", "ignore"),
        default="strict",
        help="解码出错时的处理方式，默认 strict",
    )
    cat.add_argument("--max-bytes", type=int, default=200_000)

    find = cmd("find", help_text="按名字查找")
    find.add_argument("pattern")
    find.add_argument("path", nargs="?", default="")
    find.add_argument("--no-recursive", action="store_true")
    find.add_argument("-a", "--all", action="store_true", help="包含隐藏项")

    tree = cmd("tree", help_text="树形概览")
    tree.add_argument("path", nargs="?", default="")
    tree.add_argument("-d", "--depth", type=int, default=2)
    tree.add_argument("-a", "--all", action="store_true")

    get = cmd("get", help_text="下载文件")
    get.add_argument("remote")
    get.add_argument("local")
    get.add_argument("-f", "--overwrite", action="store_true")
    get.add_argument("--resume", action="store_true")
    get.add_argument("--verify", action="store_true")

    put = cmd("put", help_text="上传文件")
    put.add_argument("local")
    put.add_argument("remote", nargs="?", default="")
    put.add_argument("-f", "--overwrite", action="store_true")
    put.add_argument("--resume", action="store_true")
    put.add_argument("--verify", action="store_true")

    mkdir = cmd("mkdir", help_text="创建目录")
    mkdir.add_argument("path")
    mkdir.add_argument("--no-parents", action="store_true", help="只建一层")

    rm = cmd("rm", help_text="删除文件")
    rm.add_argument("path")
    rm.add_argument("-f", "--force", action="store_true", help="不存在也不报错")

    rmdir = cmd("rmdir", help_text="删除目录")
    rmdir.add_argument("path")
    rmdir.add_argument("-r", "--recursive", action="store_true")
    rmdir.add_argument("-f", "--force", action="store_true")

    mv = cmd("mv", help_text="重命名/移动")
    mv.add_argument("source")
    mv.add_argument("target")
    mv.add_argument("-f", "--overwrite", action="store_true")

    serve = sub.add_parser("serve", help="以 MCP 服务方式启动")
    serve.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--name", default="cs-samba")

    return parser


def _options(args: argparse.Namespace) -> SMBOptions:
    """命令行优先，其次环境变量；缺 server/share 时给出可执行的提示。

    用 ``vars(args).get`` 而不是直接取属性：无子命令时 argparse 只填 ``command``，
    连接参数根本没被注册，硬取属性会抛 ``AttributeError`` 而不是可读的配置错误。
    """
    values = vars(args)
    provided: dict[str, Any] = {
        field: values[dest]
        for dest, field in (
            ("server", "server"),
            ("share", "share"),
            ("user", "username"),
            ("password", "password"),
            ("domain", "domain"),
            ("port", "port"),
            ("timeout", "timeout"),
        )
        if values.get(dest) is not None
    }
    if values.get("guest"):
        provided["allow_guest"] = True
    if values.get("no_signing"):
        provided["require_signing"] = False

    if "server" not in provided:
        if not (os.environ.get("SMB_HOST") or os.environ.get("SMB_SERVER")):
            raise SambaConfigError("缺少 --server（或设置环境变量 SMB_HOST）")
        return SMBOptions.from_env(**provided)

    server = provided.pop("server")
    share = provided.pop("share", os.environ.get("SMB_SHARE", ""))
    if not share:
        raise SambaConfigError("缺少 --share 或环境变量 SMB_SHARE")
    return SMBOptions(server=server, share=share).replace(**provided)


@contextmanager
def _client(args: argparse.Namespace) -> Iterator[SMBClient]:
    client = SMBClient(_options(args))
    try:
        yield client
    finally:
        client.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = args.command or "info"

    try:
        match command:
            case "serve":
                from .mcp_server import create_server

                server = create_server(args.name)
                if args.transport == "http":
                    server.run(transport="http", host=args.host, port=args.port)
                else:
                    server.run(transport="stdio", show_banner=False)
                return 0

            case "info":
                with _client(args) as client:
                    info = client.info()
                    _emit(
                        info.to_dict(),
                        as_json=args.json,
                        text=(
                            f"{info}\n"
                            f"用户: {info.username or 'anonymous'}\n"
                            f"连接于: {info.connected_at.astimezone():%Y-%m-%d %H:%M:%S}"
                        ),
                    )
                return 0

            case "ls":
                with _client(args) as client:
                    entries = client.list_dir(
                        args.path,
                        pattern=args.pattern,
                        include_hidden=args.all,
                        sort=args.sort,
                    )
                    _emit(
                        {
                            "path": client.resolve(args.path).posix or "/",
                            "count": len(entries),
                            "entries": [entry.to_dict() for entry in entries],
                        },
                        as_json=args.json,
                        text=render_table(entries),
                    )
                return 0

            case "stat":
                with _client(args) as client:
                    entry = client.stat(args.path)
                    stamp = (
                        f"{entry.mtime_local:%Y-%m-%d %H:%M:%S}"
                        if entry.mtime
                        else "未知"
                    )
                    _emit(
                        entry.to_dict(),
                        as_json=args.json,
                        text=(
                            f"路径: {entry.path or '/'}\n"
                            f"类型: {'目录' if entry.is_dir else '文件'}\n"
                            f"大小: {human_size(entry.size) if entry.is_file else '-'}\n"
                            f"修改: {stamp}"
                        ),
                    )
                return 0

            case "mtime":
                with _client(args) as client:
                    entry = client.stat(args.path)
                    if entry.mtime is None:
                        raise SambaError(f"服务端未提供修改时间: {entry.path}")
                    age = entry.age()
                    _emit(
                        {
                            "path": entry.path,
                            "mtime": entry.mtime_local.isoformat(timespec="seconds"),
                            "mtime_utc": entry.mtime.isoformat(timespec="seconds"),
                            "mtime_epoch": entry.mtime_epoch,
                            "age_seconds": round(age.total_seconds(), 1)
                            if age
                            else None,
                            "age_human": human_age(age),
                        },
                        as_json=args.json,
                        text=(
                            f"{entry.path}: {entry.mtime_local:%Y-%m-%d %H:%M:%S} "
                            f"（{human_age(age)}）"
                        ),
                    )
                return 0

            case "cat":
                with _client(args) as client:
                    print(
                        client.read_text(
                            args.path,
                            encoding=args.encoding,
                            errors=args.errors,
                            max_bytes=args.max_bytes,
                        )
                    )
                return 0

            case "find":
                with _client(args) as client:
                    entries = client.find(
                        args.pattern,
                        args.path,
                        recursive=not args.no_recursive,
                        include_hidden=args.all,
                    )
                    _emit(
                        {
                            "count": len(entries),
                            "entries": [e.to_dict() for e in entries],
                        },
                        as_json=args.json,
                        text=render_table(entries),
                    )
                return 0

            case "tree":
                with _client(args) as client:
                    if args.json:
                        entries = list(
                            client.iter_entries(args.path, include_hidden=args.all)
                        )
                        _emit(
                            {
                                "count": len(entries),
                                "entries": [e.to_dict() for e in entries],
                            },
                            as_json=True,
                            text="",
                        )
                    else:
                        print(
                            render_tree(
                                client,
                                args.path,
                                max_depth=args.depth,
                                include_hidden=args.all,
                            )
                        )
                return 0

            case "get":
                with _client(args) as client:
                    result = client.download(
                        args.remote,
                        args.local,
                        overwrite=args.overwrite,
                        resume=args.resume,
                        verify=args.verify,
                    )
                    _emit(result.to_dict(), as_json=args.json, text=str(result))
                return 0

            case "put":
                with _client(args) as client:
                    result = client.upload(
                        args.local,
                        args.remote,
                        overwrite=args.overwrite,
                        resume=args.resume,
                        verify=args.verify,
                    )
                    _emit(result.to_dict(), as_json=args.json, text=str(result))
                return 0

            case "mkdir":
                with _client(args) as client:
                    created = client.mkdir(args.path, parents=not args.no_parents)
                    _emit(
                        {"path": client.resolve(args.path).posix, "created": created},
                        as_json=args.json,
                        text=f"已创建 {created} 层目录: {client.resolve(args.path).posix}",
                    )
                return 0

            case "rm":
                with _client(args) as client:
                    removed = client.remove(args.path, missing_ok=args.force)
                    _emit(
                        {"path": client.resolve(args.path).posix, "removed": removed},
                        as_json=args.json,
                        text=f"{'已删除' if removed else '不存在'} {client.resolve(args.path).posix}",
                    )
                return 0

            case "rmdir":
                with _client(args) as client:
                    removed = client.rmdir(
                        args.path, recursive=args.recursive, missing_ok=args.force
                    )
                    _emit(
                        {"path": client.resolve(args.path).posix, "removed": removed},
                        as_json=args.json,
                        text=f"已删除 {removed} 项（含目录本身）",
                    )
                return 0

            case "mv":
                with _client(args) as client:
                    target = client.rename(
                        args.source, args.target, overwrite=args.overwrite
                    )
                    _emit(
                        {"target": target.posix},
                        as_json=args.json,
                        text=f"已重命名 -> {target.posix}",
                    )
                return 0

            case _:  # pragma: no cover - argparse 已限制取值范围
                parser.print_help()
                return 2

    except SambaError as exc:
        print(f"错误: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except UnicodeDecodeError as exc:
        print(
            f"错误: 内容不是 {exc.encoding} 文本（偏移 {exc.start}），"
            "常见于二进制文件；可加 --errors replace 强制解码，或改用 get 下载",
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        # 本地文件系统问题（磁盘满、路径非法、权限不足等），同样不给用户看 traceback
        print(f"错误: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
