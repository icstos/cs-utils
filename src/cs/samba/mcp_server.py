"""把 :mod:`cs.samba` 暴露为 MCP 工具（FastMCP ≥ 4）。

启动方式::

    python -m cs.samba serve                 # stdio，供 MCP 客户端拉起
    python -m cs.samba serve --transport http --port 8000
    fastmcp run src/cs/samba/mcp_server.py:mcp

约定
----
* 工具名统一 ``samba_*``；写操作默认 ``overwrite=False``，不会静默覆盖用户文件；
* 会话模型：先 ``samba_open`` 得到会话（默认名 ``default``），后续工具复用连接，
  不必每次传账号密码；空闲 15 分钟自动回收；
* 成功返回 ``{"ok": true, ...}``，业务异常统一转成 ``ToolError``；
* 密码只用于建连，任何返回值里都不会出现明文。
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from .client import SMBClient
from .errors import SambaError
from .models import SMBOptions, human_age, human_size
from .sessions import DEFAULT_SESSION, LiveSession, SessionRegistry
from .views import render_tree

__all__ = ["create_server", "mcp"]

_SERVER_NAME = "cs-samba"
_MAX_READ = 1_000_000
_INSTRUCTIONS = """\
本服务器读写 SMB/Samba 共享（SMB 2.0.2 ~ 3.1.1）：列目录、查属性与最后更新时间、
下载上传、建目录、改名、删除、读写文本、按名字查找。

使用建议：
1. 先调用 samba_open(server, share, username, password) 建立会话；之后所有工具
   默认复用名为 "default" 的会话，多台服务器可用 session 参数区分；
2. 路径一律相对共享根，写 "docs/a.pdf" 即可，也接受 \\\\server\\share\\docs\\a.pdf；
3. 回答"文件什么时候更新的"用 samba_mtime；看目录内容用 samba_list（可传 pattern）；
4. 下载/上传的本地路径是本机路径，默认不覆盖；需要覆盖显式传 overwrite=true；
5. 不确定共享里有什么时，先用 samba_tree 看两级结构，再决定深入哪个目录；
6. samba_close 主动释放连接，不调用也会在空闲 15 分钟后回收。
"""


def _guard[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """把领域异常转换为 MCP 可读错误。"""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except (SambaError, OSError, ValueError) as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


def _done(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def _target(pool: SessionRegistry, session: str) -> SMBClient:
    live: LiveSession = pool.get(session)
    return live.client


def create_server(
    name: str = _SERVER_NAME,
    *,
    instructions: str = _INSTRUCTIONS,
    registry: SessionRegistry | None = None,
) -> FastMCP:
    """构造服务器实例；``registry`` 可注入，便于测试隔离。"""
    app: FastMCP = FastMCP(name, instructions=instructions)
    pool = registry if registry is not None else SessionRegistry()

    # ---------------------------------------------------------- 会话管理
    @app.tool(annotations={"readOnlyHint": False})
    @_guard
    def samba_open(
        server: str,
        share: str,
        username: str = "",
        password: str = "",
        domain: str = "",
        port: int = 445,
        encrypt: bool = False,
        allow_guest: bool = False,
        timeout: float = 30.0,
        session: str = DEFAULT_SESSION,
    ) -> dict[str, Any]:
        """连接一个 SMB 共享，返回会话名与协商结果（方言、签名、加密）。

        同一 session 名重复调用会顶替旧连接，换账号或换共享时可直接重开。
        username 留空表示匿名/guest 访问；domain 用于 DOMAIN\\user 形式。
        """
        options = SMBOptions(
            server=server,
            share=share,
            username=username,
            password=password,
            domain=domain,
            port=port,
            timeout=timeout,
            encrypt=encrypt,
            allow_guest=allow_guest,
        )
        live = pool.open(options, name=session)
        return _done(session=live.name, **live.client.info().to_dict())

    @app.tool(annotations={"readOnlyHint": False})
    @_guard
    def samba_close(session: str = DEFAULT_SESSION) -> dict[str, Any]:
        """关闭会话并释放连接。"""
        closed = pool.close(session)
        return _done(session=session, closed=closed)

    @app.tool(annotations={"readOnlyHint": True})
    @_guard
    def samba_sessions() -> dict[str, Any]:
        """列出当前所有会话及其目标共享、空闲时长。"""
        items = pool.snapshot()
        return _done(count=len(items), sessions=items)

    @app.tool(annotations={"readOnlyHint": True})
    @_guard
    def samba_info(session: str = DEFAULT_SESSION) -> dict[str, Any]:
        """查看当前会话的连接详情（UNC、方言、签名/加密、登录用户）。"""
        live = pool.get(session)
        return _done(**live.info())

    # ---------------------------------------------------------- 浏览
    @app.tool(annotations={"readOnlyHint": True})
    @_guard
    def samba_list(
        path: str = "",
        pattern: str | None = None,
        include_hidden: bool = False,
        sort: str = "name",
        session: str = DEFAULT_SESSION,
    ) -> dict[str, Any]:
        """列出目录直接子项（名字、类型、大小、最后修改时间）。

        pattern 支持 glob（如 "*.pdf"）；sort 可选 name / size / mtime。
        """
        client = _target(pool, session)
        entries = client.list_dir(
            path,
            pattern=pattern,
            include_hidden=include_hidden,
            sort=sort,  # type: ignore[arg-type]
        )
        files = [entry for entry in entries if entry.is_file]
        return _done(
            path=client.resolve(path).posix or "/",
            count=len(entries),
            files=len(files),
            dirs=len(entries) - len(files),
            total_bytes=sum(entry.size for entry in files),
            total_human=human_size(sum(entry.size for entry in files)),
            entries=[entry.to_dict() for entry in entries],
        )

    @app.tool(annotations={"readOnlyHint": True})
    @_guard
    def samba_tree(
        path: str = "",
        max_depth: int = 2,
        include_hidden: bool = False,
        session: str = DEFAULT_SESSION,
    ) -> dict[str, Any]:
        """以树形文本概览目录结构（默认两层，避免刷屏）。"""
        if not 1 <= max_depth <= 6:
            raise ValueError("max_depth 必须在 1~6 之间")
        client = _target(pool, session)
        return _done(
            tree=render_tree(
                client, path, max_depth=max_depth, include_hidden=include_hidden
            )
        )

    @app.tool(annotations={"readOnlyHint": True})
    @_guard
    def samba_stat(path: str, session: str = DEFAULT_SESSION) -> dict[str, Any]:
        """查看文件/目录的详细属性（大小、时间戳、属性位、是否隐藏）。"""
        client = _target(pool, session)
        return _done(**client.stat(path).to_dict())

    @app.tool(annotations={"readOnlyHint": True})
    @_guard
    def samba_mtime(path: str, session: str = DEFAULT_SESSION) -> dict[str, Any]:
        """查询文件最后更新时间（本地时间 / UTC / UNIX 时间戳 / 距今多久）。"""
        client = _target(pool, session)
        entry = client.stat(path)
        if entry.mtime is None:
            raise ValueError(f"服务端未提供修改时间: {entry.path}")
        age = entry.age()
        return _done(
            path=entry.path,
            name=entry.name,
            type="dir" if entry.is_dir else "file",
            size=entry.size,
            mtime=entry.mtime_local.isoformat(timespec="seconds"),  # type: ignore[union-attr]
            mtime_utc=entry.mtime.isoformat(timespec="seconds"),  # type: ignore[union-attr]
            mtime_epoch=entry.mtime_epoch,
            age_seconds=round(age.total_seconds(), 1) if age else None,
            age_human=human_age(age),
        )

    @app.tool(annotations={"readOnlyHint": True})
    @_guard
    def samba_find(
        pattern: str,
        path: str = "",
        recursive: bool = True,
        include_hidden: bool = False,
        session: str = DEFAULT_SESSION,
    ) -> dict[str, Any]:
        """按文件名 glob 查找（"*.log"、"report-2026-??.pdf"），返回匹配条目。"""
        client = _target(pool, session)
        entries = client.find(
            pattern, path, recursive=recursive, include_hidden=include_hidden
        )
        return _done(count=len(entries), entries=[entry.to_dict() for entry in entries])

    @app.tool(annotations={"readOnlyHint": True})
    @_guard
    def samba_read_text(
        path: str,
        encoding: str = "utf-8",
        max_bytes: int = 200_000,
        session: str = DEFAULT_SESSION,
    ) -> dict[str, Any]:
        """读取文本文件（超过 max_bytes 会截断并置 truncated=true）。"""
        if max_bytes < 1:
            raise ValueError("max_bytes 必须为正整数")
        client = _target(pool, session)
        entry = client.stat(path)
        limit = min(max_bytes, entry.size)
        text = client.read_text(path, encoding=encoding, max_bytes=limit)
        return _done(
            path=entry.path,
            chars=len(text),
            bytes_read=min(limit, entry.size),
            size=entry.size,
            truncated=entry.size > limit,
            text=text,
        )

    # ---------------------------------------------------------- 传输
    @app.tool(annotations={"readOnlyHint": False})
    @_guard
    def samba_download(
        remote_path: str,
        local_path: str,
        overwrite: bool = False,
        resume: bool = False,
        verify: bool = False,
        session: str = DEFAULT_SESSION,
    ) -> dict[str, Any]:
        """从共享下载文件到本机（先写临时文件再原子替换）。

        local_path 是本机路径；默认不覆盖，resume=true 时按已有长度续传。
        """
        client = _target(pool, session)
        result = client.download(
            remote_path, local_path, overwrite=overwrite, resume=resume, verify=verify
        )
        return _done(**result.to_dict())

    @app.tool(annotations={"readOnlyHint": False})
    @_guard
    def samba_upload(
        local_path: str,
        remote_path: str,
        overwrite: bool = False,
        resume: bool = False,
        verify: bool = False,
        session: str = DEFAULT_SESSION,
    ) -> dict[str, Any]:
        """上传本机文件到共享（远端父目录会自动创建）。

        remote_path 指向已存在的目录时，文件名沿用本地文件名。
        """
        client = _target(pool, session)
        result = client.upload(
            local_path, remote_path, overwrite=overwrite, resume=resume, verify=verify
        )
        return _done(**result.to_dict())

    @app.tool(annotations={"readOnlyHint": False})
    @_guard
    def samba_write_text(
        path: str,
        content: str,
        encoding: str = "utf-8",
        overwrite: bool = True,
        session: str = DEFAULT_SESSION,
    ) -> dict[str, Any]:
        """把文本写入共享文件（默认覆盖；父目录不存在会自动创建）。"""
        client = _target(pool, session)
        written = client.write_text(
            path, content, encoding=encoding, overwrite=overwrite
        )
        return _done(path=client.resolve(path).posix, bytes=written)

    # ---------------------------------------------------------- 修改
    @app.tool(annotations={"readOnlyHint": False})
    @_guard
    def samba_mkdir(
        path: str,
        parents: bool = True,
        session: str = DEFAULT_SESSION,
    ) -> dict[str, Any]:
        """创建目录（默认递归创建，已存在视为成功），返回新建层数。"""
        client = _target(pool, session)
        created = client.mkdir(path, parents=parents, exist_ok=True)
        return _done(path=client.resolve(path).posix, created=created)

    @app.tool(annotations={"readOnlyHint": False})
    @_guard
    def samba_rename(
        source: str,
        target: str,
        overwrite: bool = False,
        session: str = DEFAULT_SESSION,
    ) -> dict[str, Any]:
        """重命名或移动文件/目录（同一共享内）。"""
        client = _target(pool, session)
        result = client.rename(source, target, overwrite=overwrite)
        return _done(source=source, target=result.posix)

    @app.tool(annotations={"readOnlyHint": False})
    @_guard
    def samba_remove(
        path: str, missing_ok: bool = False, session: str = DEFAULT_SESSION
    ) -> dict[str, Any]:
        """删除文件（不删目录；目录请用 samba_rmdir）。"""
        client = _target(pool, session)
        removed = client.remove(path, missing_ok=missing_ok)
        return _done(path=client.resolve(path).posix, removed=removed)

    @app.tool(annotations={"readOnlyHint": False})
    @_guard
    def samba_rmdir(
        path: str,
        recursive: bool = False,
        missing_ok: bool = False,
        session: str = DEFAULT_SESSION,
    ) -> dict[str, Any]:
        """删除目录；recursive=true 时先自底向上清空内容（风险操作，慎用）。"""
        client = _target(pool, session)
        removed = client.rmdir(path, recursive=recursive, missing_ok=missing_ok)
        return _done(path=client.resolve(path).posix, removed=removed)

    return app


#: 模块级实例，供 ``fastmcp run src/cs/samba/mcp_server.py:mcp`` 直接引用
mcp = create_server()
