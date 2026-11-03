"""把 :mod:`cs.net` 暴露为 MCP 工具（FastMCP ≥ 4）。

启动方式::

    python -m cs.net serve                 # stdio，供 MCP 客户端拉起
    python -m cs.net serve --transport http --port 8000
    fastmcp run src/cs/net/mcp_server.py:mcp

约定：工具名统一 ``net_*``；只读探测标注 ``readOnlyHint``；业务异常统一转
``ToolError``，不把 traceback 丢给模型。
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from .core import Net
from .errors import NetError
from .models import PortRange
from .server import StaticServer

__all__ = ["create_server", "mcp"]

_SERVER_NAME = "cs-net"
_INSTRUCTIONS = """\
本服务器提供网络相关能力：本机地址/网卡/Wi-Fi、域名解析、公网 IP、端口探测与
分配、端口批量扫描、HTTP 请求，以及把本地目录临时共享为 HTTP 服务。

使用建议：
1. 先 net_info 或 net_interfaces 了解本机网络，再决定后续动作；
2. 区分两个概念：port_free 表示「本机能不能绑这个端口」，port_check 的 open
   表示「对端有没有服务在监听」——被占用但无人监听时两者都为 false；
3. net_scan 的 ports 支持 "8000-8100" 或 [80, 443] 形式，单次上限 4096 个；
4. net_wait_port 会阻塞到端口就绪或超时，超时不算错误，看 returned.open；
5. net_http_get 返回 status/headers/text（text 默认截断 2000 字符），
   4xx/5xx 不抛错；
6. net_static_start 返回可直接打开的 URL，用完请 net_static_stop 释放端口。
"""

#: 由本进程启动的静态服务，按端口索引
_SERVERS: dict[int, StaticServer] = {}
_LOCK = threading.Lock()


def _guard[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """把领域异常转换成 MCP 可读错误。"""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except (NetError, OSError, ValueError) as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


def _done(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def create_server(
    name: str = _SERVER_NAME, *, instructions: str = _INSTRUCTIONS
) -> FastMCP:
    """构造一个独立的服务器实例（便于测试或多实例部署）。"""
    server: FastMCP = FastMCP(name, instructions=instructions)

    # ---------------------------------------------------------- 本机信息
    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def net_info() -> dict[str, Any]:
        """本机网络概览：主机名、FQDN、出网 IP、局域网地址、Wi-Fi、网卡列表。"""
        return _done(**Net.overview())

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def net_interfaces(up_only: bool = False) -> dict[str, Any]:
        """列出网卡：名称、IPv4/IPv6、MAC、是否启用、速率、MTU。"""
        return _done(
            interfaces=[item.to_dict() for item in Net.interfaces(up_only=up_only)]
        )

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def net_wifi() -> dict[str, Any]:
        """当前连接的 Wi-Fi 名称（SSID），未连接时 connected=false。"""
        name_ = Net.wifi()
        return _done(connected=bool(name_), ssid=name_)

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def net_ip() -> dict[str, Any]:
        """本机出网 IP 与各网卡的局域网地址。"""
        info = Net.host_info()
        return _done(
            primary_ip=info.primary_ip,
            ipv4=list(info.ipv4),
            ipv6=list(info.ipv6),
        )

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def net_public_ip(timeout: float = 5.0) -> dict[str, Any]:
        """查询公网出口 IP（依次尝试多个公共服务，需要外网）。"""
        return _done(**Net.public_ip(timeout=timeout).to_dict())

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def net_resolve(host: str, timeout: float | None = 5.0) -> dict[str, Any]:
        """解析域名或 IP，返回 A/AAAA 地址列表与规范名。"""
        return _done(**Net.resolve(host, timeout=timeout).to_dict())

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def net_reverse(ip: str) -> dict[str, Any]:
        """IP 反向解析（PTR），没有记录时抛错。"""
        return _done(ip=ip, hostname=Net.reverse(ip))

    # ---------------------------------------------------------- 端口
    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def net_port_check(
        target: str, port: int | None = None, timeout: float = 0.5
    ) -> dict[str, Any]:
        """探测一个端口：open / closed / filtered（超时无响应）/ error。

        target 可以是主机名、IP，或 "host:port"（此时 port 可省略；两者都给时以 port 为准）。
        """
        return _done(**Net.check(target, port, timeout=timeout).to_dict())

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def net_port_free(port: int, host: str = "") -> dict[str, Any]:
        """判断本机能否绑定该端口（free）以及是否已有服务在监听（open）。"""
        probe = "127.0.0.1" if host in {"", "0.0.0.0", "::"} else host
        return _done(
            host=host or "0.0.0.0",
            port=port,
            free=Net.is_free(port, host=host),
            open=Net.is_open(probe, port, timeout=0.3),
        )

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def net_free_port(host: str = "", port_range: str | None = None) -> dict[str, Any]:
        """分配一个空闲端口；给 port_range（如 "8000-8100"）时在该范围内按序挑。"""
        candidates = PortRange.parse(port_range) if port_range else None
        return _done(
            host=host or "0.0.0.0",
            port=Net.free_port(host=host, candidates=candidates),
        )

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def net_scan(
        host: str,
        ports: str | list[int],
        timeout: float = 0.3,
        workers: int = 64,
        only_open: bool = True,
    ) -> dict[str, Any]:
        """并发扫描一批端口，ports 支持 "8000-8100" 或 [80, 443]。"""
        targets: str | list[int] = ports
        result = Net.scan(host, targets, timeout=timeout, workers=workers)
        return _done(**result.to_dict(only_open=only_open))

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def net_wait_port(
        target: str,
        port: int | None = None,
        timeout: float = 10.0,
        interval: float = 0.2,
    ) -> dict[str, Any]:
        """轮询等待端口开放，返回最后一次探测结果（超时不抛错）。"""
        result = Net.wait_open(target, port, timeout=timeout, interval=interval)
        return _done(**result.to_dict())

    # ---------------------------------------------------------- HTTP
    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def net_http_get(
        url: str, timeout: float = 10.0, limit: int = 2000
    ) -> dict[str, Any]:
        """GET 一个 URL，返回状态码、响应头与正文（默认截断到 limit 字符）。"""
        return _done(**Net.get(url, timeout=timeout).to_dict(limit=limit))

    @server.tool()
    @_guard
    def net_http_post(
        url: str,
        payload: dict[str, Any] | None = None,
        text: str | None = None,
        timeout: float = 10.0,
        limit: int = 2000,
    ) -> dict[str, Any]:
        """POST 请求：payload 按 JSON 发送，text 按纯文本发送（二选一）。"""
        if payload is not None:
            response = Net.post_json(url, payload, timeout=timeout)
        elif text is not None:
            response = Net.request(
                "POST",
                url,
                data=text.encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8"},
                timeout=timeout,
            )
        else:
            raise ToolError("ValueError: payload 与 text 至少提供一个")
        return _done(**response.to_dict(limit=limit))

    # ---------------------------------------------------------- 静态服务
    @server.tool()
    @_guard
    def net_static_start(
        directory: str, port: int = 0, host: str = "127.0.0.1"
    ) -> dict[str, Any]:
        """把本地目录以 HTTP 共享出去（后台线程），返回可访问的 URL。

        port=0 表示由系统分配；同一端口不要重复启动，返回 reused=true。
        """
        with _LOCK:
            existing = _SERVERS.get(port)
            if existing is not None and existing.running:
                return _done(**existing.to_dict(), reused=True)
            server = StaticServer(directory, port=port, host=host).start()
            _SERVERS[server.port] = server
        return _done(**server.to_dict(), reused=False)

    @server.tool(annotations={"idempotentHint": True})
    @_guard
    def net_static_stop(port: int) -> dict[str, Any]:
        """停止由 net_static_start 启动的服务并释放端口。"""
        with _LOCK:
            server = _SERVERS.pop(port, None)
        if server is None:
            return _done(
                port=port, stopped=False, detail="该端口没有由本服务启动的静态服务"
            )
        server.stop()
        return _done(port=port, stopped=True)

    return server


mcp = create_server()
