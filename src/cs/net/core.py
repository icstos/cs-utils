"""``Net`` 门面：地址、端口、HTTP、静态服务四组能力收进一个类。

方法全部是静态方法（无状态），既方便脚本调用，也方便 MCP 层薄封装：

>>> from cs.net import Net
>>> Net.primary_ip()                      # doctest: +SKIP
'192.168.1.23'
>>> Net.is_open("127.0.0.1", 8000)        # doctest: +SKIP
True
>>> Net.free_port()                       # doctest: +SKIP
53124
"""

from __future__ import annotations

import socket
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .dns import resolve, reverse
from .http import (
    DEFAULT_TIMEOUT,
    PUBLIC_IP_PROVIDERS,
    download,
    get,
    post_json,
    public_ip,
)
from .http import request as http_request
from .local import host_info, interfaces, local_addresses, primary_ip, wifi_name
from .models import (
    Address,
    DnsRecord,
    HostInfo,
    HttpResponse,
    Interface,
    PortCheck,
    PortRange,
    PortScan,
    PortState,
    PublicIP,
)
from .ports import (
    DEFAULT_INTERVAL,
    check,
    free_port,
    is_free,
    is_open,
    scan,
    wait_closed,
    wait_open,
)
from .ports import (
    DEFAULT_TIMEOUT as PORT_TIMEOUT,
)
from .server import StaticServer, serve_static

__all__ = ["Net"]

type StrPath = str | Path


class Net:
    """网络门面。"""

    __slots__ = ()

    #: 常用类型挂在门面上，便于 ``Net.Address.parse(...)`` / ``Net.PortState.OPEN``
    Address = Address
    PortRange = PortRange
    PortState = PortState

    # ---------------------------------------------------------- 本机
    @staticmethod
    def host_info() -> HostInfo:
        """主机名 + 局域网地址 + 出网源 IP。"""
        return host_info()

    @staticmethod
    def interfaces(*, up_only: bool = False) -> list[Interface]:
        """网卡列表（默认「启用优先 + 名称」排序）。"""
        return interfaces(up_only=up_only)

    @staticmethod
    def wifi() -> str | None:
        """当前 Wi-Fi 名称（SSID），未连接返回 ``None``。"""
        return wifi_name()

    @staticmethod
    def primary_ip() -> str:
        """本机出网使用的源 IP（断网时给 ``127.0.0.1``）。"""
        return primary_ip()

    @staticmethod
    def local_addresses(
        *, include_loopback: bool = False
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """``(ipv4, ipv6)`` 已启用网卡上的地址。"""
        return local_addresses(include_loopback=include_loopback)

    @staticmethod
    def overview() -> dict[str, Any]:
        """本机网络概览：主机信息 + Wi-Fi + 网卡，适合一次性交给 MCP 客户端。"""
        return {
            **host_info().to_dict(),
            "wifi": wifi_name(),
            "interfaces": [item.to_dict() for item in interfaces()],
        }

    # ---------------------------------------------------------- DNS / 公网
    @staticmethod
    def resolve(
        host: str, *, family: int = socket.AF_UNSPEC, timeout: float | None = None
    ) -> DnsRecord:
        """解析域名（``family`` 默认 A + AAAA）。"""
        return resolve(host, family=family, timeout=timeout)

    @staticmethod
    def reverse(ip: str, *, timeout: float | None = None) -> str:
        """IP 反查主机名。"""
        return reverse(ip, timeout=timeout)

    @staticmethod
    def public_ip(
        *, timeout: float = 5.0, providers: Sequence[str] = PUBLIC_IP_PROVIDERS
    ) -> PublicIP:
        """公网出口 IP（按提供方顺序重试）。"""
        return public_ip(timeout=timeout, providers=providers)

    # ---------------------------------------------------------- 端口
    @staticmethod
    def check(
        target: Address | str,
        port: int | None = None,
        *,
        timeout: float = PORT_TIMEOUT,
    ) -> PortCheck:
        """探测端口（open / closed / filtered / error）。"""
        return check(target, port, timeout=timeout)

    @staticmethod
    def is_open(
        target: Address | str,
        port: int | None = None,
        *,
        timeout: float = PORT_TIMEOUT,
    ) -> bool:
        """端口是否已开放。"""
        return is_open(target, port, timeout=timeout)

    @staticmethod
    def is_free(port: int, *, host: str = "") -> bool:
        """本机能否绑定该端口。"""
        return is_free(port, host=host)

    @staticmethod
    def free_port(*, host: str = "", candidates: PortRange | None = None) -> int:
        """分配一个空闲端口。"""
        return free_port(host=host, candidates=candidates)

    @staticmethod
    def wait_open(
        target: Address | str,
        port: int | None = None,
        *,
        timeout: float = 10.0,
        interval: float = DEFAULT_INTERVAL,
    ) -> PortCheck:
        """轮询等待端口开放（超时不抛异常，看返回值）。"""
        return wait_open(target, port, timeout=timeout, interval=interval)

    @staticmethod
    def wait_closed(
        target: Address | str,
        port: int | None = None,
        *,
        timeout: float = 10.0,
        interval: float = DEFAULT_INTERVAL,
    ) -> PortCheck:
        """轮询等待端口关闭。"""
        return wait_closed(target, port, timeout=timeout, interval=interval)

    @staticmethod
    def scan(
        host: str,
        ports: PortRange | str | Iterable[int],
        *,
        timeout: float = 0.3,
        workers: int = 64,
    ) -> PortScan:
        """并发扫描一批端口。"""
        return scan(host, ports, timeout=timeout, workers=workers)

    # ---------------------------------------------------------- HTTP
    @staticmethod
    def request(
        method: str,
        url: str,
        *,
        data: bytes | None = None,
        json_body: Any = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> HttpResponse:
        """通用 HTTP 请求。"""
        return http_request(
            method,
            url,
            data=data,
            json_body=json_body,
            headers=headers,
            timeout=timeout,
        )

    @staticmethod
    def get(
        url: str, *, timeout: float = DEFAULT_TIMEOUT, **options: Any
    ) -> HttpResponse:
        """发起 GET 请求。"""
        return get(url, timeout=timeout, **options)

    @staticmethod
    def post_json(
        url: str,
        payload: Any,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        **options: Any,
    ) -> HttpResponse:
        """以 JSON 形式发起 POST 请求。"""
        return post_json(url, payload, timeout=timeout, **options)

    @staticmethod
    def download(
        url: str,
        target: StrPath,
        *,
        overwrite: bool = False,
        timeout: float = 60.0,
    ) -> Path:
        """下载到本地文件。"""
        return download(url, target, overwrite=overwrite, timeout=timeout)

    # ---------------------------------------------------------- 静态服务
    @staticmethod
    def serve_static(
        directory: StrPath,
        port: int = 0,
        *,
        host: str = "127.0.0.1",
        block: bool = False,
    ) -> StaticServer:
        """把目录以 HTTP 共享出去（默认后台线程，不阻塞）。"""
        return serve_static(directory, port, host=host, block=block)
