"""``cs.net`` 的数据模型。

全部是 ``frozen`` + ``slots`` 的 dataclass：创建后不可变、字段紧凑，
``to_dict()`` 可直接交给 MCP 工具或日志。枚举用 ``StrEnum``，序列化即字符串。

>>> Address.parse("127.0.0.1:8000").port
8000
>>> PortRange.parse("8000-8002").ports()
(8000, 8001, 8002)
"""

from __future__ import annotations

import ipaddress
import json
import socket
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .errors import AddressError, HttpError

__all__ = [
    "Address",
    "DnsRecord",
    "HostInfo",
    "HttpResponse",
    "Interface",
    "PortCheck",
    "PortRange",
    "PortScan",
    "PortState",
    "PublicIP",
]

MAX_PORT = 65535


# ============================================================
# 地址
# ============================================================


@dataclass(frozen=True, slots=True)
class Address:
    """主机 + 端口。IPv6 会带上方括号，可直接拼进 URL 或日志。

    >>> str(Address("::1", 80))
    '[::1]:80'
    """

    host: str
    port: int

    def __post_init__(self) -> None:
        host = self.host.strip()
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]
        if not host:
            raise AddressError("host 不能为空")
        try:
            port = int(self.port)
        except (TypeError, ValueError) as exc:
            raise AddressError(f"端口必须是整数，收到 {self.port!r}") from exc
        if not 0 <= port <= MAX_PORT:
            raise AddressError(f"端口必须在 0~{MAX_PORT} 之间，收到 {port}")
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "port", port)

    # ---------------------------------------------------------- 构造
    @classmethod
    def parse(cls, text: str, *, default_port: int = 0) -> Address:
        """解析 ``host:port`` / ``host`` / ``[::1]:8000``。

        端口缺省时用 ``default_port``；裸 IPv6（含多个冒号且无方括号）整体视为 host。
        """
        raw = text.strip()
        if not raw:
            raise AddressError("地址不能为空")

        if raw.startswith("["):
            end = raw.find("]")
            if end < 0:
                raise AddressError(f"IPv6 地址缺少右方括号：{text}")
            host, rest = raw[1:end], raw[end + 1 :]
            if not rest:
                return cls(host, default_port)
            if not rest.startswith(":"):
                raise AddressError(f"无法解析地址：{text}")
            try:
                return cls(host, int(rest[1:]))
            except ValueError as exc:
                raise AddressError(f"端口必须是整数：{text}") from exc

        if raw.count(":") > 1:
            return cls(raw, default_port)

        host, sep, tail = raw.rpartition(":")
        if sep and tail.isdigit():
            return cls(host, int(tail))
        return cls(raw, default_port)

    # ---------------------------------------------------------- 属性
    @property
    def family(self) -> socket.AddressFamily:
        """``AF_INET6``（含冒号）或 ``AF_INET``。"""
        return socket.AF_INET6 if ":" in self.host else socket.AF_INET

    @property
    def is_loopback(self) -> bool:
        """是否本机回环（``localhost`` / ``127.0.0.0/8`` / ``::1``）。"""
        if self.host.casefold().split(".")[0] == "localhost":
            return True
        try:
            return ipaddress.ip_address(self.host).is_loopback
        except ValueError:
            return False

    @property
    def sockaddr(self) -> tuple[str, int]:
        """可直接传给 ``socket`` 的二元组。"""
        return (self.host, self.port)

    def with_port(self, port: int) -> Address:
        """换端口，返回新对象。"""
        return Address(self.host, port)

    def __str__(self) -> str:
        return (
            f"[{self.host}]:{self.port}"
            if ":" in self.host
            else f"{self.host}:{self.port}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {"host": self.host, "port": self.port, "address": str(self)}


# ============================================================
# 端口
# ============================================================


class PortState(StrEnum):
    """端口探测结果。

    - ``OPEN``：TCP 握手成功；
    - ``CLOSED``：对端明确拒绝（RST）；
    - ``FILTERED``：超时无响应，通常被防火墙丢弃；
    - ``ERROR``：其他错误（权限、网络不可达等）。
    """

    OPEN = "open"
    CLOSED = "closed"
    FILTERED = "filtered"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class PortCheck:
    """一次端口探测的结果与耗时。"""

    address: Address
    state: PortState
    latency_ms: float = 0.0
    detail: str = ""

    @property
    def host(self) -> str:
        return self.address.host

    @property
    def port(self) -> int:
        return self.address.port

    @property
    def is_open(self) -> bool:
        return self.state is PortState.OPEN

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "state": self.state.value,
            "open": self.is_open,
            "latency_ms": round(self.latency_ms, 2),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class PortRange:
    """闭区间端口范围，``1 <= start <= end <= 65535``。

    >>> list(PortRange(8000, 8002))
    [8000, 8001, 8002]
    """

    start: int = 1
    end: int = MAX_PORT

    def __post_init__(self) -> None:
        if not 1 <= self.start <= MAX_PORT:
            raise AddressError(f"起始端口必须在 1~{MAX_PORT} 之间，收到 {self.start}")
        if not 1 <= self.end <= MAX_PORT:
            raise AddressError(f"结束端口必须在 1~{MAX_PORT} 之间，收到 {self.end}")
        if self.start > self.end:
            raise AddressError(f"起始端口不能大于结束端口：{self.start} > {self.end}")

    @classmethod
    def parse(cls, text: str) -> PortRange:
        """解析 ``8000`` / ``8000-8010`` / ``8000:8010``。

        离散端口（``"80,443"``）无法用单个闭区间表示，请直接传给
        :func:`cs.net.scan`——它会逐个展开。
        """
        raw = text.strip().replace(":", "-")
        if not raw:
            raise AddressError("端口范围不能为空")
        if "," in raw or " " in raw:
            raise AddressError(f"PortRange 不能表示离散端口：{text}（请改用 scan）")
        head, sep, tail = raw.partition("-")
        try:
            start = int(head)
            end = int(tail) if sep else start
        except ValueError as exc:
            raise AddressError(f"无法解析端口范围：{text}") from exc
        return cls(start, end)

    def __contains__(self, port: object) -> bool:
        return isinstance(port, int) and self.start <= port <= self.end

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start, self.end + 1))

    def __len__(self) -> int:
        return self.end - self.start + 1

    def ports(self, *, limit: int = 1024) -> tuple[int, ...]:
        """展开成端口元组；超过 ``limit`` 时抛错，避免误扫 6 万个端口。"""
        if len(self) > limit:
            raise AddressError(f"{self} 含 {len(self)} 个端口，超过上限 {limit}")
        return tuple(self)

    def __str__(self) -> str:
        return f"{self.start}" if self.start == self.end else f"{self.start}-{self.end}"

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "count": len(self)}


@dataclass(frozen=True, slots=True)
class PortScan:
    """批量端口扫描结果（``checks`` 按端口升序）。"""

    host: str
    checks: tuple[PortCheck, ...] = ()
    elapsed_ms: float = 0.0

    @property
    def scanned(self) -> int:
        return len(self.checks)

    @property
    def open_ports(self) -> tuple[int, ...]:
        return tuple(item.port for item in self.checks if item.is_open)

    def to_dict(self, *, only_open: bool = False, limit: int = 512) -> dict[str, Any]:
        items = [item for item in self.checks if item.is_open or not only_open]
        return {
            "host": self.host,
            "scanned": self.scanned,
            "open": list(self.open_ports),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "truncated": len(items) > limit,
            "ports": [item.to_dict() for item in items[:limit]],
        }


# ============================================================
# 本机信息
# ============================================================


@dataclass(frozen=True, slots=True)
class Interface:
    """一块网卡及其地址。"""

    name: str
    ipv4: tuple[str, ...] = ()
    ipv6: tuple[str, ...] = ()
    mac: str = ""
    is_up: bool = False
    speed_mbps: int = 0
    mtu: int = 0

    @property
    def addresses(self) -> tuple[str, ...]:
        return (*self.ipv4, *self.ipv6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ipv4": list(self.ipv4),
            "ipv6": list(self.ipv6),
            "mac": self.mac,
            "up": self.is_up,
            "speed_mbps": self.speed_mbps,
            "mtu": self.mtu,
        }


@dataclass(frozen=True, slots=True)
class HostInfo:
    """本机网络概要：主机名 + 非回环地址。"""

    hostname: str
    fqdn: str = ""
    primary_ip: str = "127.0.0.1"
    ipv4: tuple[str, ...] = ()
    ipv6: tuple[str, ...] = ()

    @property
    def ips(self) -> tuple[str, ...]:
        return (*self.ipv4, *self.ipv6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hostname": self.hostname,
            "fqdn": self.fqdn,
            "primary_ip": self.primary_ip,
            "ipv4": list(self.ipv4),
            "ipv6": list(self.ipv6),
        }


@dataclass(frozen=True, slots=True)
class DnsRecord:
    """域名解析结果（``addresses`` 保留解析器给出的顺序）。"""

    host: str
    addresses: tuple[str, ...] = ()
    canonical: str = ""
    elapsed_ms: float = 0.0

    @property
    def ipv4(self) -> tuple[str, ...]:
        return tuple(item for item in self.addresses if ":" not in item)

    @property
    def ipv6(self) -> tuple[str, ...]:
        return tuple(item for item in self.addresses if ":" in item)

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "canonical": self.canonical,
            "addresses": list(self.addresses),
            "ipv4": list(self.ipv4),
            "ipv6": list(self.ipv6),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


# ============================================================
# HTTP
# ============================================================


@dataclass(frozen=True, slots=True)
class PublicIP:
    """公网出口 IP 与提供它的服务。"""

    ip: str
    provider: str = ""
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ip": self.ip,
            "provider": self.provider,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """一次 HTTP 响应。4xx / 5xx 也走这里，用 :attr:`ok` 判断成败。

    ``body`` 保留原始字节；``to_dict()`` 默认只带截断后的文本，避免把
    几十 MB 的响应塞进 MCP 消息里。
    """

    url: str
    status: int
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def size(self) -> int:
        return len(self.body)

    @property
    def content_type(self) -> str:
        """``Content-Type``；头字段名大小写不敏感（不同服务端写法不一）。"""
        for key, value in self.headers.items():
            if key.casefold() == "content-type":
                return value
        return ""

    def text(self, encoding: str = "utf-8") -> str:
        """按 ``encoding`` 解码；解码失败用替换字符，不抛异常。"""
        return self.body.decode(encoding, errors="replace")

    def json(self) -> Any:
        """解析 JSON 正文，失败抛 :class:`~cs.net.errors.HttpError`。"""
        try:
            return json.loads(self.body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HttpError(
                f"{self.url} 的响应不是合法 JSON：{exc}", url=self.url
            ) from exc

    def to_dict(self, *, full: bool = False, limit: int = 2000) -> dict[str, Any]:
        body = self.text()
        return {
            "url": self.url,
            "status": self.status,
            "ok": self.ok,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "size": self.size,
            "content_type": self.content_type,
            "headers": dict(self.headers),
            "truncated": not full and len(body) > limit,
            "text": body if full else body[:limit],
        }
