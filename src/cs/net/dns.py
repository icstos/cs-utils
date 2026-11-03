"""域名解析。

薄封装 ``socket.getaddrinfo``：一次拿全 A/AAAA/CNAME，去重后保留解析器顺序，
失败时抛 :class:`~cs.net.errors.DnsError` 而不是裸 ``gaierror``。

>>> resolve("localhost").ipv4
('127.0.0.1',)
"""

from __future__ import annotations

import socket
import time

from .errors import AddressError, DnsError
from .models import DnsRecord

__all__ = ["resolve", "reverse"]


def resolve(
    host: str,
    *,
    port: int | None = None,
    family: int = socket.AF_UNSPEC,
    timeout: float | None = None,
) -> DnsRecord:
    """解析域名 / IP，返回去重后的地址列表。

    ``family`` 可取 ``AF_UNSPEC``（默认，A + AAAA）、``AF_INET`` 或 ``AF_INET6``。
    ``timeout`` 会临时改全局默认超时，多线程环境下建议保持 ``None``。
    """
    name = host.strip()
    if not name:
        raise AddressError("主机名不能为空")

    previous = socket.getdefaulttimeout()
    started = time.perf_counter()
    try:
        if timeout is not None:
            socket.setdefaulttimeout(timeout)
        infos = socket.getaddrinfo(name, port, family, socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise DnsError(f"无法解析 {name}：{exc.strerror or exc}") from exc
    finally:
        if timeout is not None:
            socket.setdefaulttimeout(previous)

    addresses = tuple(dict.fromkeys(info[4][0] for info in infos))
    canonical = ""
    if infos:
        try:
            canonical = socket.getfqdn(name)
        except OSError:
            canonical = ""
    return DnsRecord(
        host=name,
        addresses=addresses,
        canonical="" if canonical == name else canonical,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


def reverse(ip: str, *, timeout: float | None = None) -> str:
    """反向解析 IP 到主机名；没有 PTR 记录时抛 :class:`DnsError`。"""
    previous = socket.getdefaulttimeout()
    try:
        if timeout is not None:
            socket.setdefaulttimeout(timeout)
        return socket.gethostbyaddr(ip)[0]
    except (socket.herror, socket.gaierror) as exc:
        raise DnsError(f"{ip} 没有反向解析记录：{exc}") from exc
    finally:
        if timeout is not None:
            socket.setdefaulttimeout(previous)
