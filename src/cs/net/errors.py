"""网络模块的异常层级。

与 :mod:`cs.cv` / :mod:`cs.env` 一致：领域异常统一继承 :class:`NetError`，
MCP 层再转换成 ``ToolError``，不把 traceback 丢给模型。

``AddressError`` 同时继承 :class:`ValueError`，因此「端口越界」这类参数问题
既能被 :class:`NetError` 捕获，也符合 Python 对非法取值的惯例。
"""

from __future__ import annotations

__all__ = [
    "AddressError",
    "DnsError",
    "HttpError",
    "NetError",
    "PortError",
    "PortInUseError",
    "ServerError",
]


class NetError(Exception):
    """本包所有错误的基类。"""


class AddressError(NetError, ValueError):
    """主机名、IP 或端口不合法。"""


class DnsError(AddressError):
    """域名解析失败或结果为空。"""


class PortError(NetError):
    """端口探测、等待或分配失败。"""


class PortInUseError(PortError):
    """端口已被占用（监听失败）。"""


class HttpError(NetError):
    """HTTP 在传输层失败：连不上、超时、TLS 错误等。

    4xx / 5xx 不算异常，会作为正常响应返回（见 :class:`~cs.net.models.HttpResponse`），
    只有拿不到响应时才抛出本异常。
    """

    def __init__(self, message: str, *, url: str = "") -> None:
        super().__init__(message)
        self.url = url


class ServerError(NetError):
    """本地服务启停失败。"""
