"""网络工具：本机地址、域名解析、公网 IP、端口探测、HTTP、静态文件服务。

三种用法并存，按场景挑一种即可：

**1. 门面（最省事）**::

    from cs.net import Net

    Net.primary_ip()                      # 本机出网 IP
    Net.check("127.0.0.1", 8000).state    # open / closed / filtered / error
    Net.free_port()                       # 让系统分配一个空闲端口
    Net.public_ip().ip                    # 公网出口 IP
    Net.get("https://example.com").status

**2. 领域对象（需要长期持有 / 精细控制）**::

    from cs.net import Address, PortRange, StaticServer

    address = Address.parse("192.168.1.10:8000")
    with StaticServer("output", port=0) as server:
        print(server.url)                 # 后台线程，退出 with 自动释放端口
    Net.scan(address.host, PortRange.parse("8000-8100"))

**3. MCP 服务**::

    python -m cs.net serve                       # stdio
    python -m cs.net serve --transport http --port 8000
    fastmcp run src/cs/net/mcp_server.py:mcp

模块划分：

======================  ====================================================
:mod:`cs.net.models`     ``Address`` / ``PortRange`` / ``PortCheck`` / ``HttpResponse`` 等
:mod:`cs.net.local`      主机名、网卡、Wi-Fi、出网 IP
:mod:`cs.net.dns`        域名解析与反查
:mod:`cs.net.ports`      端口探测、等待、批量扫描、空闲端口
:mod:`cs.net.http`       基于 urllib 的请求 / 下载 / 公网 IP
:mod:`cs.net.server`     后台线程静态文件服务
:mod:`cs.net.core`       :class:`Net` 门面
:mod:`cs.net.mcp_server` MCP 工具（``net_*``）
======================  ====================================================

顶层不导入 ``psutil`` 之外的重型依赖，``import cs.net`` 只做轻量绑定。
"""

from __future__ import annotations

from .core import Net
from .dns import resolve, reverse
from .errors import (
    AddressError,
    DnsError,
    HttpError,
    NetError,
    PortError,
    PortInUseError,
    ServerError,
)
from .http import (
    DEFAULT_TIMEOUT,
    PUBLIC_IP_PROVIDERS,
    download,
    get,
    post_json,
    public_ip,
)
from .local import (
    fqdn,
    host_info,
    hostname,
    interfaces,
    local_addresses,
    primary_ip,
    wifi_name,
)
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
from .ports import check, free_port, is_free, is_open, scan, wait_closed, wait_open
from .server import StaticServer, serve_static

__all__ = [
    "DEFAULT_TIMEOUT",
    "PUBLIC_IP_PROVIDERS",
    "Address",
    "AddressError",
    "DnsError",
    "DnsRecord",
    "HostInfo",
    "HttpError",
    "HttpResponse",
    "Interface",
    "Net",
    "NetError",
    "PortCheck",
    "PortError",
    "PortInUseError",
    "PortRange",
    "PortScan",
    "PortState",
    "PublicIP",
    "ServerError",
    "StaticServer",
    "check",
    "create_img_http_server",
    "download",
    "fqdn",
    "free_port",
    "get",
    "get_free_port",
    "get_host_ip",
    "get_public_network_ip",
    "get_wifi_name",
    "host_info",
    "hostname",
    "interfaces",
    "is_free",
    "is_open",
    "is_open_port",
    "local_addresses",
    "post_json",
    "primary_ip",
    "public_ip",
    "resolve",
    "reverse",
    "scan",
    "serve_static",
    "wait_closed",
    "wait_open",
    "wifi_name",
]

# ============================================================
# 旧接口兼容（cs.internet 的函数名与参数顺序）
# ============================================================
# 这些包装只做名字映射，语义与历史版本一致；新代码请直接用上面的 API。


def get_host_ip() -> str:
    """本机 IP（等价于 :func:`primary_ip`）。"""
    return primary_ip()


def get_free_port() -> int:
    """一个空闲端口（等价于 ``free_port()``）。"""
    return free_port()


def is_open_port(port: int, ip: str = "127.0.0.1") -> bool:
    """端口是否开放；注意参数顺序是 ``(port, ip)``。"""
    return is_open(ip, port)


def get_wifi_name() -> str | None:
    """当前 Wi-Fi 名称（等价于 :func:`wifi_name`）。"""
    return wifi_name()


def get_public_network_ip() -> str:
    """公网出口 IP 字符串。

    与旧版不同：查询失败会抛 :class:`~cs.net.errors.HttpError`，而不是静默返回
    ``127.0.0.1``（那个值会让上游误以为拿到的是真实公网地址）。
    """
    return public_ip().ip


def create_img_http_server(img_dir: str, port: int = 8000) -> None:
    """阻塞地把目录共享为 HTTP 服务（Ctrl-C 退出），用于浏览器预览图片。

    与旧实现不同：不再 ``os.chdir``、不再 ``os.system``，退出时端口一定释放。
    """
    serve_static(img_dir, port, block=True)
