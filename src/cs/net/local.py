"""本机地址、网卡与 Wi-Fi 名称。

只用标准库 + ``psutil``，不引入新依赖。探测类函数遵循「永不抛异常、给合理默认值」
的原则：网络状态本来就多变，调用方不该为了拿一个 IP 而处理一堆 OSError。

>>> primary_ip()  # doctest: +SKIP
'192.168.1.23'
"""

from __future__ import annotations

import platform
import socket
import subprocess

import psutil

from .errors import NetError
from .models import HostInfo, Interface

__all__ = [
    "fqdn",
    "host_info",
    "hostname",
    "interfaces",
    "local_addresses",
    "primary_ip",
    "wifi_name",
]

#: 只用于让内核挑出默认路由的 UDP 探测目标，不会真的发包
_PROBE_TARGET = ("1.1.1.1", 53)
_CMD_TIMEOUT = 5.0


# ============================================================
# 主机名与地址
# ============================================================


def hostname() -> str:
    """本机主机名。"""
    return socket.gethostname()


def fqdn() -> str:
    """完全限定域名；解析不到时退化成 :func:`hostname`。"""
    try:
        return socket.getfqdn()
    except OSError:
        return hostname()


def primary_ip(
    *, target: tuple[str, int] = _PROBE_TARGET, fallback: str = "127.0.0.1"
) -> str:
    """本机用于出网的源 IP。

    做法是建一个 UDP socket 并 ``connect`` 到外部地址：内核只会查路由表选源地址，
    不会发出任何数据包，因此断网时也能拿到局域网 IP；完全没有路由时返回 ``fallback``。
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(target)
            return sock.getsockname()[0]
    except OSError:
        return fallback


# ============================================================
# 网卡
# ============================================================


def interfaces(*, up_only: bool = False) -> list[Interface]:
    """枚举网卡，默认按「启用优先 + 名称」排序。

    ``psutil`` 在部分平台会抛 ``RuntimeError``（例如网卡热插拔竞态），
    统一包装成 :class:`~cs.net.errors.NetError`。
    """
    try:
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()
    except (OSError, RuntimeError) as exc:
        raise NetError(f"读取网卡信息失败：{exc}") from exc

    link_family = getattr(psutil, "AF_LINK", None)
    result: list[Interface] = []
    for name, items in addrs.items():
        ipv4: list[str] = []
        ipv6: list[str] = []
        mac = ""
        for item in items:
            match item.family:
                case socket.AF_INET:
                    ipv4.append(item.address)
                case socket.AF_INET6:
                    # fe80::1%eth0 带 scope id，保留下来对 socket 无意义
                    ipv6.append(item.address.split("%", 1)[0])
                case _ if link_family is not None and item.family == link_family:
                    mac = item.address
        stat = stats.get(name)
        result.append(
            Interface(
                name=name,
                ipv4=tuple(ipv4),
                ipv6=tuple(ipv6),
                mac=mac,
                is_up=bool(stat.isup) if stat else False,
                speed_mbps=int(stat.speed) if stat else 0,
                mtu=int(stat.mtu) if stat else 0,
            )
        )

    if up_only:
        result = [item for item in result if item.is_up]
    return sorted(result, key=lambda item: (not item.is_up, item.name.casefold()))


def local_addresses(
    *, include_loopback: bool = False
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(ipv4, ipv6)`` 去重元组，来自所有已启用网卡。"""
    ipv4: list[str] = []
    ipv6: list[str] = []
    for nic in interfaces(up_only=True):
        ipv4.extend(nic.ipv4)
        ipv6.extend(nic.ipv6)

    if not include_loopback:
        ipv4 = [ip for ip in ipv4 if not ip.startswith("127.")]
        ipv6 = [
            ip
            for ip in ipv6
            if ip not in {"::1", "fe80"} and not ip.startswith("fe80:")
        ]
    return tuple(dict.fromkeys(ipv4)), tuple(dict.fromkeys(ipv6))


def host_info() -> HostInfo:
    """本机网络概要：主机名 + 局域网地址 + 出网源 IP。"""
    ipv4, ipv6 = local_addresses()
    return HostInfo(
        hostname=hostname(),
        fqdn=fqdn(),
        primary_ip=primary_ip(),
        ipv4=ipv4,
        ipv6=ipv6,
    )


# ============================================================
# 无线
# ============================================================


def _run(command: list[str]) -> str:
    """执行系统命令并返回 stdout；任何失败都返回空串。

    用参数列表而非 ``shell=True``，SSID 里出现空格/分号也不会被当成命令分隔符。
    """
    try:
        done = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_CMD_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout if done.returncode == 0 else ""


def _windows_ssid() -> str | None:
    # 形如 "    SSID                   : MyWiFi"，注意别把 BSSID 当 SSID
    for line in _run(["netsh", "WLAN", "show", "interfaces"]).splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip().casefold() == "ssid" and value.strip():
            return value.strip()
    return None


def _macos_ssid() -> str | None:
    # "Current Wi-Fi Network: MyWiFi"；未连接时是 "You are not associated with an AirPort network."
    _, _, value = _run(["networksetup", "-getairportnetwork", "en0"]).partition(":")
    name = value.strip()
    return name if name and "not associated" not in name else None


def _linux_ssid() -> str | None:
    name = _run(["iwgetid", "-r"]).strip()
    if name:
        return name
    # NetworkManager 兜底："yes:MyWiFi"
    for line in _run(["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"]).splitlines():
        active, _, ssid = line.partition(":")
        if active == "yes" and ssid:
            return ssid
    return None


def wifi_name() -> str | None:
    """当前连接的 Wi-Fi 名称（SSID）；未连接或非 Wi-Fi 环境返回 ``None``。"""
    match platform.system():
        case "Windows":
            return _windows_ssid()
        case "Darwin":
            return _macos_ssid()
        case _:
            return _linux_ssid()
