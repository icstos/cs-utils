"""端口探测与分配。

区分两个容易混淆的概念：

- **可用（free）**：本机现在能 ``bind`` 上去，代表可以起服务；
- **已开放（open）**：TCP 握手成功，代表对端有服务在监听。

二者不等价：端口被占用但没人监听时 ``is_free`` 为 ``False``、``is_open`` 也为
``False``（例如处于 TIME_WAIT 或对端只监听 UDP）。

>>> is_free(free_port())  # doctest: +SKIP
True
"""

from __future__ import annotations

import socket
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from .errors import AddressError, PortError
from .models import MAX_PORT, Address, PortCheck, PortRange, PortScan, PortState

__all__ = [
    "check",
    "free_port",
    "is_free",
    "is_open",
    "scan",
    "wait_closed",
    "wait_open",
]

DEFAULT_TIMEOUT = 0.5
DEFAULT_INTERVAL = 0.2
MAX_SCAN = 4096


# ============================================================
# 单个端口
# ============================================================


def _as_address(target: Address | str, port: int | None) -> Address:
    """统一 ``host`` / ``host:port`` / :class:`Address` 三种入参。

    显式给了 ``port`` 就以它为准（``check("10.0.0.1:80", 443)`` 探的是 443），
    避免字符串里的端口和参数打架时出现难以察觉的歧义。
    """
    if isinstance(target, Address):
        return target if port is None else Address(target.host, port)
    address = Address.parse(target)
    return Address(address.host, port) if port is not None else address


def check(
    target: Address | str,
    port: int | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> PortCheck:
    """探测一个端口，永不抛「连不上」类异常，结果体现在 ``state`` 上。

    主机名解析失败会抛 :class:`~cs.net.errors.AddressError`——这属于调用方笔误，
    不该混进「端口关闭」里。
    """
    address = _as_address(target, port)
    started = time.perf_counter()
    try:
        with socket.create_connection(address.sockaddr, timeout=timeout):
            state, detail = PortState.OPEN, ""
    except socket.gaierror as exc:
        raise AddressError(
            f"无法解析主机 {address.host}：{exc.strerror or exc}"
        ) from exc
    except ConnectionRefusedError:
        state, detail = PortState.CLOSED, "连接被拒绝"
    except TimeoutError:
        state, detail = PortState.FILTERED, f"{timeout:g}s 内无响应"
    except OSError as exc:
        state, detail = PortState.ERROR, str(exc)
    return PortCheck(
        address=address,
        state=state,
        latency_ms=(time.perf_counter() - started) * 1000,
        detail=detail,
    )


def is_open(
    target: Address | str,
    port: int | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> bool:
    """端口是否已开放（TCP 握手成功）。"""
    return check(target, port, timeout=timeout).is_open


def is_free(port: int, *, host: str = "") -> bool:
    """本机能否 ``bind`` 该端口。

    ``host`` 默认 ``""``（所有接口），这也是服务端最常见的绑定方式；
    只关心某个具体地址时可传 ``"127.0.0.1"``。
    注意 Windows 上不设 ``SO_REUSEADDR``——它会允许两个进程绑同一端口，
    反而掩盖了占用问题。
    """
    if not 0 <= port <= MAX_PORT:
        raise AddressError(f"端口必须在 0~{MAX_PORT} 之间，收到 {port}")
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
    except OSError:
        # 被占用、权限不足（<1024）、地址不存在，都属于「起不了服务」
        return False
    return True


def free_port(*, host: str = "", candidates: PortRange | None = None) -> int:
    """挑一个空闲端口。

    给 ``candidates`` 时按序试（配置里写死端口的场景）；否则让内核分配（``bind(0)``），
    后者没有竞态之外的额外开销。
    """
    if candidates is not None:
        for port in candidates.ports():
            if is_free(port, host=host):
                return port
        raise PortError(f"{candidates} 范围内没有空闲端口")

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


# ============================================================
# 等待
# ============================================================


def wait_open(
    target: Address | str,
    port: int | None = None,
    *,
    timeout: float = 10.0,
    interval: float = DEFAULT_INTERVAL,
) -> PortCheck:
    """轮询直到端口开放或超时，返回最后一次探测结果（不抛超时异常）。

    典型用法是等服务启动完：

    >>> wait_open("127.0.0.1", 8000, timeout=30).is_open  # doctest: +SKIP
    True
    """
    address = _as_address(target, port)
    deadline = time.monotonic() + max(timeout, 0.0)
    result = check(address, timeout=min(max(interval, 0.05), DEFAULT_TIMEOUT))
    while not result.is_open and time.monotonic() < deadline:
        time.sleep(max(min(interval, deadline - time.monotonic()), 0.0))
        result = check(address, timeout=min(max(interval, 0.05), DEFAULT_TIMEOUT))
    return result


def wait_closed(
    target: Address | str,
    port: int | None = None,
    *,
    timeout: float = 10.0,
    interval: float = DEFAULT_INTERVAL,
) -> PortCheck:
    """轮询直到端口不再开放（用于等旧服务释放端口）。"""
    address = _as_address(target, port)
    deadline = time.monotonic() + max(timeout, 0.0)
    result = check(address, timeout=min(max(interval, 0.05), DEFAULT_TIMEOUT))
    while result.is_open and time.monotonic() < deadline:
        time.sleep(max(min(interval, deadline - time.monotonic()), 0.0))
        result = check(address, timeout=min(max(interval, 0.05), DEFAULT_TIMEOUT))
    return result


# ============================================================
# 批量
# ============================================================


def _validate(targets: tuple[int, ...], limit: int) -> tuple[int, ...]:
    if len(targets) > limit:
        raise AddressError(f"一次最多扫描 {limit} 个端口，收到 {len(targets)}")
    for value in targets:
        if not 0 <= value <= MAX_PORT:
            raise AddressError(f"端口越界：{value}")
    return targets


def _expand(
    ports: PortRange | str | Iterable[int], *, limit: int = MAX_SCAN
) -> tuple[int, ...]:
    """把 ``PortRange`` / ``"80,443,8000-8100"`` / ``[80, 443]`` 统一成端口元组。"""
    match ports:
        case PortRange():
            return ports.ports(limit=limit)
        case str():
            values: list[int] = []
            for part in ports.replace(" ", ",").split(","):
                if part:
                    values.extend(PortRange.parse(part))
            return _validate(tuple(dict.fromkeys(values)), limit)
        case _:
            return _validate(tuple(dict.fromkeys(int(item) for item in ports)), limit)


def scan(
    host: str,
    ports: PortRange | str | Iterable[int],
    *,
    timeout: float = 0.3,
    workers: int = 64,
) -> PortScan:
    """并发探测一批端口，结果按端口升序。

    ``workers`` 默认 64：再高对局域网没收益，对公网反而容易被限速。
    """
    targets = _expand(ports)
    started = time.perf_counter()
    if not targets:
        return PortScan(host=host, checks=(), elapsed_ms=0.0)

    def probe(port: int) -> PortCheck:
        return check(host, port, timeout=timeout)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(targets)))) as pool:
        checks = tuple(pool.map(probe, targets))
    return PortScan(
        host=host,
        checks=checks,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )
