"""HTTP 客户端：基于标准库 ``urllib``，不新增依赖。

设计取舍：

- **4xx / 5xx 不抛异常**，作为带状态码的 :class:`~cs.net.models.HttpResponse` 返回，
  因为「接口返回 404」通常比「拿不到响应」更需要看响应体；
- 只有传输层失败（连不上、超时、TLS 错误）才抛 :class:`~cs.net.errors.HttpError`；
- 公网 IP 查询按提供方顺序重试，全部失败才抛错。

>>> get("https://example.com", timeout=5).status  # doctest: +SKIP
200
"""

from __future__ import annotations

import ipaddress
import json
import re
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .errors import HttpError
from .models import HttpResponse, PublicIP

__all__ = [
    "DEFAULT_TIMEOUT",
    "DEFAULT_UA",
    "PUBLIC_IP_PROVIDERS",
    "download",
    "get",
    "post_json",
    "public_ip",
    "request",
]

DEFAULT_UA = "cs-net/0.1 (+https://pypi.org/project/cs/)"
DEFAULT_TIMEOUT = 10.0
_CHUNK = 1 << 16
_IP_SPLIT = re.compile(r"[\s,]+")

#: 公网 IP 查询服务，按顺序尝试（都是纯文本或 JSON 的轻量接口）
PUBLIC_IP_PROVIDERS: tuple[str, ...] = (
    "https://api.ipify.org",
    "https://ifconfig.me/ip",
    "https://ipinfo.io/ip",
    "https://httpbin.org/ip",
    "https://api.ip.sb/ip",
)


# ============================================================
# 基础请求
# ============================================================


def request(
    method: str,
    url: str,
    *,
    data: bytes | None = None,
    json_body: Any = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> HttpResponse:
    """发一个 HTTP 请求，返回状态码 / 头 / 正文 / 耗时。

    ``json_body`` 不为 ``None`` 时自动序列化并补 ``Content-Type``；
    与 ``data`` 互斥，同时给出时以 ``json_body`` 为准。
    """
    body = data
    merged = {"User-Agent": DEFAULT_UA, **(headers or {})}
    if json_body is not None:
        body = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        merged.setdefault("Content-Type", "application/json; charset=utf-8")

    started = time.perf_counter()
    try:
        req = urllib.request.Request(
            url, data=body, headers=merged, method=method.upper()
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            status = resp.status
            raw_headers = dict(resp.headers.items())
            final = resp.geturl()
    except urllib.error.HTTPError as exc:
        # 4xx/5xx：响应体可能有错误详情，照常返回
        payload = exc.read()
        status = exc.code
        raw_headers = dict(exc.headers.items()) if exc.headers else {}
        final = exc.geturl() or url
    except (OSError, ValueError) as exc:
        # URLError / 超时 / TLS 失败，以及 "unknown url type" 这类非法 URL
        reason = getattr(exc, "reason", exc)
        raise HttpError(f"{method.upper()} {url} 请求失败：{reason}", url=url) from exc

    return HttpResponse(
        url=final,
        status=status,
        headers=raw_headers,
        body=payload,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


def get(url: str, *, timeout: float = DEFAULT_TIMEOUT, **options: Any) -> HttpResponse:
    """GET 请求（``headers`` 等参数透传）。"""
    return request("GET", url, timeout=timeout, **options)


def post_json(
    url: str,
    payload: Any,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    **options: Any,
) -> HttpResponse:
    """POST JSON 请求。"""
    return request("POST", url, json_body=payload, timeout=timeout, **options)


# ============================================================
# 下载
# ============================================================


def download(
    url: str,
    target: str | Path,
    *,
    overwrite: bool = False,
    timeout: float = 60.0,
) -> Path:
    """流式下载到本地文件，失败时删掉半成品。

    先写 ``.part`` 再改名，中途失败不会留下一个看似完整的文件。
    """
    path = Path(target).expanduser()
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} 已存在（如需覆盖请传 overwrite=True）")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_UA})
        with (
            urllib.request.urlopen(req, timeout=timeout) as resp,
            partial.open("wb") as fh,
        ):
            shutil.copyfileobj(resp, fh, length=_CHUNK)
        partial.replace(path)
    except (OSError, ValueError) as exc:
        partial.unlink(missing_ok=True)
        reason = getattr(exc, "reason", exc)
        raise HttpError(f"下载 {url} 失败：{reason}", url=url) from exc
    return path


# ============================================================
# 公网 IP
# ============================================================


def _extract_ip(text: str) -> str:
    """从纯文本或 JSON（``{"origin": "1.2.3.4"}``）里揪出第一个合法 IP。"""
    payload: Any = None
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        for key in ("origin", "ip", "query", "address"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                stripped = value.strip()
                break

    for token in _IP_SPLIT.split(stripped):
        candidate = token.strip().strip('"')
        try:
            return str(ipaddress.ip_address(candidate))
        except ValueError:
            continue
    return ""


def public_ip(
    *,
    timeout: float = 5.0,
    providers: Sequence[str] = PUBLIC_IP_PROVIDERS,
) -> PublicIP:
    """查询公网出口 IP，按 ``providers`` 顺序重试，全部失败抛 :class:`HttpError`。"""
    failures: list[str] = []
    for provider in providers:
        started = time.perf_counter()
        try:
            response = get(provider, timeout=timeout)
        except HttpError as exc:
            failures.append(f"{provider}: {exc}")
            continue
        ip = _extract_ip(response.text()) if response.ok else ""
        if ip:
            return PublicIP(
                ip=ip,
                provider=provider,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        failures.append(f"{provider}: HTTP {response.status}，未取到 IP")

    raise HttpError("无法获取公网 IP：" + "；".join(failures))
