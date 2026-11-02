"""环境变量的读取、脱敏与临时覆盖。

- :func:`read` 读取全部变量，默认把密码/密钥类字段打成 ``***``；
- :func:`put` / :func:`drop` 修改当前进程的环境（子进程可见）；
- :func:`patched` / :func:`scoped` 是上下文管理器，退出时精确还原被改动的键
  （只快照涉及的键，不复制整个 ``os.environ``，并发下不会互相踩踏）。
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

__all__ = [
    "SENSITIVE_HINTS",
    "drop",
    "get",
    "is_sensitive",
    "mask",
    "patched",
    "put",
    "read",
    "scoped",
]

#: 变量名里出现这些片段就视为敏感值
SENSITIVE_HINTS: tuple[str, ...] = (
    "PASS",
    "SECRET",
    "TOKEN",
    "KEY",
    "CREDENTIAL",
    "PRIVATE",
    "AUTH",
    "COOKIE",
    "SESSION",
)

MASK = "***"


def is_sensitive(name: str) -> bool:
    """变量名是否像密码 / 令牌 / 密钥。"""
    upper = name.upper()
    return any(hint in upper for hint in SENSITIVE_HINTS)


def mask(value: str) -> str:
    """把敏感值替换成 ``***``（空值保持为空）。"""
    return MASK if value else ""


def get(name: str, default: str | None = None) -> str | None:
    """读取单个变量。"""
    return os.environ.get(name, default)


def put(name: str, value: str) -> None:
    """设置变量（仅当前进程及其子进程可见）。"""
    os.environ[name] = str(value)


def drop(name: str) -> bool:
    """删除变量，返回是否真的存在过。"""
    return os.environ.pop(name, None) is not None


def read(
    prefix: str | None = None,
    *,
    reveal: bool = False,
    mapping: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """读取环境变量；``prefix`` 过滤前缀，``reveal`` 为假时脱敏敏感项。"""
    source = os.environ if mapping is None else mapping
    items = (
        (name, value)
        for name, value in source.items()
        if prefix is None or name.startswith(prefix)
    )
    return {
        name: value if reveal or not is_sensitive(name) else mask(value)
        for name, value in sorted(items)
    }


@contextmanager
def patched(values: Mapping[str, str | None]) -> Iterator[None]:
    """临时覆盖若干变量，退出时还原（``None`` 表示临时删除）。"""
    saved = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)
        yield
    finally:
        for name, previous in saved.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


@contextmanager
def scoped(name: str, value: str | None) -> Iterator[None]:
    """单变量的 :func:`patched` 简写；``value=None`` 时不做任何改动。"""
    if value is None:
        yield
        return
    with patched({name: value}):
        yield
