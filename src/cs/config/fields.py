"""字段声明助手：给 ``dataclasses.field`` 挂上配置元数据。

用法与 ``dataclasses.field`` 完全一致，只是多了几个可选约束::

    @dataclass(slots=True)
    class Database:
        host: str = option("localhost", description="数据库主机")
        port: int = option(5432, ge=1, le=65535)
        password: str = option("", secret=True)

所有元数据都存放在 ``field.metadata["cs.config"]`` 下，因此不侵入 dataclass 本身，
不依赖 pydantic，也不影响 ``__eq__`` / ``__repr__``。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import MISSING, Field, dataclass, field
from typing import Any

__all__ = ["MISSING", "OPTION_KEY", "Option", "get_option", "option"]

#: ``field.metadata`` 里存放本模块元数据的键
OPTION_KEY = "cs.config"


@dataclass(frozen=True, slots=True)
class Option:
    """一个配置字段的附加说明与约束（不可变值对象）。"""

    # 注意：不能写 `default: Any = MISSING` —— dataclasses 会把 MISSING 当成
    # "该字段没有默认值"，从而把 default 变成必填参数。用工厂返回哨兵即可绕开。
    default: Any = field(default_factory=lambda: MISSING)
    default_factory: Callable[[], Any] | None = None
    description: str = ""
    ge: float | None = None
    gt: float | None = None
    le: float | None = None
    lt: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    choices: tuple[Any, ...] | None = None
    secret: bool = False
    alias: str | None = None
    env: str | None = None

    @property
    def has_default(self) -> bool:
        return self.default is not MISSING or self.default_factory is not None

    def make_default(self) -> Any:
        """取默认值；``default_factory`` 存在时优先用工厂（避免可变默认值共享）。"""
        if self.default_factory is not None:
            return self.default_factory()
        if self.default is not MISSING:
            return self.default
        raise LookupError("该字段没有默认值")

    def __repr__(self) -> str:
        bits = [
            f"{name}={getattr(self, name)!r}"
            for name in ("ge", "gt", "le", "lt", "min_length", "max_length", "pattern")
            if getattr(self, name) is not None
        ]
        if self.default is not MISSING:
            bits.insert(0, f"default={self.default!r}")
        elif self.default_factory is not None:
            bits.insert(0, f"default_factory={self.default_factory!r}")
        if self.choices:
            bits.append(f"choices={self.choices!r}")
        if self.secret:
            bits.append("secret=True")
        if self.alias:
            bits.append(f"alias={self.alias!r}")
        if self.description:
            bits.append(f"description={self.description!r}")
        return f"Option({', '.join(bits)})"


def option(
    default: Any = MISSING,
    *,
    default_factory: Callable[[], Any] | None = None,
    description: str = "",
    ge: float | None = None,
    gt: float | None = None,
    le: float | None = None,
    lt: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    pattern: str | None = None,
    choices: tuple[Any, ...] | None = None,
    secret: bool = False,
    alias: str | None = None,
    env: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """声明一个带约束的配置字段（返回值直接赋给字段即可）。"""
    if default is not MISSING and default_factory is not None:
        raise TypeError("default 与 default_factory 只能给一个")
    meta = dict(metadata or {})
    meta[OPTION_KEY] = Option(
        default=default,
        default_factory=default_factory,
        description=description,
        ge=ge,
        gt=gt,
        le=le,
        lt=lt,
        min_length=min_length,
        max_length=max_length,
        pattern=pattern,
        choices=choices,
        secret=secret,
        alias=alias,
        env=env,
    )
    kwargs: dict[str, Any] = {"metadata": meta}
    if default is not MISSING:
        kwargs["default"] = default
    if default_factory is not None:
        kwargs["default_factory"] = default_factory
    return field(**kwargs)


def get_option(item: Field[Any]) -> Option | None:
    """从 ``Field`` 上取回 :class:`Option`（没有则返回 ``None``）。"""
    return item.metadata.get(OPTION_KEY)  # type: ignore[return-value]
