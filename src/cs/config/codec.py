"""TOML 编解码：读用标准库 ``tomllib``，写用本模块自带的小型写出器。

为什么不依赖 ``tomli_w``：本包只想多引一个纯标准库的读 + 一个可控的写，
避免为「写一个配置文件」引入第三方依赖。写出器覆盖 TOML 1.0 的全部值类型，
并保证 ``loads(dumps(data)) == data``（往返一致由测试兜底）。

写出规则：
* 标量表在前，子表在后（TOML 要求一个表内先写 key-value 再写子表头）；
* 字典 → ``[a.b]``，字典数组 → ``[[a.b]]``，空字典写成只有表头的 ``[a]``；
* ``None`` 直接丢弃（TOML 没有 null）；
* 键名不合法（含 ``.``/空格等）时自动加引号。
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import IO, Any

from .errors import (
    ConfigFileNotFoundError,
    ConfigLoadError,
    ConfigParseError,
    ConfigPathError,
    ConfigWriteError,
)

__all__ = [
    "dumps",
    "format_value",
    "load_file",
    "loads",
    "read_file",
    "write_file",
]

_BARE_KEY = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
)
_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


# ---------------------------------------------------------------- 读取
def loads(text: str, *, source: str = "<string>") -> dict[str, Any]:
    """解析 TOML 文本（语法错误统一转成 :class:`ConfigParseError`）。"""
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigParseError(source, str(exc)) from exc


def read_file(path: str | os.PathLike[str], *, encoding: str = "utf-8") -> str:
    """读文本文件（带 BOM 也能处理）。"""
    target = Path(path)
    try:
        return target.read_text(encoding=encoding, errors="strict").lstrip("\ufeff")
    except FileNotFoundError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigLoadError(f"读取 {target} 失败：{exc}") from exc


def load_file(
    path: str | os.PathLike[str], *, encoding: str = "utf-8"
) -> dict[str, Any]:
    """读并解析 TOML 文件。"""
    target = Path(path)
    if not target.is_file():
        raise ConfigFileNotFoundError(f"配置文件不存在：{target}")
    return loads(read_file(target, encoding=encoding), source=str(target))


# ---------------------------------------------------------------- 写出
def _quote_key(key: str) -> str:
    if key and all(char in _BARE_KEY for char in key):
        return key
    return _quote_string(key)


def _quote_string(text: str) -> str:
    """优先用 literal string（不处理转义），否则退回 basic string。"""
    if '"' in text and "'" not in text and all(char.isprintable() for char in text):
        return f"'{text}'"
    escaped = "".join(
        _ESCAPES.get(char, char if char.isprintable() else f"\\u{ord(char):04X}")
        for char in text
    )
    return f'"{escaped}"'


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    rendered = repr(value)
    return rendered if any(char in rendered for char in ".eE") else f"{rendered}.0"


def format_value(value: Any) -> str:
    match value:
        case bool():  # 必须在 int 之前，bool 是 int 的子类
            return "true" if value else "false"
        case str():
            return _quote_string(value)
        case int():
            return str(value)
        case float():
            return _format_float(value)
        case Decimal():
            return _format_float(float(value))
        case datetime():
            return value.isoformat(sep="T")
        case date():
            return value.isoformat()
        case time():
            return value.isoformat()
        case Path():
            return _quote_string(str(value))
        case Mapping():
            return _format_inline_table(value)
        case Sequence():
            items = [format_value(item) for item in value if item is not None]
            return "[" + ", ".join(items) + "]"
        case _:
            return _quote_string(str(value))


def _format_inline_table(data: Mapping[str, Any]) -> str:
    parts = [
        f"{_quote_key(str(key))} = {format_value(item)}"
        for key, item in data.items()
        if item is not None
    ]
    return "{" + ", ".join(parts) + "}"


def _is_table_array(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) > 0
        and all(isinstance(item, Mapping) for item in value)
    )


def _emit_table(lines: list[str], data: Mapping[str, Any], prefix: str) -> None:
    scalars: list[tuple[str, Any]] = []
    tables: list[tuple[str, Mapping[str, Any]]] = []
    arrays: list[tuple[str, Sequence[Mapping[str, Any]]]] = []

    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            tables.append((str(key), value))
        elif _is_table_array(value):
            arrays.append((str(key), value))
        else:
            scalars.append((str(key), value))

    for key, value in scalars:
        lines.append(f"{_quote_key(key)} = {format_value(value)}")

    for key, table in tables:
        path = f"{prefix}.{_quote_key(key)}" if prefix else _quote_key(key)
        lines.append("")
        lines.append(f"[{path}]")
        _emit_table(lines, table, path)

    for key, items in arrays:
        path = f"{prefix}.{_quote_key(key)}" if prefix else _quote_key(key)
        for item in items:
            lines.append("")
            lines.append(f"[[{path}]]")
            _emit_table(lines, item, path)


def dumps(data: Mapping[str, Any], *, header: str = "") -> str:
    """把字典渲染成 TOML 文本（``header`` 会作为文件开头的注释/说明原样保留）。"""
    if not isinstance(data, Mapping):
        raise ConfigPathError(f"TOML 顶层必须是字典，收到 {type(data).__name__}")
    lines: list[str] = []
    if header:
        lines.extend(f"# {line}".rstrip() for line in header.splitlines())
        lines.append("")
    _emit_table(lines, data, "")
    text = "\n".join(lines).strip("\n")
    return f"{text}\n" if text else ""


def write_file(
    path: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
    atomic: bool = True,
    backup: bool = False,
    mkdirs: bool = True,
) -> Path:
    """写文件：默认「同目录临时文件 + ``os.replace``」，避免写到一半断电留下半截配置。"""
    target = Path(path)
    if mkdirs:
        target.parent.mkdir(parents=True, exist_ok=True)

    if backup and target.exists():
        try:
            shutil.copy2(target, target.with_name(target.name + ".bak"))
        except OSError as exc:  # 备份失败不该阻止主流程，但要让调用方知道
            raise ConfigWriteError(f"备份 {target} 失败：{exc}") from exc

    if not atomic:
        try:
            target.write_text(text, encoding=encoding)
        except OSError as exc:
            raise ConfigWriteError(f"写入 {target} 失败：{exc}") from exc
        return target

    handle: IO[str] | None = None
    temp_path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
        )
        temp_path = Path(raw_path)
        handle = os.fdopen(descriptor, "w", encoding=encoding, newline="\n")
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.replace(temp_path, target)
        temp_path = None
        return target
    except OSError as exc:
        raise ConfigWriteError(f"写入 {target} 失败：{exc}") from exc
    finally:
        if handle is not None:
            handle.close()
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
