"""``TomlDocument`` —— 一份 TOML 文件的「文档」抽象（无 schema）。

它只关心「文件 ↔ 嵌套字典」的读写与路径寻址，不做类型校验；带类型校验的
是 :class:`~cs.config.manager.ConfigManager`，后者内部就是包着一个本类。

点号路径支持：

* ``app.debug`` 逐层下钻；
* ``a\\.b`` 表示键名本身含点（转义）；
* ``servers.0.name`` 用数字下标访问数组元素。
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Self

from . import codec, export
from .errors import (
    ConfigExistsError,
    ConfigFileNotFoundError,
    ConfigKeyError,
    ConfigPathError,
)

__all__ = ["TomlDocument", "lookup", "parse_path"]

_MISSING = object()


def lookup(data: Any, path: str | tuple[str, ...] = "", default: Any = _MISSING) -> Any:
    """在嵌套结构里按点号路径取值；``default`` 给定时不抛 :class:`ConfigKeyError`。"""
    parts = parse_path(path) if isinstance(path, str) else path
    if not parts:
        return data
    try:
        return _walk(data, parts)
    except ConfigKeyError:
        if default is not _MISSING:
            return default
        raise


def parse_path(text: str) -> tuple[str, ...]:
    """把 ``a.b.0.c`` 拆成 ``("a", "b", "0", "c")``；支持 ``\\.`` 转义。"""
    if text is None:
        raise ConfigPathError("路径不能为 None")
    if not isinstance(text, str):
        raise ConfigPathError(f"路径必须是字符串，收到 {type(text).__name__}")
    stripped = text.strip()
    if not stripped:
        return ()

    parts: list[str] = []
    buffer: list[str] = []
    escaped = False
    for char in stripped:
        if escaped:
            buffer.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ".":
            parts.append("".join(buffer))
            buffer = []
        else:
            buffer.append(char)
    if escaped:
        buffer.append("\\")
    parts.append("".join(buffer))

    if any(part == "" for part in parts):
        raise ConfigPathError(f"路径含空段：{text!r}（多余的 '.' 或末尾有点）")
    return tuple(parts)


def _render_path(parts: tuple[str, ...]) -> str:
    return ".".join(parts) if parts else ""


def _index(node: Any, key: str, trail: tuple[str, ...]) -> Any:
    if isinstance(node, Mapping):
        if key not in node:
            raise ConfigKeyError(f"配置项不存在：{_render_path(trail)}")
        return node[key]
    if isinstance(node, list):
        try:
            return node[int(key)]
        except (ValueError, IndexError):
            raise ConfigKeyError(f"数组下标越界或非法：{_render_path(trail)}") from None
    raise ConfigKeyError(f"{_render_path(trail[:-1]) or '<root>'} 不是可下钻的容器")


def _assign(node: Any, parts: tuple[str, ...], value: Any) -> None:
    key = parts[0]
    if len(parts) == 1:
        if isinstance(node, Mapping):
            node[key] = value  # type: ignore[index]
            return
        if isinstance(node, list):
            try:
                node[int(key)] = value
            except (ValueError, IndexError):
                raise ConfigKeyError(f"数组下标越界或非法：{key}") from None
            return
        raise ConfigPathError("只能在字典或数组上赋值")
    child = _index(node, key, parts)
    _assign(child, parts[1:], value)


def _remove(node: Any, parts: tuple[str, ...]) -> bool:
    key = parts[0]
    if len(parts) == 1:
        if isinstance(node, Mapping) and key in node:
            del node[key]  # type: ignore[attr-defined]
            return True
        if isinstance(node, list):
            try:
                node.pop(int(key))
                return True
            except (ValueError, IndexError):
                return False
        return False
    try:
        child = _index(node, key, parts)
    except ConfigKeyError:
        return False
    return _remove(child, parts[1:])


def _deep_merge(target: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """嵌套字典递归合并；数组与标量直接替换。"""
    for key, value in patch.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            _deep_merge(current, value)
        else:
            target[key] = value
    return target


class TomlDocument:
    """一份 TOML 配置文件的内存映像。"""

    __slots__ = (
        "_data",
        "_dirty",
        "_loaded",
        "atomic",
        "backup",
        "encoding",
        "header",
        "path",
    )

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        header: str = "",
        encoding: str = "utf-8",
        atomic: bool = True,
        backup: bool = False,
    ) -> None:
        self.path = Path(path).expanduser()
        self.header = header
        self.encoding = encoding
        self.atomic = atomic
        self.backup = backup
        self._data: dict[str, Any] = {}
        self._loaded = False
        self._dirty = False

    # ------------------------------------------------------------ 构造
    @classmethod
    def create(
        cls,
        path: str | os.PathLike[str],
        data: Mapping[str, Any] | None = None,
        *,
        overwrite: bool = False,
        header: str = "",
        atomic: bool = True,
        backup: bool = False,
    ) -> Self:
        """初始化一份配置文件并立即落盘。"""
        document = cls(path, header=header, atomic=atomic, backup=backup)
        if document.path.exists() and not overwrite:
            raise ConfigExistsError(
                f"配置文件已存在：{document.path}（如需覆盖请显式传入 overwrite=True）"
            )
        document._data = dict(data or {})
        document._loaded = True
        document._dirty = True
        document.save()
        return document

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        missing_ok: bool = False,
        header: str = "",
        atomic: bool = True,
        backup: bool = False,
    ) -> Self:
        """读取已有文件；``missing_ok=True`` 时文件不存在会得到一个空文档。"""
        document = cls(path, header=header, atomic=atomic, backup=backup)
        document.reload(missing_ok=missing_ok)
        return document

    # ------------------------------------------------------------ 状态
    @property
    def data(self) -> dict[str, Any]:
        """当前内容（直接暴露内部字典，便于只读遍历）。"""
        return self._data

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def dirty(self) -> bool:
        return self._dirty

    def __repr__(self) -> str:
        return (
            f"TomlDocument(path={str(self.path)!r}, "
            f"keys={len(self._data)}, dirty={self._dirty})"
        )

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __contains__(self, path: object) -> bool:
        if not isinstance(path, str):
            return False
        try:
            self.get(path)
        except ConfigKeyError:
            return False
        return True

    def __getitem__(self, path: str) -> Any:
        return self.get(path)

    def __setitem__(self, path: str, value: Any) -> None:
        self.set(path, value)

    # ------------------------------------------------------------ 读
    def reload(self, *, missing_ok: bool = False) -> Self:
        if not self.path.is_file():
            if not missing_ok:
                raise ConfigFileNotFoundError(f"配置文件不存在：{self.path}")
            self._data = {}
        else:
            self._data = codec.load_file(self.path, encoding=self.encoding)
        self._loaded = True
        self._dirty = False
        return self

    def get(self, path: str | tuple[str, ...] = "", default: Any = _MISSING) -> Any:
        return lookup(self._data, path, default)

    def get_all(self, prefix: str = "") -> dict[str, Any]:
        """取某个子树（``prefix=""`` 表示整个文档）。"""
        node = self.get(prefix)
        if not isinstance(node, Mapping):
            raise ConfigKeyError(f"{prefix or '<root>'} 不是字典，无法取子树")
        return dict(node)

    def keys(self, prefix: str = "") -> list[str]:
        node = self.get(prefix)
        if not isinstance(node, Mapping):
            raise ConfigKeyError(f"{prefix or '<root>'} 不是字典，没有键")
        return list(node)

    def contains(self, path: str) -> bool:
        return path in self

    # ------------------------------------------------------------ 写
    def set(self, path: str, value: Any) -> Self:
        parts = parse_path(path)
        if not parts:
            raise ConfigPathError("不能整体赋值，请提供具体路径（如 'app.debug'）")
        _walk_create(self._data, parts, value)
        self._dirty = True
        return self

    def unset(self, path: str, *, missing_ok: bool = True) -> bool:
        parts = parse_path(path)
        if not parts:
            raise ConfigPathError("不能删除根节点")
        removed = _remove(self._data, parts)
        if not removed and not missing_ok:
            raise ConfigKeyError(f"配置项不存在：{path}")
        self._dirty = self._dirty or removed
        return removed

    def update(self, patch: Mapping[str, Any], *, deep: bool = True) -> Self:
        """批量写入。**键按点号路径解释**，与 :meth:`get` / :meth:`set` 完全一致。

        * ``{"log.level": "INFO"}`` 等价于 ``set("log.level", "INFO")``；
        * ``{"app": {"debug": True}}`` 会与已有的 ``[app]`` 深合并，而不是整体替换；
        * 键名本身含点时用 ``\\.`` 转义：``{"a\\.b": 1}`` 写的是顶层键 ``"a.b"``。

        数组和标量整体替换（不做元素级合并，避免语义含糊）。
        """
        if not isinstance(patch, Mapping):
            raise ConfigPathError("update 需要字典")
        for key, value in patch.items():
            existing = lookup(self._data, key, None)
            if deep and isinstance(existing, dict) and isinstance(value, Mapping):
                merged = dict(existing)
                _deep_merge(merged, value)
                self.set(key, merged)
            else:
                self.set(key, value)
        return self

    def replace(self, data: Mapping[str, Any]) -> Self:
        self._data = dict(data)
        self._dirty = True
        return self

    def clear(self) -> Self:
        self._data.clear()
        self._dirty = True
        return self

    # ------------------------------------------------------------ 落盘/导出
    def dump(self, *, fmt: str = "toml", **kwargs: Any) -> str:
        """把当前内容渲染成文本（不写盘）。"""
        return self.export(fmt, **kwargs)

    def export(
        self,
        fmt: str = "toml",
        *,
        header: str | None = None,
        indent: int = 2,
        sort_keys: bool = False,
        sep: str = "__",
        prefix: str = "",
        upper: bool = True,
        redact: bool = False,
    ) -> str:
        match fmt.lower():
            case "toml":
                return codec.dumps(
                    self._data, header=self.header if header is None else header
                )
            case "json":
                return export.dumps_json(
                    self._data, indent=indent, sort_keys=sort_keys, redact=redact
                )
            case "env" | "dotenv":
                return export.dumps_env(self._data, sep=sep, prefix=prefix, upper=upper)
            case _:
                from .errors import ConfigUnsupportedError

                raise ConfigUnsupportedError(
                    f"不支持的导出格式：{fmt!r}（可选 toml / json / env）"
                )

    def save(
        self,
        *,
        backup: bool | None = None,
        atomic: bool | None = None,
        header: str | None = None,
    ) -> Path:
        text = codec.dumps(self._data, header=self.header if header is None else header)
        codec.write_file(
            self.path,
            text,
            encoding=self.encoding,
            atomic=self.atomic if atomic is None else atomic,
            backup=self.backup if backup is None else backup,
        )
        self._dirty = False
        self._loaded = True
        return self.path

    def unlink(self) -> bool:
        """删除配置文件（连同 ``.bak``），返回主文件是否真的被删掉。"""
        removed = False
        for target in (self.path, self.path.with_name(self.path.name + ".bak")):
            try:
                target.unlink()
            except FileNotFoundError:
                continue
            removed = removed or target == self.path
        self._data = {}
        self._dirty = False
        return removed


def _walk(node: Any, parts: tuple[str, ...]) -> Any:
    current = node
    for depth, key in enumerate(parts):
        current = _index(current, key, parts[: depth + 1])
    return current


def _walk_create(node: dict[str, Any], parts: tuple[str, ...], value: Any) -> None:
    current: Any = node
    for depth, key in enumerate(parts[:-1]):
        trail = parts[: depth + 1]
        if isinstance(current, dict) and key not in current:
            current[key] = {}
        current = _index(current, key, trail)
        if not isinstance(current, (dict, list)):
            raise ConfigPathError(
                f"{_render_path(trail)} 已经是标量，无法在其下继续赋值"
            )
    if not isinstance(current, (dict, list)):
        raise ConfigPathError("只能在字典或数组上赋值")
    current[parts[-1]] = value
