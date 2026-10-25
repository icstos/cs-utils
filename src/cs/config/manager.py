"""``ConfigManager`` —— 带 dataclass schema 的配置管理器。

文件内容始终是唯一事实来源（:class:`TomlDocument` 持有原始字典）；本类负责：

* **加载即校验**：读盘 → 环境变量覆盖 → 按 dataclass 强制类型 → 聚合报告全部错误；
* **改完立刻校验**：``set`` / ``update`` 若产生非法值会当场抛错，不让坏值落到文件里；
* **保存只写文件内容**：环境变量覆盖是只读叠加层，不会被「烤」进配置文件；
* **导出时脱敏**：``secret=True`` 的字段在导出时替换成 ``***``。

它是 lazy 的：``ConfigManager(Model, path)`` 只记住参数，``load()`` 才真正读盘。
"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Literal, Self

from . import codec, export, schema
from .document import TomlDocument, lookup
from .errors import (
    ConfigExistsError,
    ConfigKeyError,
    ConfigPathError,
    ConfigSchemaError,
    ConfigUnsupportedError,
)

__all__ = ["ConfigManager"]

type Source = Literal["effective", "file"]

_MISSING = object()


def _convert_env(value: str) -> Any:
    """环境变量只能是字符串，这里做保守推断（真值/整数/浮点/其余保持字符串）。"""
    text = value.strip()
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _merge(target: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """嵌套字典递归合并；数组与标量直接替换。"""
    for key, value in patch.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            _merge(current, value)
        else:
            target[key] = value
    return target


def _redact(data: Mapping[str, Any], model: type | None) -> dict[str, Any]:
    """按模型里的 ``secret=True`` 给敏感字段打码（递归进子表）。"""
    if model is None or not is_dataclass(model):
        return dict(data)
    specs = {spec.key: spec for spec in schema.field_specs(model)}
    result: dict[str, Any] = {}
    for key, value in data.items():
        spec = specs.get(key)
        if spec is not None and spec.secret and value not in ("", None):
            result[key] = "***"
        elif (
            spec is not None and spec.nested is not None and isinstance(value, Mapping)
        ):
            result[key] = _redact(value, spec.nested)
        else:
            result[key] = value
    return result


class ConfigManager[T]:
    """把一份 TOML 文件绑定到一个 dataclass 模型上。"""

    __slots__ = (
        "_document",
        "_instance",
        "atomic",
        "backup",
        "encoding",
        "env_prefix",
        "env_sep",
        "extra",
        "header",
        "model",
        "path",
        "strict",
    )

    def __init__(
        self,
        model: type[T] | None = None,
        path: str | os.PathLike[str] = "config.toml",
        *,
        header: str = "",
        encoding: str = "utf-8",
        atomic: bool = True,
        backup: bool = False,
        env_prefix: str | None = None,
        env_sep: str = "__",
        strict: bool = False,
        extra: Literal["forbid", "ignore"] = "forbid",
    ) -> None:
        if model is not None and not (isinstance(model, type) and is_dataclass(model)):
            raise ConfigSchemaError(
                f"配置模型必须是 dataclass 类型，收到 {model!r}"
                "（推荐 @dataclass(slots=True)）"
            )
        self.model: type[T] | None = model
        self.path = Path(path).expanduser()
        self.header = header
        self.encoding = encoding
        self.atomic = atomic
        self.backup = backup
        self.env_prefix = env_prefix
        self.env_sep = env_sep
        self.strict = strict
        self.extra = extra
        self._document = TomlDocument(
            self.path,
            header=header,
            encoding=encoding,
            atomic=atomic,
            backup=backup,
        )
        self._instance: T | None = None

    # ------------------------------------------------------------ 构造
    @classmethod
    def typed(
        cls,
        model: type[T],
        path: str | os.PathLike[str] = "config.toml",
        **kwargs: Any,
    ) -> Self:
        """强类型构造（比位置参数更好读）。"""
        return cls(model, path, **kwargs)

    @classmethod
    def raw(
        cls, path: str | os.PathLike[str] = "config.toml", **kwargs: Any
    ) -> ConfigManager[dict[str, Any]]:
        """无 schema 模式：只做文件级读写，不做类型校验。"""
        return cls(None, path, **kwargs)

    @classmethod
    def from_env(
        cls,
        model: type[T] | None = None,
        *,
        env: str = "CS_CONFIG",
        fallback: str | os.PathLike[str] = "config.toml",
        **kwargs: Any,
    ) -> Self:
        """配置文件路径取自环境变量（容器部署常见做法）。"""
        return cls(model, os.environ.get(env) or fallback, **kwargs)

    # ------------------------------------------------------------ 基础属性
    @property
    def document(self) -> TomlDocument:
        return self._document

    @property
    def exists(self) -> bool:
        return self._document.exists

    @property
    def loaded(self) -> bool:
        return self._document.loaded

    @property
    def dirty(self) -> bool:
        return self._document.dirty

    @property
    def name(self) -> str:
        return self.model.__name__ if self.model is not None else "raw"

    def __repr__(self) -> str:
        return (
            f"ConfigManager(model={self.name}, path={str(self.path)!r}, "
            f"loaded={self.loaded}, dirty={self.dirty})"
        )

    def __enter__(self) -> Self:
        if not self.loaded:
            self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    # ------------------------------------------------------------ 生命周期
    def load(self, *, missing_ok: bool = True) -> Self:
        """读盘并校验。

        文件不存在时：``missing_ok=True`` 会用「全默认值」构建（缺必填项则报校验错），
        无 schema 模式则得到空文档。
        """
        self._document.reload(missing_ok=missing_ok)
        self._refresh()
        return self

    def initialize(
        self,
        *,
        overwrite: bool = False,
        template: bool = False,
        data: Mapping[str, Any] | None = None,
    ) -> Path:
        """初始化配置文件（默认值或给定数据），返回写入路径。

        ``template=True`` 时写出**带注释的骨架**（更适合给人改），而不是纯默认值。
        """
        if self._document.exists and not overwrite:
            raise ConfigExistsError(
                f"配置文件已存在：{self.path}（如需覆盖请显式传入 overwrite=True）"
            )

        if data is not None:
            # 给的部分用给定值，没给的字段由默认值补齐 —— 生成的文件总是完整可用的
            payload: dict[str, Any] = (
                schema.dump(self.defaults()) if self.model is not None else {}
            )
            _merge(payload, data)
        elif self.model is not None:
            payload = schema.dump(self.defaults())
        else:
            payload = {}

        if template and self.model is not None:
            text = export.render_template(self.model, header=self.header)
            codec.write_file(
                self.path,
                text,
                encoding=self.encoding,
                atomic=self.atomic,
                backup=False,
            )
            self._document.reload(missing_ok=False)
        else:
            self._document.replace(payload)
            self._document.save()
        self._refresh()
        return self.path

    def reload(self, *, missing_ok: bool = True) -> Self:
        return self.load(missing_ok=missing_ok)

    def save(self, *, backup: bool | None = None, atomic: bool | None = None) -> Path:
        """写盘；有 schema 时先过一遍校验（坏配置绝不落盘）。"""
        if self.model is not None:
            self.instance()
        return self._document.save(backup=backup, atomic=atomic)

    # ------------------------------------------------------------ 取值
    def defaults(self) -> T:
        """只用默认值构造实例（缺必填项会抛校验错误）。"""
        if self.model is None:
            raise ConfigSchemaError("无 schema 模式没有默认值，请用 data() 取内容")
        return schema.build(self.model, {}, strict=self.strict, extra=self.extra)

    def env_overrides(self, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
        """把 ``PREFIX__A__B=1`` 形式的环境变量还原成嵌套字典。"""
        if not self.env_prefix:
            return {}
        source = os.environ if environ is None else environ
        # 前缀允许写成 "CS" / "CS_" / "CS__"，统一归一成 "CS__"（不重复叠加分隔符）
        name = self.env_prefix
        while name.endswith(self.env_sep[0]):
            name = name[:-1]
        prefix = f"{name}{self.env_sep}"
        tree: dict[str, Any] = {}
        for key, value in source.items():
            if not key.startswith(prefix):
                continue
            trail = [
                part.lower() for part in key[len(prefix) :].split(self.env_sep) if part
            ]
            if not trail:
                continue
            node = tree
            for part in trail[:-1]:
                child = node.get(part)
                if not isinstance(child, dict):
                    child = {}
                    node[part] = child
                node = child
            node[trail[-1]] = _convert_env(value)
        return tree

    def data(self, *, source: Source = "effective") -> dict[str, Any]:
        """取配置内容。

        ``source="file"`` 只看文件；``"effective"`` 叠加环境变量（程序实际看到的值）。
        """
        base = copy.deepcopy(self._document.data)
        if source == "file":
            return base
        overrides = self.env_overrides()
        return _merge(base, overrides) if overrides else base

    def instance(self) -> T:
        """取校验后的 dataclass 实例（缓存；写操作会让缓存失效重建）。"""
        if self.model is None:
            raise ConfigSchemaError("无 schema 模式没有实例，请用 data() / get()")
        if self._instance is None:
            self._instance = schema.build(
                self.model, self.data(), strict=self.strict, extra=self.extra
            )
        return self._instance

    def get(
        self,
        path: str = "",
        default: Any = _MISSING,
        *,
        source: Source = "effective",
    ) -> Any:
        """按点号路径取原始值（未做类型转换，类型转换在 :meth:`instance`）。

        不传 ``default`` 时，键不存在会抛 :class:`ConfigKeyError`（而不是返回哨兵）。
        """
        payload = self.data(source=source)
        if default is _MISSING:
            return lookup(payload, path)
        return lookup(payload, path, default)

    # ------------------------------------------------------------ 更新
    def set(self, path: str, value: Any, *, save: bool = False) -> Self:
        """设置一个值。

        语义是「要么全成功，要么什么都没变」：先快照，改完立刻校验，校验失败就回滚，
        所以一个被拒绝的 ``set`` 不会把内存里的配置留在非法状态。
        校验成功时会把**类型归一化后的值**写回（``"6543"`` → ``6543``）。
        """
        if not self.loaded:
            self.load()
        with self._atomic():
            self._document.set(path, value)
        self._normalize([path])
        if save:
            self.save()
        return self

    def update(
        self, patch: Mapping[str, Any], *, deep: bool = True, save: bool = False
    ) -> Self:
        """深合并且只校验一次（比多次 ``set`` 更高效，同样具备回滚语义）。"""
        if not self.loaded:
            self.load()
        with self._atomic():
            self._document.update(patch, deep=deep)
        self._normalize(list(patch))
        if save:
            self.save()
        return self

    def unset(self, path: str, *, missing_ok: bool = True, save: bool = False) -> bool:
        if not self.loaded:
            self.load()
        with self._atomic():
            removed = self._document.unset(path, missing_ok=missing_ok)
        if removed and save:
            self.save()
        return removed

    def replace(self, data: Mapping[str, Any], *, save: bool = False) -> Self:
        if not self.loaded:
            self.load()
        with self._atomic():
            self._document.replace(data)
        if self.model is not None:
            # 整体替换时把所有字段都归一化，文件里就是「模型的样子」
            self._document.replace(schema.dump(self.instance()))
        if save:
            self.save()
        return self

    # ------------------------------------------------------------ 导出/描述
    def export(
        self,
        fmt: str = "toml",
        *,
        redact: bool = False,
        source: Source = "file",
        header: str | None = None,
        indent: int = 2,
        sort_keys: bool = False,
        sep: str = "__",
        prefix: str = "",
        upper: bool = True,
    ) -> str:
        """导出为文本。默认导出**文件内容**（不含环境变量覆盖）。"""
        payload = self.data(source=source)
        if redact:
            payload = _redact(payload, self.model)
        match fmt.lower():
            case "toml":
                return codec.dumps(
                    payload, header=self.header if header is None else header
                )
            case "json":
                return export.dumps_json(payload, indent=indent, sort_keys=sort_keys)
            case "env" | "dotenv":
                return export.dumps_env(payload, sep=sep, prefix=prefix, upper=upper)
            case _:
                raise ConfigUnsupportedError(
                    f"不支持的导出格式：{fmt!r}（可选 toml / json / env）"
                )

    def dump(self, *, fmt: str = "toml", **kwargs: Any) -> str:
        return self.export(fmt, **kwargs)

    def schema(self) -> dict[str, Any]:
        """模型的 JSON Schema 风格描述。"""
        if self.model is None:
            raise ConfigSchemaError("无 schema 模式没有 schema")
        return schema.describe(self.model)

    def describe(self) -> dict[str, Any]:
        """一眼看懂的概况（CLI / MCP 都用它）。"""
        return {
            "path": str(self.path),
            "model": self.name,
            "exists": self.exists,
            "loaded": self.loaded,
            "dirty": self.dirty,
            "sections": sorted(self._document.data),
            "env_prefix": self.env_prefix,
            "size": self.path.stat().st_size if self.exists else 0,
        }

    def validate(self, data: Mapping[str, Any] | None = None) -> T:
        """校验给定内容（默认校验当前生效内容），返回实例或抛 :class:`ConfigValidationError`。"""
        if self.model is None:
            raise ConfigSchemaError("无 schema 模式无法校验")
        payload = self.data() if data is None else dict(data)
        return schema.build(self.model, payload, strict=self.strict, extra=self.extra)

    # ------------------------------------------------------------ 内部
    @contextmanager
    def _atomic(self) -> Iterator[None]:
        """写事务：校验不过就整份回滚，绝不留半截非法配置。"""
        snapshot = copy.deepcopy(self._document.data)
        self._instance = None
        try:
            yield
            if self.model is not None:
                self._instance = schema.build(
                    self.model, self.data(), strict=self.strict, extra=self.extra
                )
        except BaseException:
            self._document.replace(snapshot)
            self._instance = None
            raise

    def _normalize(self, paths: Sequence[str]) -> None:
        """把刚改过的键换成校验后的类型（``"6543"`` → ``6543``）。"""
        if self.model is None or self._instance is None:
            return
        dumped = schema.dump(self._instance)
        for path in paths:
            try:
                value = lookup(dumped, path)
            except ConfigKeyError:
                continue
            if lookup(self._document.data, path, _MISSING) is not _MISSING:
                try:
                    self._document.set(path, value)
                except (ConfigPathError, ConfigKeyError):
                    continue

    def _refresh(self) -> None:
        """重建校验缓存：非法内容当场抛错。"""
        self._instance = None
        if self.model is None:
            return
        self._instance = schema.build(
            self.model, self.data(), strict=self.strict, extra=self.extra
        )
