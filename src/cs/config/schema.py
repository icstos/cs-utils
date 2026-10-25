"""dataclass 模型的类型强制、校验与序列化。

这是整个包的核心：把「TOML 读出来的宽松字典 / 环境变量里的字符串」安全地变成
dataclass 实例，也把实例换回可写入 TOML 的普通字典。

支持的类型注解（其余抛 :class:`ConfigUnsupportedError`）：

* 标量：``str`` / ``int`` / ``float`` / ``bool`` / ``pathlib.Path`` / ``Decimal``
* 时间：``datetime`` / ``date`` / ``time``
* 容器：``list[T]`` / ``set[T]`` / ``frozenset[T]`` / ``tuple[T, ...]`` / ``dict[str, T]``
* 联合：``X | None``（其余联合按顺序逐个尝试）
* 字面量：``Literal["a", "b"]``
* 枚举：任何 ``enum.Enum`` 子类（按值或名字匹配）
* 嵌套：任意 ``@dataclass``（递归校验）

校验策略是**收集全部问题再抛**，而不是撞到第一个就停 —— 配置文件里一次看全更省事。
"""

from __future__ import annotations

import copy
import enum
import re
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from .errors import (
    ConfigSchemaError,
    ConfigUnsupportedError,
    ConfigValidationError,
    Issue,
)
from .fields import MISSING, Option, get_option

__all__ = [
    "FieldSpec",
    "build",
    "describe",
    "dump",
    "field_specs",
]

#: TOML 能原生表达的值（写入前的一切都必须落在这个范围内）
type TomlScalar = str | int | float | bool | datetime | date | time
type TomlValue = TomlScalar | list[TomlValue] | dict[str, TomlValue]

_TRUTHY = frozenset({"true", "yes", "1", "on", "y", "t"})
_FALSY = frozenset({"false", "no", "0", "off", "n", "f", ""})
_NONE_WORDS = frozenset({"none", "null", "nil", "~"})

_CONTAINER_ORIGINS = {
    dict: dict,
    list: list,
    set: set,
    frozenset: frozenset,
    tuple: tuple,
    Sequence: list,
    typing.Sequence: list,
    Mapping: dict,
    typing.Mapping: dict,
}

_SCALARS: tuple[type, ...] = (
    str,
    int,
    float,
    bool,
    Path,
    Decimal,
    datetime,
    date,
    time,
)
#: 内层函数用来表达"这个标量分支没命中"的哨兵
_UNHANDLED = object()


# ---------------------------------------------------------------- 字段描述
class FieldSpec:
    """一个模型字段的解析结果：名字、TOML 键名、注解、约束、默认值、嵌套模型。"""

    __slots__ = (
        "annotation",
        "default",
        "has_default",
        "key",
        "name",
        "nested",
        "option",
    )

    def __init__(
        self,
        *,
        name: str,
        key: str,
        annotation: Any,
        option: Option | None,
        has_default: bool,
        default: Any,
        nested: type | None,
    ) -> None:
        self.name = name
        self.key = key
        self.annotation = annotation
        self.option = option
        self.has_default = has_default
        self.default = default
        self.nested = nested

    @property
    def secret(self) -> bool:
        return bool(self.option and self.option.secret)

    @property
    def description(self) -> str:
        return self.option.description if self.option else ""

    def __repr__(self) -> str:
        return (
            f"FieldSpec(name={self.name!r}, key={self.key!r}, "
            f"nested={getattr(self.nested, '__name__', None)!r})"
        )


def _unwrap(annotation: Any) -> Any:
    """剥掉 ``Annotated[T, ...]`` 的外壳。"""
    while get_origin(annotation) is typing.Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _hints(cls: type) -> dict[str, Any]:
    try:
        return get_type_hints(cls, include_extras=True)
    except Exception:  # noqa: BLE001 - 前向引用在局部作用域里可能解析失败
        return dict(getattr(cls, "__annotations__", {}))


def _is_optional(annotation: Any) -> tuple[bool, Any]:
    """``X | None`` → ``(True, X)``；否则 ``(False, annotation)``。"""
    args = get_args(annotation)
    if get_origin(annotation) in (Union, types.UnionType):
        rest = [arg for arg in args if arg is not type(None)]
        if len(rest) == len(args) - 1:
            return True, rest[0] if len(rest) == 1 else Union[tuple(rest)]  # noqa: UP007
    return False, annotation


def _ensure_model(cls: type) -> None:
    if not isinstance(cls, type) or not is_dataclass(cls):
        raise ConfigSchemaError(
            f"配置模型必须是 dataclass 类型，收到 {cls!r}"
            "（请用 @dataclass 装饰，推荐 @dataclass(slots=True)）"
        )


@lru_cache(maxsize=256)
def field_specs(cls: type) -> tuple[FieldSpec, ...]:
    """解析 dataclass 的全部字段（结果缓存，模型类型本身可哈希）。"""
    _ensure_model(cls)
    hints = _hints(cls)
    specs: list[FieldSpec] = []
    for item in fields(cls):
        annotation = _unwrap(hints.get(item.name, Any))
        option = get_option(item)
        key = option.alias if option and option.alias else item.name

        has_default = False
        default: Any = None
        if option is not None and option.has_default:
            has_default, default = True, option.make_default()
        elif item.default is not MISSING:
            has_default, default = True, item.default
        elif item.default_factory is not MISSING:  # type: ignore[comparison-overlap]
            has_default, default = True, item.default_factory()

        _, bare = _is_optional(annotation)
        nested = bare if (isinstance(bare, type) and is_dataclass(bare)) else None
        specs.append(
            FieldSpec(
                name=item.name,
                key=key,
                annotation=annotation,
                option=option,
                has_default=has_default,
                default=default,
                nested=nested,
            )
        )
    return tuple(specs)


# ---------------------------------------------------------------- 值强制
def _describe_type(annotation: Any) -> str:
    return getattr(annotation, "__name__", None) or str(annotation)


def _fail(issues: list[Issue], path: str, message: str, value: Any = None) -> None:
    issues.append(Issue(path, message, value))


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUTHY:
            return True
        if lowered in _FALSY:
            return False
    return None


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        text = value.strip().replace("_", "")
        try:
            return int(text, 10)
        except ValueError:
            try:
                number = float(text)
            except ValueError:
                return None
            return int(number) if number.is_integer() else None
    return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace("_", ""))
        except ValueError:
            return None
    return None


def _to_str(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, Decimal)):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return None


def _coerce_scalar(
    value: Any, annotation: Any, *, path: str, issues: list[Issue], strict: bool
) -> Any:
    if annotation is bool:
        if strict and not isinstance(value, bool):
            _fail(issues, path, "需要布尔值", value)
            return None
        converted = _to_bool(value)
        if converted is None:
            _fail(issues, path, "需要布尔值（true/false）", value)
        return converted

    if annotation is int:
        if strict and not (isinstance(value, int) and not isinstance(value, bool)):
            _fail(issues, path, "需要整数", value)
            return None
        converted = _to_int(value)
        if converted is None:
            _fail(issues, path, "需要整数", value)
        return converted

    if annotation is float:
        if strict and not isinstance(value, (int, float)):
            _fail(issues, path, "需要浮点数", value)
            return None
        converted = _to_float(value)
        if converted is None:
            _fail(issues, path, "需要浮点数", value)
        return converted

    if annotation is str:
        if strict and not isinstance(value, str):
            _fail(issues, path, "需要字符串", value)
            return None
        converted = _to_str(value)
        if converted is None:
            _fail(issues, path, "需要字符串", value)
        return converted

    if annotation is Path:
        text = _to_str(value)
        if text is None:
            _fail(issues, path, "需要路径字符串", value)
            return None
        return Path(text)

    if annotation is Decimal:
        try:
            return Decimal(str(value).strip())
        except InvalidOperation:
            _fail(issues, path, "需要十进制数", value)
            return None

    if annotation is datetime:
        if isinstance(value, datetime):
            return value
        text = _to_str(value)
        if text:
            try:
                return datetime.fromisoformat(text)
            except ValueError:
                pass
        _fail(issues, path, "需要 ISO 8601 时间（如 2026-01-02T03:04:05）", value)
        return None

    if annotation is date:
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        text = _to_str(value)
        if text:
            try:
                return date.fromisoformat(text)
            except ValueError:
                pass
        _fail(issues, path, "需要 ISO 8601 日期（如 2026-01-02）", value)
        return None

    if annotation is time:
        if isinstance(value, time):
            return value
        text = _to_str(value)
        if text:
            try:
                return time.fromisoformat(text)
            except ValueError:
                pass
        _fail(issues, path, "需要 ISO 8601 时间（如 03:04:05）", value)
        return None

    return _UNHANDLED


def _coerce_enum(
    value: Any, annotation: type[enum.Enum], *, path: str, issues: list[Issue]
) -> Any:
    if isinstance(value, annotation):
        return value
    for member in annotation:
        if value == member.value:
            return member
    if isinstance(value, str):
        text = value.strip()
        for member in annotation:
            if text == member.name or text == str(member.value):
                return member
    _fail(
        issues,
        path,
        f"需要 {annotation.__name__} 的取值之一："
        f"{', '.join(str(member.value) for member in annotation)}",
        value,
    )
    return None


def _coerce_sequence(
    value: Any,
    item_type: Any,
    *,
    origin: Any,
    path: str,
    issues: list[Issue],
    strict: bool,
) -> Any:
    if isinstance(value, str):
        # 环境变量里只能给字符串，逗号分隔是运维里最常见的写法
        if strict:
            _fail(issues, path, "需要数组，收到字符串", value)
            return None
        items: list[Any] = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Mapping):
        _fail(issues, path, "需要数组，收到字典", value)
        return None
    elif isinstance(value, (Sequence, set, frozenset)):
        items = list(value)
    else:
        _fail(issues, path, "需要数组", value)
        return None

    converted = [
        _coerce(item, item_type, path=f"{path}[{index}]", issues=issues, strict=strict)
        for index, item in enumerate(items)
    ]
    if origin is set:
        return set(converted)
    if origin is frozenset:
        return frozenset(converted)
    if origin is tuple:
        return tuple(converted)
    return converted


def _coerce_mapping(
    value: Any, item_type: Any, *, path: str, issues: list[Issue], strict: bool
) -> Any:
    if not isinstance(value, Mapping):
        _fail(issues, path, "需要字典", value)
        return None
    return {
        str(key): _coerce(
            item, item_type, path=f"{path}.{key}", issues=issues, strict=strict
        )
        for key, item in value.items()
    }


def _coerce(
    value: Any, annotation: Any, *, path: str, issues: list[Issue], strict: bool
) -> Any:
    annotation = _unwrap(annotation)

    if annotation is Any or annotation is object:
        return value

    optional, inner = _is_optional(annotation)
    if optional:
        if value is None or (
            isinstance(value, str) and value.strip().lower() in _NONE_WORDS
        ):
            return None
        if get_origin(inner) in (Union, types.UnionType):
            return _coerce_union(value, inner, path=path, issues=issues, strict=strict)
        return _coerce(value, inner, path=path, issues=issues, strict=strict)

    origin = get_origin(annotation)

    if origin is Literal:
        allowed = get_args(annotation)
        candidate = value
        if candidate not in allowed:
            for option in allowed:
                if isinstance(option, str) and isinstance(candidate, str):
                    if candidate.strip() == option:
                        return option
                elif type(option) is not type(candidate) and isinstance(candidate, str):
                    converted = _coerce(
                        candidate, type(option), path=path, issues=[], strict=False
                    )
                    if converted == option:
                        return option
            _fail(issues, path, f"取值必须是 {allowed} 之一", value)
            return None
        return candidate

    if get_origin(annotation) in (Union, types.UnionType):
        return _coerce_union(value, annotation, path=path, issues=issues, strict=strict)

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return _coerce_enum(value, annotation, path=path, issues=issues)

    if annotation in _SCALARS:
        result = _coerce_scalar(
            value, annotation, path=path, issues=issues, strict=strict
        )
        if result is not _UNHANDLED:
            return result

    container = _CONTAINER_ORIGINS.get(origin)
    if container is not None:
        args = get_args(annotation)
        if container is dict:
            return _coerce_mapping(
                value,
                args[1] if len(args) == 2 else Any,
                path=path,
                issues=issues,
                strict=strict,
            )
        return _coerce_sequence(
            value,
            args[0] if args else Any,
            origin=container,
            path=path,
            issues=issues,
            strict=strict,
        )

    if isinstance(annotation, type) and is_dataclass(annotation):
        if isinstance(value, annotation):
            return value
        if not isinstance(value, Mapping):
            _fail(issues, path, f"需要字典以构造 {annotation.__name__}", value)
            return None
        return _build_model(annotation, value, path=path, issues=issues, strict=strict)

    raise ConfigUnsupportedError(
        f"字段 {path or '<root>'} 的注解 {_describe_type(annotation)} 暂不支持"
    )


def _coerce_union(
    value: Any, annotation: Any, *, path: str, issues: list[Issue], strict: bool
) -> Any:
    """联合类型：按声明顺序逐个尝试，第一个无错的结果胜出。"""
    best: list[Issue] = []
    for candidate in get_args(annotation):
        if candidate is type(None):
            if value is None:
                return None
            continue
        attempt: list[Issue] = []
        result = _coerce(value, candidate, path=path, issues=attempt, strict=strict)
        if not attempt:
            return result
        if not best:
            best = attempt
    issues.extend(best)
    return None


# ---------------------------------------------------------------- 约束检查
def _check_constraints(
    value: Any, *, path: str, option: Option | None, issues: list[Issue]
) -> Any:
    if option is None or value is None:
        return value

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if option.ge is not None and value < option.ge:
            _fail(issues, path, f"不能小于 {option.ge}", value)
        if option.gt is not None and value <= option.gt:
            _fail(issues, path, f"必须大于 {option.gt}", value)
        if option.le is not None and value > option.le:
            _fail(issues, path, f"不能大于 {option.le}", value)
        if option.lt is not None and value >= option.lt:
            _fail(issues, path, f"必须小于 {option.lt}", value)

    if isinstance(value, (str, bytes, list, tuple, set, frozenset, dict)):
        length = len(value)
        if option.min_length is not None and length < option.min_length:
            _fail(issues, path, f"长度不能小于 {option.min_length}", value)
        if option.max_length is not None and length > option.max_length:
            _fail(issues, path, f"长度不能大于 {option.max_length}", value)

    if (
        option.pattern is not None
        and isinstance(value, str)
        and re.search(option.pattern, value) is None
    ):
        _fail(issues, path, f"不匹配模式 {option.pattern!r}", value)

    if option.choices is not None and value not in option.choices:
        _fail(issues, path, f"取值必须是 {option.choices} 之一", value)

    return value


# ---------------------------------------------------------------- 模型构建
def _fresh_default(spec: FieldSpec) -> Any:
    """每次构建都重新取一遍默认值，避免可变默认值（list/dict/嵌套实例）在多次构建间共享。"""
    if spec.option is not None and spec.option.has_default:
        return copy.deepcopy(spec.option.make_default())
    return copy.deepcopy(spec.default)


def _build_model(
    cls: type,
    data: Mapping[str, Any],
    *,
    path: str,
    issues: list[Issue],
    strict: bool,
    extra: str = "forbid",
) -> Any:
    specs = field_specs(cls)
    known = {spec.key for spec in specs} | {spec.name for spec in specs}

    if extra == "forbid":
        for key in data:
            if key not in known:
                _fail(
                    issues,
                    f"{path}.{key}" if path else str(key),
                    f"{cls.__name__} 没有这个配置项",
                    data[key],
                )

    values: dict[str, Any] = {}
    for spec in specs:
        child = f"{path}.{spec.key}" if path else spec.key
        if spec.key in data:
            raw = data[spec.key]
        elif spec.name in data and spec.name != spec.key:
            raw = data[spec.name]
        elif spec.has_default:
            values[spec.name] = _fresh_default(spec)
            continue
        else:
            _fail(issues, child, "缺少必填项")
            continue

        if raw is None and spec.has_default and not _is_optional(spec.annotation)[0]:
            values[spec.name] = _fresh_default(spec)
            continue

        converted = _coerce(
            raw, spec.annotation, path=child, issues=issues, strict=strict
        )
        values[spec.name] = _check_constraints(
            converted, path=child, option=spec.option, issues=issues
        )

    if issues:
        return None
    try:
        return cls(**values)
    except TypeError as exc:  # pragma: no cover - 防御：字段名解析异常
        _fail(issues, path or "<root>", f"构造 {cls.__name__} 失败：{exc}")
        return None


def build(
    cls: type,
    data: Mapping[str, Any],
    *,
    strict: bool = False,
    extra: Literal["forbid", "ignore"] = "forbid",
) -> Any:
    """把普通字典校验成 ``cls`` 实例；有错抛 :class:`ConfigValidationError`（一次列全）。"""
    _ensure_model(cls)
    if not isinstance(data, Mapping):
        raise ConfigValidationError([Issue("<root>", "配置内容必须是字典", data)])
    issues: list[Issue] = []
    result = _build_model(cls, data, path="", issues=issues, strict=strict, extra=extra)
    if issues:
        raise ConfigValidationError(issues)
    return result


# ---------------------------------------------------------------- 反向序列化
def dump(value: Any, *, mask_secrets: bool = False) -> Any:
    """dataclass 实例 → 可写入 TOML 的普通结构。

    ``None`` 会被丢弃（TOML 没有 null），``set``/``tuple`` 转数组，
    ``Path`` 转字符串，``Enum`` 取其值。
    """
    if is_dataclass(value) and not isinstance(value, type):
        result: dict[str, Any] = {}
        for spec in field_specs(type(value)):
            item = getattr(value, spec.name, None)
            if item is None:
                continue
            if mask_secrets and spec.secret and item != "":
                result[spec.key] = "***"
            else:
                result[spec.key] = dump(item)
        return result
    if value is None:
        return None  # TOML 没有 null，由调用方决定丢弃
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            converted = dump(item)
            if converted is not None:
                result[str(key)] = converted
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        return [dump(item) for item in value]
    if isinstance(value, enum.Enum):
        return dump(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (str, int, float, bool, datetime, date, time)):
        return value
    return str(value)


# ---------------------------------------------------------------- Schema 描述
_JSON_TYPES: tuple[tuple[type, str], ...] = (
    (bool, "boolean"),
    (int, "integer"),
    (float, "number"),
    (str, "string"),
    (Path, "string"),
    (Decimal, "number"),
    (datetime, "string"),
    (date, "string"),
    (time, "string"),
)


def _type_name(annotation: Any) -> str:
    annotation = _unwrap(annotation)
    optional, inner = _is_optional(annotation)
    if optional:
        return f"{_type_name(inner)} | null"
    origin = get_origin(annotation)
    if origin is Literal:
        return " | ".join(repr(arg) for arg in get_args(annotation))
    if origin in (Union, types.UnionType):
        return " | ".join(_type_name(arg) for arg in get_args(annotation))
    container = _CONTAINER_ORIGINS.get(origin)
    if container is not None:
        args = get_args(annotation)
        inner_name = ", ".join(_type_name(arg) for arg in args) or "any"
        return f"{getattr(origin, '__name__', 'list')}[{inner_name}]"
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return annotation.__name__
    if annotation in _SCALARS:
        return getattr(annotation, "__name__", str(annotation))
    return getattr(annotation, "__name__", str(annotation))


def _json_type(annotation: Any) -> str:
    annotation = _unwrap(annotation)
    _, inner = _is_optional(annotation)
    if get_origin(inner) in (list, set, frozenset, tuple):
        return "array"
    if get_origin(inner) is dict:
        return "object"
    if isinstance(inner, type) and is_dataclass(inner):
        return "object"
    for scalar, name in _JSON_TYPES:
        if inner is scalar:
            return name
    return "string"


def describe(cls: type) -> dict[str, Any]:
    """把模型描述成 JSON Schema 风格的结构（用于 ``config_schema`` 工具与文档）。"""
    _ensure_model(cls)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for spec in field_specs(cls):
        entry: dict[str, Any] = {
            "type": _json_type(spec.annotation),
            "annotation": _type_name(spec.annotation),
        }
        if spec.description:
            entry["description"] = spec.description
        if spec.has_default:
            entry["default"] = dump(_fresh_default(spec))
        else:
            required.append(spec.key)
        option = spec.option
        if option is not None:
            for name in ("ge", "gt", "le", "lt", "min_length", "max_length", "pattern"):
                value = getattr(option, name)
                if value is not None:
                    entry[name] = value
            if option.choices:
                entry["enum"] = list(option.choices)
            if option.secret:
                entry["writeOnly"] = True
        if spec.nested is not None:
            entry = {**describe(spec.nested), **entry, "type": "object"}
            entry["title"] = spec.nested.__name__
        properties[spec.key] = entry
    result: dict[str, Any] = {
        "title": cls.__name__,
        "type": "object",
        "properties": properties,
    }
    if required:
        result["required"] = required
    doc = (cls.__doc__ or "").strip()
    if doc:
        result["description"] = doc
    return result
