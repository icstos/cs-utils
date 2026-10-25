"""导出格式：JSON、环境变量、以及带注释的 TOML 模板。

导出是「只读」操作，永远不写文件 —— 调用方要落盘可以自己写。
默认 ``redact=True`` 时，标了 ``secret=True`` 的字段会被替换成 ``***``，
避免把密码打进日志或 MCP 返回值里。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import is_dataclass
from typing import Any

from .codec import format_value
from .errors import ConfigUnsupportedError
from .schema import dump, field_specs

__all__ = [
    "dumps_env",
    "dumps_json",
    "flatten",
    "render_template",
]


def flatten(
    data: Mapping[str, Any],
    *,
    sep: str = "__",
    prefix: str = "",
    upper: bool = True,
) -> dict[str, str]:
    """把嵌套字典压平成 ``A__B__C`` 形式的扁平映射（数组用逗号连接）。"""

    def render(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (list, tuple, set, frozenset)):
            return ",".join(render(item) for item in value)
        return str(value)

    result: dict[str, str] = {}

    def walk(node: Mapping[str, Any], trail: list[str]) -> None:
        for key, value in node.items():
            segment = f"{key}".upper() if upper else f"{key}"
            path = [*trail, segment]
            if isinstance(value, Mapping):
                if value:
                    walk(value, path)
                else:
                    result[sep.join(path)] = ""
            else:
                result[sep.join(path)] = render(value)

    walk(data, [prefix] if prefix else [])
    return result


def dumps_env(
    data: Mapping[str, Any],
    *,
    sep: str = "__",
    prefix: str = "",
    upper: bool = True,
) -> str:
    """导出成 ``KEY=VALUE`` 行（可直接喂给 ``.env`` 或 shell ``export``）。"""
    items = flatten(data, sep=sep, prefix=prefix, upper=upper)
    return "\n".join(f"{key}={value}" for key, value in items.items()) + "\n"


def dumps_json(
    data: Any,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    redact: bool = False,
) -> str:
    """导出成 JSON 文本（自动处理 datetime / Path / set / Enum / dataclass）。"""
    payload = dump(data, mask_secrets=redact)
    return json.dumps(
        payload, ensure_ascii=False, indent=indent, sort_keys=sort_keys, default=str
    )


def _constraint_note(spec: Any) -> str:
    option = spec.option
    if option is None:
        return ""
    bits: list[str] = []
    if option.ge is not None:
        bits.append(f">= {option.ge}")
    if option.gt is not None:
        bits.append(f"> {option.gt}")
    if option.le is not None:
        bits.append(f"<= {option.le}")
    if option.lt is not None:
        bits.append(f"< {option.lt}")
    if option.min_length is not None:
        bits.append(f"长度 >= {option.min_length}")
    if option.max_length is not None:
        bits.append(f"长度 <= {option.max_length}")
    if option.pattern is not None:
        bits.append(f"匹配 {option.pattern}")
    if option.choices:
        bits.append("可选 " + " / ".join(str(choice) for choice in option.choices))
    return "，".join(bits)


def render_template(
    model: type, *, header: str = "", secret_placeholder: str = ""
) -> str:
    """按模型渲染一份带注释的 TOML 模板（用于 ``init`` 生成可读的初始配置）。"""
    if not isinstance(model, type) or not is_dataclass(model):
        raise ConfigUnsupportedError(
            f"render_template 需要 dataclass 类型，收到 {model!r}"
        )

    lines: list[str] = []
    if header:
        lines.extend(f"# {line}".rstrip() for line in header.splitlines())
        lines.append("")

    def emit(cls: type, prefix: str) -> None:
        scalars: list[list[str]] = []
        tables: list[tuple[str, type]] = []
        for spec in field_specs(cls):
            if spec.nested is not None:
                tables.append((spec.key, spec.nested))
                continue
            note = _constraint_note(spec)
            description = spec.description or ""
            comment = description + (f"（{note}）" if note else "")
            block: list[str] = [f"# {comment}"] if comment else []
            if spec.has_default:
                value = secret_placeholder if spec.secret else spec.default
                block.append(f"{spec.key} = {format_value(value)}")
            else:
                block.append(f"# {spec.key} =  ???  # 必填项")
            scalars.append(block)

        for block in scalars:
            lines.extend(block)

        for key, nested in tables:
            path = f"{prefix}.{key}" if prefix else key
            lines.append("")
            doc = (nested.__doc__ or "").strip().splitlines()
            if doc:
                lines.append(f"# {doc[0]}")
            lines.append(f"[{path}]")
            emit(nested, path)

    emit(model, "")
    text = "\n".join(lines).strip("\n")
    return f"{text}\n" if text else ""
