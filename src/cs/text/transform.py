"""文本变换：类型归一、空白整理、宽度计算、截断/折行/对齐、打码与命名风格转换。

宽度按「终端显示列」计算：东亚宽字符（``W``/``F``）占 2 列，组合符号与零宽
字符占 0 列。所有函数都接受 ``str``，入参为 ``bytes`` 时请先用 :func:`to_str`。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .errors import TransformError

__all__ = [
    "ALIGNS",
    "CASES",
    "camel",
    "char_width",
    "collapse_whitespace",
    "constant",
    "indent",
    "kebab",
    "mask",
    "normalize",
    "one_line",
    "pad",
    "pascal",
    "slugify",
    "snake",
    "split_words",
    "strip_control",
    "strip_lines",
    "title_case",
    "to_halfwidth",
    "to_str",
    "truncate",
    "width",
    "wrap",
]

ALIGNS = ("left", "right", "center")
CASES = ("snake", "camel", "pascal", "kebab", "constant", "title")

#: 零宽/软连字符，肉眼不可见但会让长度计算失真
_INVISIBLE = frozenset("\u200b\u200c\u200d\ufeff\u00ad")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_WORD_SEPARATOR = re.compile(r"[^0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff]+")
_CONTROL_EXEMPT = frozenset("\n\r\t")


def to_str(
    value: Any,
    encoding: str = "utf-8",
    errors: str = "strict",
    default: str | None = None,
) -> str | None:
    """把任意值转成 ``str``；``None`` 返回 ``default``。

    >>> to_str(b"hi"), to_str(None), to_str(12)
    ('hi', None, '12')
    """
    match value:
        case None:
            return default
        case str():
            return value
        case bytes() | bytearray() | memoryview():
            return bytes(value).decode(encoding, errors)
        case _:
            return str(value)


def char_width(char: str) -> int:
    """单个字符的显示列宽：0（组合/零宽/控制）、1 或 2。"""
    if not char:
        return 0
    if unicodedata.combining(char) or char in _INVISIBLE:
        return 0
    if not char.isprintable():
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def width(text: str) -> int:
    """字符串的显示列宽（中文按 2 列计）。"""
    return sum(char_width(char) for char in text)


def truncate(
    text: str,
    limit: int,
    *,
    suffix: str = "…",
    word_boundary: bool = False,
) -> str:
    """按显示列宽截断，超长时补 ``suffix``（保留 ``suffix`` 自身宽度）。"""
    if limit <= 0:
        return ""
    if width(text) <= limit:
        return text
    area = limit - width(suffix)
    if area <= 0:
        return _cut(suffix, limit)
    head = _cut(text, area)
    if word_boundary:
        index = max(head.rfind(" "), head.rfind("\t"))
        if index > 0:
            head = head[:index]
    return head.rstrip() + suffix


def wrap(text: str, limit: int, *, word_boundary: bool = True) -> list[str]:
    """按显示列宽折行；``word_boundary=True`` 时优先在空白处断开。

    段落内不会插入额外空行，``text.splitlines()`` 的每一行独立折行。
    """
    if limit <= 0:
        raise TransformError("wrap 的 limit 必须是正整数")
    lines: list[str] = []
    for raw in text.splitlines() or [""]:
        current = ""
        current_width = 0
        for char in raw:
            char_w = char_width(char)
            if current_width + char_w > limit and current:
                if char.isspace():
                    lines.append(current)
                    current, current_width = "", 0
                    continue
                if word_boundary:
                    index = max(current.rfind(" "), current.rfind("\t"))
                    if index > 0:
                        lines.append(current[:index])
                        current = current[index + 1 :]
                        current_width = width(current)
                if current_width + char_w > limit:
                    lines.append(current)
                    current, current_width = "", 0
            current += char
            current_width += char_w
        lines.append(current)
    return lines


def pad(text: str, limit: int, *, align: str = "left", fill: str = " ") -> str:
    """把文本补齐到 ``limit`` 列宽；``align`` 取 left/right/center。"""
    if align not in ALIGNS:
        raise TransformError(f"align 只能是 {ALIGNS}，收到 {align!r}")
    if not fill or width(fill) < 1:
        raise TransformError("fill 必须是至少 1 列宽的可见字符")
    gap = limit - width(text)
    if gap <= 0:
        return text
    if align == "right":
        return fill * gap + text
    if align == "center":
        left = gap // 2
        return fill * left + text + fill * (gap - left)
    return text + fill * gap


def mask(text: str, *, keep: int = 4, tail: int = 0, char: str = "*") -> str:
    """保留首 ``keep`` 与尾 ``tail`` 个字符，中间替换为 ``char``。

    文本长度不足 ``keep + tail`` 时整体打码，避免「只打一位」形同明文。
    """
    if not text:
        return ""
    keep = max(keep, 0)
    tail = max(tail, 0)
    total = len(text)
    if total <= keep + tail:
        return char * total
    middle = char * (total - keep - tail)
    return text[:keep] + middle + (text[-tail:] if tail else "")


def normalize(
    text: str,
    form: str = "NFKC",
    *,
    collapse: bool = False,
    strip: bool = False,
) -> str:
    """Unicode 归一化（默认 NFKC），可选折叠空白与去首尾空白。"""
    if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise TransformError(f"未知归一化形式 {form!r}")
    value = unicodedata.normalize(form, text)
    if collapse:
        value = collapse_whitespace(value)
    return value.strip() if strip else value


def collapse_whitespace(text: str, *, max_blank_lines: int = 1) -> str:
    """把行内连续空白压成一个空格，并限制连续空行数量。"""
    if max_blank_lines < 0:
        raise TransformError("max_blank_lines 不能为负")
    value = re.sub(r"[^\S\n]+", " ", text)
    value = re.sub(r" *\n *", "\n", value)
    return re.sub(rf"\n{{{max_blank_lines + 2},}}", "\n" * (max_blank_lines + 1), value)


def one_line(text: str) -> str:
    """把所有空白（含换行）压成单空格并去掉首尾空白。"""
    return re.sub(r"\s+", " ", text).strip()


def strip_control(text: str, *, keep: str = "") -> str:
    """删除控制字符与零宽字符；``keep`` 中的字符例外（如 ``"\\n"``）。"""
    allowed = _CONTROL_EXEMPT | frozenset(keep)
    return "".join(
        char
        for char in text
        if char in allowed
        or (char not in _INVISIBLE and unicodedata.category(char) not in {"Cc", "Cf"})
    )


def to_halfwidth(text: str) -> str:
    """全角 ASCII（``！＂＃``…）与全角空格转半角，中文标点等保持不变。"""
    out: list[str] = []
    for char in text:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            out.append(" ")
        else:
            out.append(char)
    return "".join(out)


def split_words(text: str) -> list[str]:
    """按驼峰、下划线、连字符与标点切词。

    >>> split_words("HTTPServer_error-v2")
    ['HTTP', 'Server', 'error', 'v2']
    """
    cleaned = _CAMEL_BOUNDARY.sub(" ", unicodedata.normalize("NFKC", text))
    return [part for part in _WORD_SEPARATOR.split(cleaned) if part]


def snake(text: str) -> str:
    """``userID`` → ``user_id``。"""
    return "_".join(word.lower() for word in split_words(text))


def kebab(text: str) -> str:
    """``user_id`` → ``user-id``。"""
    return "-".join(word.lower() for word in split_words(text))


def constant(text: str) -> str:
    """``user_id`` → ``USER_ID``。"""
    return "_".join(word.upper() for word in split_words(text))


def camel(text: str) -> str:
    """``user_id`` → ``userId``。"""
    words = split_words(text)
    if not words:
        return ""
    head = words[0][0].lower() + words[0][1:]
    return head + "".join(word.capitalize() for word in words[1:])


def pascal(text: str) -> str:
    """``user_id`` → ``UserId``。"""
    return "".join(word.capitalize() for word in split_words(text))


def title_case(text: str) -> str:
    """``hello world`` → ``Hello World``（保留原有空白与标点）。"""
    return " ".join(word.capitalize() for word in split_words(text))


def slugify(text: str, *, allow_unicode: bool = False, separator: str = "-") -> str:
    """生成 URL 友好的 slug；``allow_unicode=True`` 时保留中文等非 ASCII 字母。"""
    if not separator:
        raise TransformError("separator 不能为空")
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    out: list[str] = []
    for char in stripped:
        if allow_unicode:
            keep = char.isalnum() or char in "_-"
        else:
            keep = char.isascii() and char.isalnum()
        out.append(char if keep else separator)
    value = re.sub(f"{re.escape(separator)}+", separator, "".join(out))
    return value.strip(separator).lower()


def indent(text: str, prefix: str = "    ", *, skip_first: bool = False) -> str:
    """给每一行加前缀；``skip_first=True`` 时首行不加。"""
    lines = text.splitlines(keepends=True)
    return "".join(
        (line if skip_first and index == 0 else prefix + line)
        for index, line in enumerate(lines)
    )


def strip_lines(text: str) -> str:
    """逐行去首尾空白（保留换行结构）。"""
    return "\n".join(line.strip() for line in text.splitlines())


def _cut(text: str, limit: int) -> str:
    """按显示列宽硬切，不补后缀。"""
    if limit <= 0:
        return ""
    total = 0
    out: list[str] = []
    for char in text:
        char_w = char_width(char)
        if total + char_w > limit:
            break
        out.append(char)
        total += char_w
    return "".join(out)
