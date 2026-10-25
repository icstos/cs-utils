"""文本呈现：``ls`` 风格表格与目录树。

SMB 里中文文件名很常见，所以列宽按 **显示宽度**（东亚宽字符占 2 列）计算，
而不是 ``len()``，否则终端里会歪。
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from datetime import datetime

from .models import SMBEntry, human_size
from .paths import RemotePath

__all__ = ["display_width", "pad", "render_table", "render_tree"]


def display_width(text: str) -> int:
    """终端显示宽度：全角/宽字符记 2 列，其余记 1 列。"""
    return sum(
        2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1 for char in text
    )


def pad(text: str, width: int, *, align: str = "left") -> str:
    """按显示宽度补齐到 ``width`` 列。"""
    gap = max(0, width - display_width(text))
    if align == "right":
        return " " * gap + text
    if align == "center":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return text + " " * gap


def _stamp(mtime: datetime | None) -> str:
    return mtime.astimezone().strftime("%Y-%m-%d %H:%M") if mtime else " " * 16


def render_table(entries: Sequence[SMBEntry], *, with_time: bool = True) -> str:
    """把目录项渲染成对齐的表格（目录在前，``dir`` 标记）。"""
    if not entries:
        return "（空目录）"

    rows: list[tuple[str, str, str]] = []
    for entry in entries:
        rows.append(
            (
                _stamp(entry.mtime) if with_time else "",
                "dir" if entry.is_dir else "file",
                "-" if entry.is_dir else human_size(entry.size),
            )
        )
    size_width = max((len(row[2]) for row in rows), default=4)
    lines: list[str] = []
    for entry, (stamp, kind, size) in zip(entries, rows, strict=True):
        prefix = (
            f"{stamp}  {kind:<4}  {size:>{size_width}}  "
            if with_time
            else (f"{kind:<4}  {size:>{size_width}}  ")
        )
        # 名字是最后一列，补齐只会产生行尾空格，不做对齐
        lines.append(prefix + entry.name)
    return "\n".join(lines)


def render_tree(
    client: object,
    path: str = "",
    *,
    max_depth: int = 3,
    include_hidden: bool = False,
    limit: int = 500,
) -> str:
    """递归渲染目录树（``client`` 需提供 ``list_dir`` 与 ``unc``）。"""
    base: RemotePath = client.resolve(path)  # type: ignore[attr-defined]
    header = f"{client.unc}/{base.posix}" if base.posix else client.unc  # type: ignore[attr-defined]
    lines = [header]
    emitted = 0

    def walk(current: RemotePath, prefix: str, depth: int) -> None:
        nonlocal emitted
        if depth > max_depth or emitted >= limit:
            return
        try:
            children = client.list_dir(current, include_hidden=include_hidden)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - 树渲染容错：不因单个子目录失败中断
            lines.append(f"{prefix}└── <{type(exc).__name__}: {exc}>")
            return
        for index, entry in enumerate(children):
            if emitted >= limit:
                lines.append(f"{prefix}└── …（超过 {limit} 项，已截断）")
                return
            last = index == len(children) - 1
            branch = "└── " if last else "├── "
            suffix = "/" if entry.is_dir else f"  {human_size(entry.size)}"
            lines.append(f"{prefix}{branch}{entry.name}{suffix}")
            emitted += 1
            if entry.is_dir:
                walk(
                    RemotePath(entry.path),
                    prefix + ("    " if last else "│   "),
                    depth + 1,
                )

    walk(base, "", 1)
    return "\n".join(lines)
