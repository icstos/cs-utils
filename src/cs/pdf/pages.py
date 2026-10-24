"""页码范围解析（1 起始、支持负索引与步长）。

表达式语法
----------
===============  ==================================================
``all`` / ``*``  全部页面
``odd`` / ``even``  奇数页 / 偶数页（按 1 起始的页码）
``last``         最后一页
``3``            第 3 页
``-1``           最后一页；``-3`` 表示最后 3 页（Python 切片语义）
``2-5``          第 2 到第 5 页（含两端）
``3-``           第 3 页到末页
``-``            非法
``1-10:2``       第 1 到 10 页，步长 2
``1-3,7``        多个片段，用 ``,`` / ``;`` / 空白分隔
===============  ==================================================

列表顺序会被保留、重复项会被剔除，因此 ``pages="3,1,2"`` 同时也能表达重排。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Self

from .errors import PageSpecError

__all__ = ["PageSelection"]

_SEPARATORS = re.compile(r"[,;\s]+")
_DIGITS = re.compile(r"\d+")
_ALL_TOKENS = frozenset({"all", "*"})

type PageSpec = str | int | Sequence[int]


def _single(page: int, total: int) -> range:
    if page > total:
        raise PageSpecError(f"第 {page} 页超出总页数 {total}")
    return range(page - 1, page)


@dataclass(frozen=True, slots=True)
class PageSelection:
    """不可变的页码范围，可反复针对不同文档解析。"""

    spec: str = "all"

    def __post_init__(self) -> None:
        if not isinstance(self.spec, str):
            raise PageSpecError(
                f"页码范围必须是字符串，收到 {type(self.spec).__name__}"
            )
        tokens = self.tokens
        if not tokens:
            raise PageSpecError("页码范围不能为空")
        if len(tokens) > 1 and _ALL_TOKENS.intersection(tokens):
            raise PageSpecError("'all'/'*' 不能与其他片段混用")
        for token in tokens:  # 只校验语法，页数在 resolve() 时校验
            self._compile(token)

    # ---------------------------------------------------------------- 构造
    @classmethod
    def parse(cls, value: PageSpec | PageSelection | None) -> Self:
        """把 ``None`` / 单个页码 / 页码序列 / 字符串统一转换为选择器。"""
        match value:
            case PageSelection():
                return value
            case None:
                return cls("all")
            case int():
                return cls(str(value))
            case str():
                return cls(value.strip() or "all")
            case _:
                return cls.from_pages(value)

    @classmethod
    def from_pages(cls, pages: Iterable[int]) -> Self:
        """用 1 起始的页码序列构造（负数表示倒数）。"""
        numbers = [int(page) for page in pages]
        if not numbers:
            raise PageSpecError("页码序列不能为空")
        if any(number == 0 for number in numbers):
            raise PageSpecError("页码从 1 开始，不存在第 0 页")
        return cls(",".join(str(number) for number in numbers))

    # ---------------------------------------------------------------- 解析
    @property
    def tokens(self) -> tuple[str, ...]:
        """按书写顺序切分出的片段（已转小写、已去空）。"""
        return tuple(
            token
            for token in (part.strip().lower() for part in _SEPARATORS.split(self.spec))
            if token
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self.tokens)

    def __str__(self) -> str:
        return self.spec

    def chunks(self, total: int) -> list[list[int]]:
        """按片段切分：``"1-3,5"`` → ``[[0, 1, 2], [4]]``（0 起始）。"""
        if total < 1:
            raise PageSpecError("文档没有可操作的页面")
        return [list(self._compile(token)(total)) for token in self.tokens]

    def resolve(self, total: int) -> list[int]:
        """解析为 0 起始的页索引，保持书写顺序并去重。"""
        seen: set[int] = set()
        result: list[int] = []
        for chunk in self.chunks(total):
            for index in chunk:
                if index not in seen:
                    seen.add(index)
                    result.append(index)
        if not result:
            raise PageSpecError(f"页码范围 {self.spec!r} 未匹配到任何页面")
        return result

    # ---------------------------------------------------------------- 内部
    def _compile(self, token: str) -> Callable[[int], range]:
        """把单个片段编译成 ``total -> range`` 的函数（语法错误在编译期抛出）。"""
        match token:
            case "all" | "*":
                return lambda total: range(total)
            case "odd":
                return lambda total: range(0, total, 2)
            case "even":
                return lambda total: range(1, total, 2)
            case "last":
                return lambda total: range(max(total - 1, 0), total)

        base, _, raw_step = token.partition(":")
        step = 1
        if raw_step:
            if not _DIGITS.fullmatch(raw_step) or int(raw_step) < 1:
                raise PageSpecError(f"步长必须是正整数: {token!r}")
            step = int(raw_step)
        if not base:
            raise PageSpecError(f"缺少页码: {token!r}")

        if base.startswith("-"):  # 负索引：-3 → 最后 3 页
            digits = base[1:]
            if not _DIGITS.fullmatch(digits) or int(digits) < 1:
                raise PageSpecError(f"非法页码: {token!r}")
            count = int(digits)
            return lambda total, count=count: range(max(total - count, 0), total, step)

        if "-" in base:  # 区间：2-5 / 3- / 1-10:2
            head, _, tail = base.partition("-")
            if not head and not tail:
                raise PageSpecError(f"非法页码区间: {token!r}")
            if head and not _DIGITS.fullmatch(head):
                raise PageSpecError(f"非法页码区间: {token!r}")
            if tail and not _DIGITS.fullmatch(tail):
                raise PageSpecError(f"非法页码区间: {token!r}")
            start = int(head) if head else 1
            end = int(tail) if tail else None
            if start < 1 or (end is not None and end < 1):
                raise PageSpecError(f"页码从 1 开始: {token!r}")
            if end is not None and end < start:
                raise PageSpecError(f"非法页码区间: {token!r}（起始页大于结束页）")

            def _range(
                total: int, start: int = start, end: int | None = end, step: int = step
            ) -> range:
                if start > total:
                    raise PageSpecError(f"起始页 {start} 超出总页数 {total}")
                stop = total if end is None else min(end, total)
                if stop < start:
                    raise PageSpecError(
                        f"非法页码区间: {start}-{end}（起始页大于结束页）"
                    )
                return range(start - 1, stop, step)

            return _range

        if _DIGITS.fullmatch(base):
            page = int(base)
            if page < 1:
                raise PageSpecError(f"页码从 1 开始: {token!r}")
            return lambda total, page=page: _single(page, total)

        raise PageSpecError(f"无法识别的页码范围: {token!r}")
