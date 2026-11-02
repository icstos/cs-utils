"""文本比较：相似度、差异片段、统一差异、编辑距离与最相近候选。

比较前可统一大小写、剔除忽略字符（如空格），三者都只影响比较口径，
``DiffResult.a`` / ``.b`` 仍保存原文。
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Sequence

from .errors import TextError
from .models import Chunk, DiffResult, Granularity, Match, OpTag

__all__ = [
    "best_matches",
    "diff",
    "distance",
    "quick_ratio",
    "ratio",
    "similar",
    "tokenize",
    "unified",
]


def _ignore_set(ignore: str | Iterable[str] | None) -> tuple[str, ...]:
    if not ignore:
        return ()
    if isinstance(ignore, str):
        return (ignore,) if len(ignore) == 1 else tuple(dict.fromkeys(ignore))
    return tuple(dict.fromkeys(ignore))


def _as_granularity(value: Granularity | str) -> Granularity:
    if isinstance(value, Granularity):
        return value
    try:
        return Granularity(value.strip().lower())
    except (AttributeError, ValueError):
        allowed = " / ".join(item.value for item in Granularity)
        raise TextError(f"未知比较粒度 {value!r}，可选：{allowed}") from None


def _prepare(text: str, ignore: tuple[str, ...], case_sensitive: bool) -> str:
    value = text if case_sensitive else text.casefold()
    if not ignore:
        return value
    drop = set(ignore)
    return "".join(char for char in value if char not in drop)


def tokenize(text: str, granularity: Granularity | str = Granularity.CHAR) -> list[str]:
    """按比较粒度切分；``word`` 保留空白与标点，便于还原原位置。"""
    kind = _as_granularity(granularity)
    match kind:
        case Granularity.CHAR:
            return list(text)
        case Granularity.LINE:
            return text.splitlines(keepends=True)
        case Granularity.WORD:
            tokens: list[str] = []
            buffer: list[str] = []
            for char in text:
                if char.isalnum() or char == "_":
                    buffer.append(char)
                else:
                    if buffer:
                        tokens.append("".join(buffer))
                        buffer.clear()
                    tokens.append(char)
            if buffer:
                tokens.append("".join(buffer))
            return tokens


def ratio(
    a: str,
    b: str,
    *,
    ignore: str | Iterable[str] | None = None,
    case_sensitive: bool = True,
    quick: bool = False,
) -> float:
    """相似度 0.0~1.0；``quick=True`` 用 ``quick_ratio`` 取上界，适合粗筛。"""
    skip = _ignore_set(ignore)
    left = _prepare(a, skip, case_sensitive)
    right = _prepare(b, skip, case_sensitive)
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    return matcher.quick_ratio() if quick else matcher.ratio()


def quick_ratio(
    a: str,
    b: str,
    *,
    ignore: str | Iterable[str] | None = None,
    case_sensitive: bool = True,
) -> float:
    """:func:`ratio` 的 ``quick=True`` 简写。"""
    return ratio(a, b, ignore=ignore, case_sensitive=case_sensitive, quick=True)


def similar(
    a: str,
    b: str,
    *,
    threshold: float = 0.6,
    ignore: str | Iterable[str] | None = None,
    case_sensitive: bool = True,
    quick: bool = False,
) -> bool:
    """相似度是否达到阈值。"""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold 需在 0~1 之间，收到 {threshold}")
    return (
        ratio(a, b, ignore=ignore, case_sensitive=case_sensitive, quick=quick)
        >= threshold
    )


def diff(
    a: str,
    b: str,
    *,
    ignore: str | Iterable[str] | None = None,
    case_sensitive: bool = True,
    granularity: Granularity | str = Granularity.CHAR,
) -> DiffResult:
    """逐片段比较，返回相似度与增删片段（含定位信息）。"""
    kind = _as_granularity(granularity)
    skip = _ignore_set(ignore)
    left = tokenize(_prepare(a, skip, case_sensitive), kind)
    right = tokenize(_prepare(b, skip, case_sensitive), kind)
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)

    chunks: list[Chunk] = []
    added = removed = 0
    for tag, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        chunk = Chunk(
            tag=OpTag(tag),
            a_start=a_start,
            a_end=a_end,
            b_start=b_start,
            b_end=b_end,
            a_text="".join(left[a_start:a_end]),
            b_text="".join(right[b_start:b_end]),
        )
        chunks.append(chunk)
        if tag != OpTag.EQUAL.value:
            added += b_end - b_start
            removed += a_end - a_start

    return DiffResult(
        ratio=matcher.ratio(),
        chunks=tuple(chunks),
        added=added,
        removed=removed,
        granularity=kind,
        ignore=skip,
        case_sensitive=case_sensitive,
        a=a,
        b=b,
    )


def unified(
    a: str,
    b: str,
    *,
    fromfile: str = "a",
    tofile: str = "b",
    context: int = 3,
) -> str:
    """按行渲染统一差异（unified diff），无差异时返回空串。"""
    lines = difflib.unified_diff(
        a.splitlines(),
        b.splitlines(),
        fromfile=fromfile,
        tofile=tofile,
        n=context,
        lineterm="",
    )
    return "\n".join(lines)


def distance(
    a: str,
    b: str,
    *,
    ignore: str | Iterable[str] | None = None,
    case_sensitive: bool = True,
    limit: int | None = None,
) -> int:
    """Levenshtein 编辑距离；给 ``limit`` 时超过即提前返回 ``limit + 1``。"""
    skip = _ignore_set(ignore)
    left = _prepare(a, skip, case_sensitive)
    right = _prepare(b, skip, case_sensitive)
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return len(left) if limit is None or len(left) <= limit else limit + 1

    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + cost,
                )
            )
        if limit is not None and min(current) > limit:
            return limit + 1
        previous = current
    result = previous[-1]
    return result if limit is None or result <= limit else limit + 1


def best_matches(
    query: str,
    candidates: Sequence[str] | Iterable[str],
    *,
    limit: int = 5,
    cutoff: float = 0.0,
    ignore: str | Iterable[str] | None = None,
    case_sensitive: bool = True,
) -> tuple[Match, ...]:
    """按相似度从高到低挑选候选，``cutoff`` 以下丢弃。"""
    if limit <= 0:
        return ()
    scored = [
        Match(
            text=item,
            score=ratio(query, item, ignore=ignore, case_sensitive=case_sensitive),
        )
        for item in candidates
    ]
    hits = [match for match in scored if match.score >= cutoff]
    hits.sort(key=lambda match: match.score, reverse=True)
    return tuple(hits[:limit])
