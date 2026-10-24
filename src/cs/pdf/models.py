"""``cs.pdf`` 的数据模型：只做承载，不含 I/O 副作用。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pages import PageSelection

__all__ = ["STD_PAGE_SIZES", "PDFInfo", "PageInfo", "describe_size"]

# ISO 216 A/B 系列（单位 pt，1pt = 1/72 inch）
STD_PAGE_SIZES: dict[str, tuple[float, float]] = {
    "A0": (2383.9, 3370.4),
    "A1": (1683.8, 2383.9),
    "A2": (1190.6, 1683.8),
    "A3": (841.9, 1190.6),
    "A4": (595.3, 841.9),
    "A5": (419.5, 595.3),
    "A6": (297.6, 419.5),
    "B5": (498.9, 708.7),
    "Letter": (612.0, 792.0),
    "Legal": (612.0, 1008.0),
}


def describe_size(width: float, height: float, *, tolerance: float = 3.0) -> str | None:
    """把页面尺寸识别为标准纸型名（A4 / Letter ...），识别不出返回 ``None``。"""
    short, long = sorted((width, height))
    for name, (sw, sh) in STD_PAGE_SIZES.items():
        if abs(short - sw) <= tolerance and abs(long - sh) <= tolerance:
            return name
    return None


@dataclass(frozen=True, slots=True)
class PageInfo:
    """单页信息。``number`` 为 1 起始页码。"""

    number: int
    width: float
    height: float
    rotation: int
    label: str | None = None

    @property
    def size(self) -> str | None:
        """标准纸型名称（如 ``A4``），无法识别时为 ``None``。"""
        return describe_size(self.width, self.height)

    @property
    def display_size(self) -> tuple[float, float]:
        """考虑 ``/Rotate`` 之后的视觉宽高。"""
        return (
            (self.height, self.width)
            if self.rotation % 180 == 90
            else (self.width, self.height)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "rotation": self.rotation,
            "size": self.size,
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class PDFInfo:
    """打开一次的只读快照。"""

    path: Path
    page_count: int
    encrypted: bool
    version: str | None
    metadata: dict[str, str]
    pages: tuple[PageInfo, ...]
    size_bytes: int

    @property
    def title(self) -> str | None:
        return self.metadata.get("title") or None

    def page_numbers(self, pages: PageSelection | None = None) -> list[int]:
        """把页码表达式校验/解析到本文件的 0 起始索引。"""
        return (pages or PageSelection("all")).resolve(self.page_count)

    def to_dict(self, *, include_pages: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "path": str(self.path),
            "page_count": self.page_count,
            "encrypted": self.encrypted,
            "version": self.version,
            "metadata": dict(self.metadata),
            "size_bytes": self.size_bytes,
        }
        if include_pages:
            data["pages"] = [page.to_dict() for page in self.pages]
        return data

    def __str__(self) -> str:
        title = self.title or self.path.name
        lock = "已加密 " if self.encrypted else ""
        return f"{title} | {self.page_count} 页 | {self.size_bytes / 1024:.0f} KB | {lock}{self.path}"
