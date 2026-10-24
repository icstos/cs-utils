"""PDF 常用操作门面。

设计原则
--------
* **无状态**：不持有打开的文档，所有操作都是"输入路径 → 输出路径"的纯函数式调用，
  天然线程/进程安全，也便于被 MCP、CLI、Web 等任意宿主复用。
* **安全落盘**：全部写出走临时文件 + ``os.replace``，失败不会留下半个 PDF。
* **显式覆盖**：目标已存在且 ``overwrite=False`` 时抛 :class:`OutputExistsError`。
* **懒加载重依赖**：``pypdf`` 是必需依赖；渲染（PyMuPDF）等按需导入。

典型用法::

    from cs.pdf import PDF

    PDF.info("in.pdf")
    PDF.rotate("in.pdf", "rotated.pdf", 90)
    PDF.split("in.pdf", "out/", every=10)
    PDF.merge(["a.pdf", "b.pdf"], "merged.pdf")
    PDF.extract("in.pdf", "part.pdf", "1-3,7")

    with PDF.open("in.pdf") as doc:
        print(doc.page_count, doc.info().title)
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from functools import cache
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar, Self

from pypdf import PageObject, PdfReader, PdfWriter
from pypdf.generic import NameObject, RectangleObject

from .errors import (
    MissingDependencyError,
    OutputExistsError,
    PageSpecError,
    PDFError,
    PDFOpenError,
    PDFPasswordError,
)
from .models import PageInfo, PDFInfo
from .pages import PageSelection, PageSpec
from .storage import (
    PathLike,
    as_path,
    ensure_dir,
    ensure_input,
    open_reader,
    require,
    write_pdf,
)

__all__ = ["PDF", "ImageFormat", "PDFHandle"]

_ROTATION_STEP = 90
_MAX_DPI = 1200


class ImageFormat:
    """渲染图片格式白名单（避免把任意字符串拼进文件名）。"""

    PNG = "png"
    JPEG = "jpg"
    WEBP = "webp"

    _ALIASES: ClassVar[dict[str, str]] = {
        "jpeg": JPEG,
        "jpg": JPEG,
        "png": PNG,
        "webp": WEBP,
    }

    @classmethod
    def normalize(cls, value: str) -> str:
        key = str(value).lower().lstrip(".")
        try:
            return cls._ALIASES[key]
        except KeyError:
            raise ValueError(
                f"不支持的图片格式 {value!r}，可选：{', '.join(sorted(set(cls._ALIASES)))}"
            ) from None


# ---------------------------------------------------------------- 内部工具


def _build_info(reader: PdfReader, path: Path) -> PDFInfo:
    raw_metadata = reader.metadata or {}
    metadata = {
        str(key).lstrip("/").lower(): str(value)
        for key, value in raw_metadata.items()
        if value is not None
    }
    labels: list[str | None]
    try:
        labels = list(reader.page_labels)
    except (KeyError, TypeError, ValueError):  # 页码标签是可选特性，缺失时不影响其它信息
        labels = []

    pages = tuple(
        PageInfo(
            number=index,
            width=float(page.mediabox.width),
            height=float(page.mediabox.height),
            rotation=int(page.get("/Rotate", 0) or 0) % 360,
            label=labels[index - 1] if index <= len(labels) else None,
        )
        for index, page in enumerate(reader.pages, start=1)
    )
    header = getattr(reader, "pdf_header", None)
    return PDFInfo(
        path=path,
        page_count=len(reader.pages),
        encrypted=bool(reader.is_encrypted),
        version=header.removeprefix("%PDF-") if header else None,
        metadata=metadata,
        pages=pages,
        size_bytes=path.stat().st_size,
    )


def _select(pages: PageSpec | PageSelection | None, total: int) -> list[int]:
    return PageSelection.parse(pages).resolve(total)


def _add_pages(writer: PdfWriter, reader: PdfReader, indices: Iterable[int]) -> None:
    for index in indices:
        writer.add_page(reader.pages[index])


def _rotate_page(page: PageObject, angle: int, *, relative: bool) -> None:
    if angle % _ROTATION_STEP:
        raise PDFError(f"旋转角度必须是 {_ROTATION_STEP} 的整数倍，收到 {angle}")
    if relative:
        page.rotate(angle)  # pypdf 内部按顺时针累加
    else:  # 绝对角度：直接设定，覆盖原有 /Rotate
        page.rotation = angle % 360


def _crop_page(page: PageObject, box: Sequence[float]) -> None:
    left, top, right, bottom = (float(value) for value in box)
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise PDFError(
            f"裁剪框 {tuple(box)} 超出页面范围 {width:.1f}x{height:.1f}（左上角为原点，单位 pt）"
        )
    page[NameObject("/CropBox")] = RectangleObject(
        (left, height - bottom, right, height - top)
    )


def _clone(reader: PdfReader) -> PdfWriter:
    return PdfWriter(clone_from=reader)


@cache
def _render_engine() -> Any:
    """加载 PyMuPDF（新版本包名为 ``pymupdf``，``fitz`` 兼容旧版本）。"""
    try:
        return require("pymupdf", extra="render", purpose="PDF 转图片")
    except MissingDependencyError:
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return import_module("fitz")
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "pymupdf", extra="render", purpose="PDF 转图片"
        ) from exc


# ---------------------------------------------------------------- 文档句柄


class PDFHandle:
    """一次打开、多次读取的轻量句柄，支持 ``with`` 语句。

    只读：需要修改时请调用 :class:`PDF` 的静态方法，输出到新文件更可预期。
    """

    __slots__ = ("_password", "_path", "_reader")

    def __init__(self, path: PathLike, password: str | None = None) -> None:
        self._path = ensure_input(path)
        self._password = password
        self._reader: PdfReader | None = None

    # -- 生命周期 ------------------------------------------------------
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._reader = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def reader(self) -> PdfReader:
        if self._reader is None:
            self._reader = open_reader(self._path, self._password)
        return self._reader

    # -- 读取能力 ------------------------------------------------------
    @property
    def page_count(self) -> int:
        return len(self.reader.pages)

    def info(self) -> PDFInfo:
        return _build_info(self.reader, self._path)

    def text(
        self, pages: PageSpec | PageSelection | None = "all", *, layout: bool = False
    ) -> str:
        return PDF.text(self._path, pages, layout=layout, password=self._password)

    def render(
        self,
        out_dir: PathLike,
        *,
        dpi: int = 200,
        pages: PageSpec | PageSelection | None = "all",
        fmt: str = ImageFormat.PNG,
    ) -> list[Path]:
        return PDF.render(
            self._path, out_dir, dpi=dpi, pages=pages, fmt=fmt, password=self._password
        )


# ---------------------------------------------------------------- 门面


class PDF:
    """PDF 常用操作集合（全部为静态方法，调用即用）。"""

    # ------------------------------------------------------------ 打开
    @staticmethod
    def open(path: PathLike, password: str | None = None) -> PDFHandle:
        """打开文档并返回只读句柄，建议配合 ``with`` 使用。"""
        return PDFHandle(path, password)

    @staticmethod
    def info(source: PathLike, *, password: str | None = None) -> PDFInfo:
        """读取页数、页面尺寸、旋转、元数据、文件大小等快照信息。"""
        path = ensure_input(source)
        return _build_info(open_reader(path, password), path)

    @staticmethod
    def page_count(source: PathLike, *, password: str | None = None) -> int:
        return len(open_reader(source, password).pages)

    # ------------------------------------------------------------ 文本
    @staticmethod
    def text(
        source: PathLike,
        pages: PageSpec | PageSelection | None = "all",
        *,
        layout: bool = False,
        password: str | None = None,
    ) -> str:
        """抽取文本，页与页之间用换页符 ``\\f`` 分隔。"""
        pages_data = PDF.text_pages(source, pages, layout=layout, password=password)
        return "\f".join(page["text"] for page in pages_data)

    @staticmethod
    def text_pages(
        source: PathLike,
        pages: PageSpec | PageSelection | None = "all",
        *,
        layout: bool = False,
        password: str | None = None,
    ) -> list[dict[str, Any]]:
        """逐页抽取文本，返回 ``[{"number": 1, "text": "..."}, ...]``。"""
        reader = open_reader(source, password)
        indices = _select(pages, len(reader.pages))
        mode = "layout" if layout else "plain"
        result: list[dict[str, Any]] = []
        for index in indices:
            text = reader.pages[index].extract_text(extraction_mode=mode) or ""
            result.append({"number": index + 1, "text": text, "chars": len(text)})
        return result

    # ------------------------------------------------------------ 渲染
    @staticmethod
    def render(
        source: PathLike,
        out_dir: PathLike,
        *,
        dpi: int = 200,
        pages: PageSpec | PageSelection | None = "all",
        fmt: str = ImageFormat.PNG,
        password: str | None = None,
        overwrite: bool = True,
    ) -> list[Path]:
        """把页面渲染为图片（需要 ``pymupdf``），返回生成的文件列表。"""
        if not 36 <= dpi <= _MAX_DPI:
            raise ValueError(f"dpi 需在 36-{_MAX_DPI} 之间，收到 {dpi}")
        suffix = ImageFormat.normalize(fmt)
        engine = _render_engine()

        path = ensure_input(source)
        directory = ensure_dir(out_dir)
        try:
            document = engine.open(str(path))
        except Exception as exc:
            raise PDFOpenError(path, f"{type(exc).__name__}: {exc}") from exc

        try:
            if document.needs_pass and not document.authenticate(password or ""):
                raise PDFPasswordError(path, "文件已加密，请提供正确的 password")
            indices = _select(pages, document.page_count)
            targets = [
                directory / f"{path.stem}_{index + 1:04d}.{suffix}" for index in indices
            ]
            if not overwrite:  # 先整体校验，避免只写出一半图片
                existing = next((item for item in targets if item.exists()), None)
                if existing is not None:
                    raise OutputExistsError(existing)
            for index, target in zip(indices, targets, strict=True):
                pixmap = document.load_page(index).get_pixmap(dpi=dpi, alpha=False)
                pixmap.save(str(target))
            return targets
        finally:
            document.close()

    # ------------------------------------------------------------ 页面级操作
    @staticmethod
    def extract(
        source: PathLike,
        output: PathLike,
        pages: PageSpec | PageSelection,
        *,
        password: str | None = None,
        overwrite: bool = True,
    ) -> Path:
        """抽取指定页面另存为新文件（顺序按表达式）。"""
        reader = open_reader(source, password)
        indices = _select(pages, len(reader.pages))
        writer = PdfWriter()
        _add_pages(writer, reader, indices)
        return write_pdf(writer, as_path(output), overwrite=overwrite)

    @staticmethod
    def delete(
        source: PathLike,
        output: PathLike,
        pages: PageSpec | PageSelection,
        *,
        password: str | None = None,
        overwrite: bool = True,
    ) -> Path:
        """删除指定页面，保留其余页面（至少保留一页）。"""
        reader = open_reader(source, password)
        total = len(reader.pages)
        removed = set(_select(pages, total))
        remaining = [index for index in range(total) if index not in removed]
        if not remaining:
            raise PageSpecError("不能删除全部页面")
        writer = PdfWriter()
        _add_pages(writer, reader, remaining)
        return write_pdf(writer, as_path(output), overwrite=overwrite)

    @staticmethod
    def reorder(
        source: PathLike,
        output: PathLike,
        order: PageSpec | PageSelection | Sequence[int],
        *,
        password: str | None = None,
        overwrite: bool = True,
    ) -> Path:
        """按给定顺序重排全部页面，例如 ``"3,1,2"``；必须完整覆盖所有页。"""
        reader = open_reader(source, password)
        total = len(reader.pages)
        indices = _select(order, total)
        if sorted(indices) != list(range(total)):
            raise PageSpecError(
                f"重排顺序必须不重不漏地覆盖全部 {total} 页，收到 {len(indices)} 项"
            )
        writer = PdfWriter()
        _add_pages(writer, reader, indices)
        return write_pdf(writer, as_path(output), overwrite=overwrite)

    @staticmethod
    def insert(
        source: PathLike,
        output: PathLike,
        other: PathLike,
        at: int,
        *,
        pages: PageSpec | PageSelection | None = "all",
        password: str | None = None,
        other_password: str | None = None,
        overwrite: bool = True,
    ) -> Path:
        """把 ``other`` 的页面插入到 ``source`` 的第 ``at`` 页之前（1 起始，``at=总页数+1`` 表示追加）。"""
        reader = open_reader(source, password)
        donor = open_reader(other, other_password)
        total = len(reader.pages)
        if not 1 <= at <= total + 1:
            raise PageSpecError(f"插入位置需在 1-{total + 1} 之间，收到 {at}")
        donor_indices = _select(pages, len(donor.pages))

        writer = PdfWriter()
        _add_pages(writer, reader, range(at - 1))
        _add_pages(writer, donor, donor_indices)
        _add_pages(writer, reader, range(at - 1, total))
        return write_pdf(writer, as_path(output), overwrite=overwrite)

    @staticmethod
    def split(
        source: PathLike,
        out_dir: PathLike,
        *,
        pages: PageSpec | PageSelection | None = None,
        every: int | None = None,
        prefix: str | None = None,
        password: str | None = None,
        overwrite: bool = True,
    ) -> list[Path]:
        """拆分文档。

        * 都不传：每页一个文件；
        * ``every=n``：每 n 页一个文件；
        * ``pages="1-3,5"``：每个片段一个文件。

        文件名为 ``<prefix|stem>_<起页>[-<止页>].pdf``。
        """
        if pages is not None and every is not None:
            raise ValueError("pages 与 every 只能二选一")
        if every is not None and every < 1:
            raise ValueError("every 必须是正整数")

        reader = open_reader(source, password)
        total = len(reader.pages)
        if pages is not None:
            groups: list[list[int]] = []
            seen: set[int] = set()
            for chunk in PageSelection.parse(pages).chunks(total):
                fresh = [index for index in chunk if index not in seen]
                seen.update(fresh)
                if fresh:
                    groups.append(fresh)
        else:
            groups = [
                list(group) for group in itertools.batched(range(total), every or 1)
            ]

        path = ensure_input(source)
        directory = ensure_dir(out_dir)
        base = prefix or path.stem
        outputs: list[Path] = []
        for group in groups:
            writer = PdfWriter()
            _add_pages(writer, reader, group)
            label = (
                f"{group[0] + 1:04d}"
                if len(group) == 1
                else f"{group[0] + 1:04d}-{group[-1] + 1:04d}"
            )
            outputs.append(
                write_pdf(
                    writer, directory / f"{base}_{label}.pdf", overwrite=overwrite
                )
            )
        return outputs

    # ------------------------------------------------------------ 合并
    @staticmethod
    def merge(
        sources: Sequence[PathLike],
        output: PathLike,
        *,
        outline: bool = False,
        overwrite: bool = True,
    ) -> Path:
        """按顺序合并多个 PDF；``outline=True`` 时为每个源文件生成书签。"""
        if not sources:
            raise ValueError("sources 不能为空")
        writer = PdfWriter()
        for item in sources:
            reader = open_reader(item)
            if outline:
                writer.append(reader, outline_item=as_path(item).stem)
            else:
                writer.append(reader, import_outline=False)
        return write_pdf(writer, as_path(output), overwrite=overwrite)

    # ------------------------------------------------------------ 变形
    @staticmethod
    def rotate(
        source: PathLike,
        output: PathLike,
        angle: int,
        pages: PageSpec | PageSelection | None = "all",
        *,
        relative: bool = True,
        password: str | None = None,
        overwrite: bool = True,
    ) -> Path:
        """顺时针旋转指定页面；``relative=False`` 时把 ``/Rotate`` 设为绝对角度。"""
        if angle % _ROTATION_STEP:
            raise PDFError(f"旋转角度必须是 {_ROTATION_STEP} 的整数倍，收到 {angle}")
        reader = open_reader(source, password)
        indices = _select(pages, len(reader.pages))
        writer = _clone(reader)
        for index in indices:
            _rotate_page(writer.pages[index], angle, relative=relative)
        return write_pdf(writer, as_path(output), overwrite=overwrite)

    @staticmethod
    def crop(
        source: PathLike,
        output: PathLike,
        box: Sequence[float],
        pages: PageSpec | PageSelection | None = "all",
        *,
        password: str | None = None,
        overwrite: bool = True,
    ) -> Path:
        """裁剪页面：``box=(左, 上, 右, 下)``，以左上角为原点、单位 pt。"""
        if len(tuple(box)) != 4:
            raise ValueError("box 需要 4 个数值：(left, top, right, bottom)")
        reader = open_reader(source, password)
        indices = _select(pages, len(reader.pages))
        writer = _clone(reader)
        for index in indices:
            _crop_page(writer.pages[index], box)
        return write_pdf(writer, as_path(output), overwrite=overwrite)

    @staticmethod
    def stamp(
        source: PathLike,
        output: PathLike,
        stamp: PathLike,
        pages: PageSpec | PageSelection | None = "all",
        *,
        over: bool = True,
        password: str | None = None,
        stamp_password: str | None = None,
        overwrite: bool = True,
    ) -> Path:
        """叠加印章/水印 PDF；印章只有一页时复用到所有目标页。"""
        reader = open_reader(source, password)
        stamp_reader = open_reader(stamp, stamp_password)
        indices = _select(pages, len(reader.pages))
        writer = _clone(reader)
        for order, index in enumerate(indices):
            stamp_index = order if len(stamp_reader.pages) > 1 else 0
            if stamp_index >= len(stamp_reader.pages):
                raise PageSpecError(
                    f"印章文档只有 {len(stamp_reader.pages)} 页，不足以覆盖 {len(indices)} 个目标页"
                )
            writer.pages[index].merge_page(stamp_reader.pages[stamp_index], over=over)
        return write_pdf(writer, as_path(output), overwrite=overwrite)

    # ------------------------------------------------------------ 体积与安全
    @staticmethod
    def compress(
        source: PathLike,
        output: PathLike,
        *,
        level: int = 9,
        password: str | None = None,
        overwrite: bool = True,
    ) -> Path:
        """压缩内容流并合并重复对象（无损、不改变页面外观）。"""
        if not 0 <= level <= 9:
            raise ValueError("level 需在 0-9 之间")
        reader = open_reader(source, password)
        writer = _clone(reader)
        for page in writer.pages:
            with suppress(Exception):  # 个别页面的内容流无法压缩，跳过即可
                page.compress_content_streams(level=level)
        with suppress(Exception):
            writer.compress_identical_objects()
        return write_pdf(writer, as_path(output), overwrite=overwrite)

    @staticmethod
    def encrypt(
        source: PathLike,
        output: PathLike,
        user_password: str,
        *,
        owner_password: str | None = None,
        algorithm: str = "AES-256",
        password: str | None = None,
        overwrite: bool = True,
    ) -> Path:
        """加密输出（默认 AES-256，需要 ``cryptography``）。"""
        if not user_password:
            raise ValueError("user_password 不能为空")
        reader = open_reader(source, password)
        writer = _clone(reader)
        writer.encrypt(
            user_password=user_password,
            owner_password=owner_password or user_password,
            algorithm=algorithm,
        )
        return write_pdf(writer, as_path(output), overwrite=overwrite)

    @staticmethod
    def decrypt(
        source: PathLike,
        output: PathLike,
        password: str,
        *,
        overwrite: bool = True,
    ) -> Path:
        """用密码去除加密（输出为无密码 PDF）。"""
        reader = open_reader(source, password)
        return write_pdf(_clone(reader), as_path(output), overwrite=overwrite)

    # ------------------------------------------------------------ 元数据
    @staticmethod
    def set_metadata(
        source: PathLike,
        output: PathLike,
        values: Mapping[str, str | None],
        *,
        password: str | None = None,
        overwrite: bool = True,
    ) -> Path:
        """合并写入元数据（键名可带或不带 ``/``，``None`` 值会被忽略）。"""
        reader = open_reader(source, password)
        writer = _clone(reader)
        updates = {
            f"/{key.lstrip('/')}": str(value)
            for key, value in values.items()
            if value is not None
        }
        if updates:
            writer.add_metadata(updates)
        return write_pdf(writer, as_path(output), overwrite=overwrite)
