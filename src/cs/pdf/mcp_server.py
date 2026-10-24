"""把 :class:`cs.pdf.PDF` 暴露为 MCP 工具（FastMCP ≥ 4）。

启动方式::

    python -m cs.pdf serve                 # stdio，供 MCP 客户端拉起
    python -m cs.pdf serve --transport http --port 8000
    fastmcp run src/cs/pdf/mcp_server.py:mcp

约定：工具名统一 ``pdf_*``；成功返回 ``{"ok": true, ...}``，
业务异常统一转换为 ``ToolError``，避免把 traceback 泄露给模型。
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from .core import PDF, ImageFormat
from .errors import PDFError

__all__ = ["create_server", "mcp"]

_SERVER_NAME = "cs-pdf"
_INSTRUCTIONS = """\
本服务器提供本地 PDF 文件操作：读取信息与文本、渲染图片、抽取/删除/重排/插入页面、
拆分、合并、旋转、裁剪、加水印、压缩、加密解密、修改元数据。

使用建议：
1. 不确定页码时先调用 pdf_info 查看页数；
2. pages 参数支持 "all"、"1-3"、"1-3,7"、"-2"（最后两页）、"odd"/"even"、"1-10:2"；
3. 所有写操作都返回新文件路径，默认不覆盖已存在文件（需显式 overwrite=true）；
4. 建议传绝对路径（相对路径按服务进程的工作目录解析），输出目录会自动创建。
"""


def _guard[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """把领域异常转换为 MCP 可读错误。"""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except (PDFError, OSError, ValueError) as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


def _done(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def _path(target: Path) -> str:
    return str(target)


def create_server(
    name: str = _SERVER_NAME, *, instructions: str = _INSTRUCTIONS
) -> FastMCP:
    """构造一个独立的服务器实例（便于测试或多实例部署）。"""
    server: FastMCP = FastMCP(name, instructions=instructions)

    # ---------------------------------------------------------- 只读
    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def pdf_info(
        path: str, include_pages: bool = True, password: str | None = None
    ) -> dict[str, Any]:
        """读取 PDF 概览：页数、页面尺寸/旋转、元数据、文件大小、是否加密。

        include_pages=False 时省略逐页明细，适合页数很多的大文档。
        """
        info = PDF.info(path, password=password)
        return _done(**info.to_dict(include_pages=include_pages))

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def pdf_text(
        path: str,
        pages: str = "all",
        layout: bool = False,
        max_chars: int = 200_000,
        password: str | None = None,
    ) -> dict[str, Any]:
        """抽取文本（按页返回）。layout=True 时保留版面排版，适合表格类文档。

        max_chars 为总字符上限，超出会截断并置 truncated=true。
        """
        if max_chars < 1:
            raise ValueError("max_chars 必须为正整数")
        pages_data = PDF.text_pages(path, pages, layout=layout, password=password)
        total = sum(page["chars"] for page in pages_data)
        remain = max_chars
        clipped: list[dict[str, Any]] = []
        for page in pages_data:
            if remain <= 0:
                break
            text = page["text"][:remain]
            clipped.append({"number": page["number"], "text": text, "chars": len(text)})
            remain -= len(text)
        return _done(pages=clipped, total_chars=total, truncated=total > max_chars)

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def pdf_render(
        path: str,
        out_dir: str,
        dpi: int = 200,
        pages: str = "all",
        fmt: str = ImageFormat.PNG,
        password: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """把页面渲染为图片（需要 pymupdf），返回生成的文件路径列表。"""
        outputs = PDF.render(
            path,
            out_dir,
            dpi=dpi,
            pages=pages,
            fmt=fmt,
            password=password,
            overwrite=overwrite,
        )
        return _done(files=[_path(item) for item in outputs], count=len(outputs))

    # ---------------------------------------------------------- 页面级
    @server.tool()
    @_guard
    def pdf_extract(
        path: str,
        output: str,
        pages: str,
        password: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """抽取指定页面另存为新 PDF，页面顺序与表达式一致。"""
        result = PDF.extract(
            path, output, pages, password=password, overwrite=overwrite
        )
        return _done(output=_path(result))

    @server.tool()
    @_guard
    def pdf_delete_pages(
        path: str,
        output: str,
        pages: str,
        password: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """删除指定页面，其余页面按原顺序输出（不允许删空）。"""
        result = PDF.delete(path, output, pages, password=password, overwrite=overwrite)
        return _done(output=_path(result))

    @server.tool()
    @_guard
    def pdf_reorder(
        path: str,
        output: str,
        order: str,
        password: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """重排页面，如 order="3,1,2"；必须不重不漏地覆盖全部页面。"""
        result = PDF.reorder(
            path, output, order, password=password, overwrite=overwrite
        )
        return _done(output=_path(result))

    @server.tool()
    @_guard
    def pdf_insert(
        path: str,
        output: str,
        other: str,
        at: int,
        pages: str = "all",
        password: str | None = None,
        other_password: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """把 other 的页面插入到 path 的第 at 页之前（1 起始；at=页数+1 表示追加）。"""
        result = PDF.insert(
            path,
            output,
            other,
            at,
            pages=pages,
            password=password,
            other_password=other_password,
            overwrite=overwrite,
        )
        return _done(output=_path(result))

    @server.tool()
    @_guard
    def pdf_split(
        path: str,
        out_dir: str,
        pages: str | None = None,
        every: int | None = None,
        prefix: str | None = None,
        password: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """拆分 PDF。不传参数=每页一个文件；every=N=每 N 页一个文件；pages="1-3,5"=按片段拆。"""
        outputs = PDF.split(
            path,
            out_dir,
            pages=pages,
            every=every,
            prefix=prefix,
            password=password,
            overwrite=overwrite,
        )
        return _done(files=[_path(item) for item in outputs], count=len(outputs))

    @server.tool()
    @_guard
    def pdf_merge(
        paths: list[str],
        output: str,
        outline: bool = False,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """按给定顺序合并多个 PDF；outline=True 时为每个源文件生成书签。"""
        result = PDF.merge(paths, output, outline=outline, overwrite=overwrite)
        return _done(output=_path(result), sources=len(paths))

    # ---------------------------------------------------------- 变形与安全
    @server.tool()
    @_guard
    def pdf_rotate(
        path: str,
        output: str,
        angle: int,
        pages: str = "all",
        relative: bool = True,
        password: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """顺时针旋转页面，角度须为 90 的倍数；relative=False 表示设为绝对角度。"""
        result = PDF.rotate(
            path,
            output,
            angle,
            pages,
            relative=relative,
            password=password,
            overwrite=overwrite,
        )
        return _done(output=_path(result))

    @server.tool()
    @_guard
    def pdf_crop(
        path: str,
        output: str,
        left: float,
        top: float,
        right: float,
        bottom: float,
        pages: str = "all",
        password: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """裁剪页面，坐标以左上角为原点、单位 pt（1pt=1/72 英寸）。"""
        result = PDF.crop(
            path,
            output,
            (left, top, right, bottom),
            pages,
            password=password,
            overwrite=overwrite,
        )
        return _done(output=_path(result))

    @server.tool()
    @_guard
    def pdf_stamp(
        path: str,
        output: str,
        stamp: str,
        pages: str = "all",
        over: bool = True,
        password: str | None = None,
        stamp_password: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """把印章/水印 PDF 叠加到目标页；印章一页时复用到所有目标页。"""
        result = PDF.stamp(
            path,
            output,
            stamp,
            pages,
            over=over,
            password=password,
            stamp_password=stamp_password,
            overwrite=overwrite,
        )
        return _done(output=_path(result))

    @server.tool()
    @_guard
    def pdf_compress(
        path: str,
        output: str,
        level: int = 9,
        password: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """无损压缩：压缩内容流并合并重复对象，返回压缩前后的字节数。"""
        before = Path(path).stat().st_size
        result = PDF.compress(
            path, output, level=level, password=password, overwrite=overwrite
        )
        after = result.stat().st_size
        ratio = round(after / before, 4) if before else None
        return _done(
            output=_path(result), size_before=before, size_after=after, ratio=ratio
        )

    @server.tool()
    @_guard
    def pdf_encrypt(
        path: str,
        output: str,
        user_password: str,
        owner_password: str | None = None,
        algorithm: str = "AES-256",
        password: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """加密 PDF。algorithm 可选 AES-256 / AES-128 / RC4-128 / RC4-40。"""
        result = PDF.encrypt(
            path,
            output,
            user_password,
            owner_password=owner_password,
            algorithm=algorithm,
            password=password,
            overwrite=overwrite,
        )
        return _done(output=_path(result), algorithm=algorithm)

    @server.tool()
    @_guard
    def pdf_decrypt(
        path: str,
        output: str,
        password: str,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """用密码去除加密，输出无密码保护的 PDF。"""
        result = PDF.decrypt(path, output, password, overwrite=overwrite)
        return _done(output=_path(result))

    @server.tool()
    @_guard
    def pdf_set_metadata(
        path: str,
        output: str,
        title: str | None = None,
        author: str | None = None,
        subject: str | None = None,
        keywords: str | None = None,
        creator: str | None = None,
        producer: str | None = None,
        password: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """写入/合并文档元数据，未提供的字段保持原值。"""
        values = {
            "title": title,
            "author": author,
            "subject": subject,
            "keywords": keywords,
            "creator": creator,
            "producer": producer,
        }
        result = PDF.set_metadata(
            path, output, values, password=password, overwrite=overwrite
        )
        return _done(
            output=_path(result), updated=sorted(k for k, v in values.items() if v)
        )

    return server


mcp = create_server()
