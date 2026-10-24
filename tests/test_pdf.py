import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs.pdf import (
    PDF,
    OutputExistsError,
    PageSelection,
    PageSpecError,
    PDFError,
    PDFPasswordError,
)
from cs.pdf.__main__ import main

A4 = (595.3, 841.9)


def _make_pdf(
    path: Path, pages: int, *, size: tuple[float, float] = A4, title: str | None = None
) -> Path:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=size[0], height=size[1])
    if title:
        writer.add_metadata({"/Title": title, "/Author": "ShawnChen"})
    with path.open("wb") as stream:
        writer.write(stream)
    return path


@pytest.fixture
def base_pdf(tmp_path: Path) -> Path:
    return _make_pdf(tmp_path / "base.pdf", 5, title="示例文档")


@pytest.fixture
def other_pdf(tmp_path: Path) -> Path:
    return _make_pdf(tmp_path / "other.pdf", 2, size=(841.9, 1190.6))


@pytest.fixture
def text_pdf(tmp_path: Path) -> Path:
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    for number in range(1, 3):
        page = document.new_page()
        page.insert_text((72, 72), f"Hello Page {number}")
    path = tmp_path / "text.pdf"
    document.save(path)
    document.close()
    return path


# ---------------------------------------------------------------- 页码范围


@pytest.mark.parametrize(
    ("spec", "total", "expected"),
    [
        ("all", 5, [0, 1, 2, 3, 4]),
        ("*", 5, [0, 1, 2, 3, 4]),
        ("1", 5, [0]),
        ("-1", 5, [4]),
        ("-2", 5, [3, 4]),
        ("2-4", 5, [1, 2, 3]),
        ("3-", 5, [2, 3, 4]),
        ("odd", 5, [0, 2, 4]),
        ("even", 5, [1, 3]),
        ("1-10:3", 20, [0, 3, 6, 9]),
        ("3,1,2", 5, [2, 0, 1]),
        ("1-2,2-3", 5, [0, 1, 2]),  # 去重
        ("1-100", 4, [0, 1, 2, 3]),  # 越界收敛
    ],
)
def test_page_selection_resolve(spec: str, total: int, expected: list[int]) -> None:
    assert PageSelection(spec).resolve(total) == expected


def test_page_selection_chunks() -> None:
    assert PageSelection("1-2,4").chunks(5) == [[0, 1], [3]]


@pytest.mark.parametrize(
    "spec", ["", "abc", "0", "5-3", "-", "2-abc", "1-3,all", "1:0"]
)
def test_page_selection_invalid(spec: str) -> None:
    with pytest.raises(PageSpecError):
        PageSelection(spec)


def test_page_selection_out_of_range() -> None:
    with pytest.raises(PageSpecError):
        PageSelection("9").resolve(5)


def test_page_selection_parse_types() -> None:
    assert PageSelection.parse(None).spec == "all"
    assert PageSelection.parse(3).spec == "3"
    assert PageSelection.parse([1, 3]).resolve(4) == [0, 2]
    assert PageSelection.parse(PageSelection("2")).resolve(3) == [1]


# ---------------------------------------------------------------- 打开 / 信息


def test_info(base_pdf: Path) -> None:
    info = PDF.info(base_pdf)
    assert info.page_count == 5
    assert info.metadata["title"] == "示例文档"
    assert info.pages[0].size == "A4"
    assert info.pages[0].rotation == 0
    assert info.size_bytes > 0
    assert info.version and info.version.startswith("1.")


def test_handle(base_pdf: Path) -> None:
    with PDF.open(base_pdf) as document:
        assert document.page_count == 5
        assert document.info().page_count == 5
        assert document.text("1") == ""


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PDF.info(tmp_path / "nope.pdf")


def test_output_exists_policy(base_pdf: Path, tmp_path: Path) -> None:
    output = PDF.extract(base_pdf, tmp_path / "part.pdf", "1", overwrite=False)
    assert output.exists()
    with pytest.raises(OutputExistsError):
        PDF.extract(base_pdf, output, "1", overwrite=False)
    PDF.extract(base_pdf, output, "1-2", overwrite=True)  # 显式覆盖
    assert PDF.page_count(output) == 2


# ---------------------------------------------------------------- 页面操作


def test_extract(base_pdf: Path, tmp_path: Path) -> None:
    output = PDF.extract(base_pdf, tmp_path / "part.pdf", "3,1")
    assert PDF.page_count(output) == 2


def test_delete(base_pdf: Path, tmp_path: Path) -> None:
    output = PDF.delete(base_pdf, tmp_path / "kept.pdf", "1")
    assert PDF.page_count(output) == 4
    with pytest.raises(PageSpecError):
        PDF.delete(base_pdf, tmp_path / "empty.pdf", "all")


def test_reorder(base_pdf: Path, tmp_path: Path) -> None:
    output = PDF.reorder(base_pdf, tmp_path / "sorted.pdf", "5,4,3,2,1")
    assert PDF.page_count(output) == 5
    with pytest.raises(PageSpecError):
        PDF.reorder(base_pdf, tmp_path / "bad.pdf", "1,2")


def test_insert(base_pdf: Path, other_pdf: Path, tmp_path: Path) -> None:
    output = PDF.insert(base_pdf, tmp_path / "inserted.pdf", other_pdf, 2)
    assert PDF.page_count(output) == 7
    info = PDF.info(output)
    assert info.pages[1].size == "A3"  # 插入页在第 2 个位置
    with pytest.raises(PageSpecError):
        PDF.insert(base_pdf, tmp_path / "bad.pdf", other_pdf, 99)


def test_merge(base_pdf: Path, other_pdf: Path, tmp_path: Path) -> None:
    output = PDF.merge([base_pdf, other_pdf], tmp_path / "merged.pdf", outline=True)
    assert PDF.page_count(output) == 7
    from pypdf import PdfReader

    assert len(PdfReader(str(output)).outline) == 2
    with pytest.raises(ValueError):
        PDF.merge([], tmp_path / "none.pdf")


def test_split(base_pdf: Path, tmp_path: Path) -> None:
    every_page = PDF.split(base_pdf, tmp_path / "each")
    assert len(every_page) == 5
    assert every_page[0].name == "base_0001.pdf"

    batches = PDF.split(base_pdf, tmp_path / "batch", every=2)
    assert [PDF.page_count(item) for item in batches] == [2, 2, 1]

    chunks = PDF.split(base_pdf, tmp_path / "chunk", pages="1-3,5", prefix="part")
    assert [item.name for item in chunks] == ["part_0001-0003.pdf", "part_0005.pdf"]

    with pytest.raises(ValueError):
        PDF.split(base_pdf, tmp_path / "bad", pages="1", every=1)


# ---------------------------------------------------------------- 变形


def test_rotate_relative(base_pdf: Path, tmp_path: Path) -> None:
    output = PDF.rotate(base_pdf, tmp_path / "rot.pdf", 90, "1-2")
    from pypdf import PdfReader

    pages = PdfReader(str(output)).pages
    assert [int(page.get("/Rotate", 0)) for page in pages] == [90, 90, 0, 0, 0]

    doubled = PDF.rotate(output, tmp_path / "rot2.pdf", 90, "1")
    assert int(PdfReader(str(doubled)).pages[0].get("/Rotate", 0)) == 180


def test_rotate_absolute(base_pdf: Path, tmp_path: Path) -> None:
    output = PDF.rotate(base_pdf, tmp_path / "abs.pdf", 270, "1", relative=False)
    from pypdf import PdfReader

    assert int(PdfReader(str(output)).pages[0].get("/Rotate", 0)) == 270
    with pytest.raises(PDFError):
        PDF.rotate(base_pdf, tmp_path / "bad.pdf", 45)


def test_crop(base_pdf: Path, tmp_path: Path) -> None:
    output = PDF.crop(base_pdf, tmp_path / "crop.pdf", (10, 20, 500, 700), "1")
    from pypdf import PdfReader

    box = [float(value) for value in PdfReader(str(output)).pages[0].cropbox]
    assert box == pytest.approx([10, 141.9, 500, 821.9], abs=0.2)


def test_stamp(base_pdf: Path, other_pdf: Path, tmp_path: Path) -> None:
    output = PDF.stamp(base_pdf, tmp_path / "stamped.pdf", other_pdf, "1-2")
    assert PDF.page_count(output) == 5


# ---------------------------------------------------------------- 元数据 / 安全


def test_metadata(base_pdf: Path, tmp_path: Path) -> None:
    output = PDF.set_metadata(
        base_pdf, tmp_path / "meta.pdf", {"title": "新标题", "subject": "测试"}
    )
    info = PDF.info(output)
    assert info.metadata["title"] == "新标题"
    assert info.metadata["subject"] == "测试"
    assert info.metadata["author"] == "ShawnChen"


def test_encrypt_decrypt(base_pdf: Path, tmp_path: Path) -> None:
    locked = PDF.encrypt(base_pdf, tmp_path / "locked.pdf", "s3cret")
    with pytest.raises(PDFPasswordError):
        PDF.info(locked)
    with pytest.raises(PDFPasswordError):
        PDF.info(locked, password="wrong")
    assert PDF.info(locked, password="s3cret").encrypted is True

    unlocked = PDF.decrypt(locked, tmp_path / "unlocked.pdf", "s3cret")
    info = PDF.info(unlocked)
    assert info.encrypted is False
    assert info.page_count == 5


def test_compress(base_pdf: Path, tmp_path: Path) -> None:
    output = PDF.compress(base_pdf, tmp_path / "small.pdf")
    assert PDF.page_count(output) == 5


# ---------------------------------------------------------------- 文本 / 渲染


def test_text(text_pdf: Path) -> None:
    pages = PDF.text_pages(text_pdf)
    assert [page["number"] for page in pages] == [1, 2]
    assert "Hello Page 1" in pages[0]["text"]
    assert "Hello Page 2" in PDF.text(text_pdf, "2")
    assert PDF.text(text_pdf, "2") == PDF.text_pages(text_pdf, "2")[0]["text"]


def test_render(text_pdf: Path, tmp_path: Path) -> None:
    pytest.importorskip("pymupdf")
    files = PDF.render(text_pdf, tmp_path / "img", dpi=72, pages="1")
    assert len(files) == 1
    assert files[0].suffix == ".png"
    assert files[0].stat().st_size > 0
    with pytest.raises(ValueError):
        PDF.render(text_pdf, tmp_path / "img", fmt="bmp")


# ---------------------------------------------------------------- MCP


def test_mcp_tools_registered() -> None:
    pytest.importorskip("fastmcp")
    from fastmcp import Client

    server = __import__("cs.pdf.mcp_server", fromlist=["create_server"]).create_server()

    async def collect() -> list[str]:
        async with Client(server) as client:
            return sorted(tool.name for tool in await client.list_tools())

    names = asyncio.run(collect())
    assert names == [
        "pdf_compress",
        "pdf_crop",
        "pdf_decrypt",
        "pdf_delete_pages",
        "pdf_encrypt",
        "pdf_extract",
        "pdf_info",
        "pdf_insert",
        "pdf_merge",
        "pdf_render",
        "pdf_reorder",
        "pdf_rotate",
        "pdf_set_metadata",
        "pdf_split",
        "pdf_stamp",
        "pdf_text",
    ]


def test_mcp_call_info_and_error(base_pdf: Path) -> None:
    pytest.importorskip("fastmcp")
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    from cs.pdf.mcp_server import create_server

    server = create_server()

    async def run() -> tuple[dict, str]:
        async with Client(server) as client:
            result = await client.call_tool("pdf_info", {"path": str(base_pdf)})
            payload = result.structured_content or json.loads(result.content[0].text)
            try:
                await client.call_tool(
                    "pdf_info", {"path": str(base_pdf.parent / "missing.pdf")}
                )
            except ToolError as exc:
                return payload, str(exc)
            raise AssertionError("应当抛出 ToolError")

    payload, message = asyncio.run(run())
    assert payload["ok"] is True
    assert payload["page_count"] == 5
    assert "FileNotFoundError" in message


# ---------------------------------------------------------------- CLI


def test_cli_info(base_pdf: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", str(base_pdf), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["page_count"] == 5
