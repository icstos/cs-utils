"""cs.cv_draw 的行为测试。

覆盖：拷贝/原图两种绘制语义、各图元落笔、样式与 JSON 规格校验、
图像读写（中文路径、覆盖策略）、掩码与网格，以及 MCP 工具注册。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs.cv import Box, Category, Circle, Line, Point, Polygon
from cs.cv_draw import (
    Anno,
    Canvas,
    ImageError,
    ShapeError,
    Stroke,
    Style,
    Text,
    image_info,
    imread,
    imwrite,
    parse_shape,
    render,
)

BACKGROUND = 30


def blank(width: int = 320, height: int = 240) -> np.ndarray:
    return np.full((height, width, 3), BACKGROUND, np.uint8)


def changed(image: np.ndarray) -> bool:
    return bool((image != BACKGROUND).any())


def test_copy_semantics_keeps_source_intact():
    source = blank()
    drawn = Canvas(source).box(Box(Point(10, 10), Point(80, 60))).result()
    assert not changed(source)
    assert changed(drawn)


def test_inplace_modifies_source_array():
    source = blank()
    canvas = Canvas(source, inplace=True).circle(Circle(Point(60, 60), 20))
    assert canvas.result() is source
    assert changed(source)


def test_chain_returns_same_canvas():
    canvas = Canvas(blank())
    assert canvas.point(Point(5, 5)) is canvas
    assert canvas.grid(2, 2) is canvas


def test_grayscale_input_is_converted():
    drawn = Canvas(np.full((80, 80), 10, np.uint8)).point(Point(40, 40)).result()
    assert drawn.shape == (80, 80, 3)


@pytest.mark.parametrize(
    "image",
    [np.zeros((0, 0, 3), np.uint8), np.zeros((10, 10, 3), np.float32)],
    ids=["empty", "float32"],
)
def test_invalid_images_rejected(image):
    with pytest.raises(ImageError):
        Canvas(image)


def test_each_shape_leaves_pixels():
    canvas = Canvas(blank(400, 300))
    canvas.points([Point(20, 20), Point(40, 20)], style=Style(thickness=3))
    canvas.line(Line(Point(10, 60), Point(200, 60)))
    canvas.box(Box(Point(30, 80), Point(150, 140)))
    canvas.polygon(Polygon([Point(200, 90), Point(280, 90), Point(240, 160)]))
    canvas.circle(Circle(Point(330, 120), 25))
    canvas.text("标签 ABC", Point(20, 200), size=18)
    assert changed(canvas.result())


def test_label_uses_category_and_conf():
    category = Category(["缺陷", "划痕"])
    canvas = Canvas(blank(400, 200), category=category)
    canvas.box(Box(Point(50, 60), Point(200, 150), conf=0.87, category_idx=1))
    assert changed(canvas.result())

    silent = Canvas(blank(400, 200), category=category)
    silent.box(
        Box(Point(50, 60), Point(200, 150), conf=0.87),
        style=Style(label=False),
    )
    assert changed(silent.result())


def test_explicit_label_overrides_auto():
    assert Canvas(blank())._text_of(0, 0.5, "自定义") == "自定义"
    canvas = Canvas(blank(), category=Category(["cat"]))
    assert canvas._text_of(0, 0.5, None) == "cat: 0.50"
    assert canvas._text_of(0, -1, None) == "cat"
    assert canvas._text_of(9, 0.5, None) == "0.50"


def test_stroke_styles_all_render():
    for stroke in Stroke:
        canvas = Canvas(blank())
        canvas.line(
            Line(Point(10, 10), Point(300, 200)),
            style=Style(stroke=stroke, thickness=2),
        )
        assert changed(canvas.result()), stroke


def test_fill_alpha_only_touches_region():
    canvas = Canvas(blank())
    canvas.box(Box(Point(20, 20), Point(120, 120)), style=Style(fill=True, alpha=0.5))
    result = canvas.result()
    assert not np.array_equal(result[70, 70], np.array([BACKGROUND] * 3))
    assert tuple(result[5, 5]) == (BACKGROUND,) * 3


def test_mask_and_grid():
    mask = np.zeros((240, 320), np.uint8)
    mask[50:150, 50:150] = 255
    masked = Canvas(blank()).mask(mask, color=(0, 0, 255), alpha=0.5).result()
    assert not np.array_equal(masked[100, 100], np.array([BACKGROUND] * 3))
    assert tuple(masked[10, 10]) == (BACKGROUND,) * 3

    gridded = Canvas(blank()).grid(4, 3).result()
    assert changed(gridded)


def test_out_of_bounds_shapes_are_clipped():
    canvas = Canvas(blank(60, 50))
    canvas.box(Box(Point(-40, -40), Point(500, 500)))
    canvas.text("超出边界的文字内容", Point(55, 45), size=26)
    assert canvas.result().shape == (50, 60, 3)


def test_empty_shapes_are_noop():
    canvas = Canvas(blank(40, 40))
    canvas.polyline([]).polygon(Polygon()).text("", Point(1, 1))
    assert np.array_equal(canvas.result(), blank(40, 40))


def test_style_from_dict_roundtrip():
    style = Style.from_dict(
        {"color": [1, 2, 3], "stroke": "dashed", "alpha": 0.4, "thickness": 3}
    )
    assert style.color == (1, 2, 3)
    assert style.stroke is Stroke.DASHED
    assert style.to_dict()["stroke"] == "dashed"


@pytest.mark.parametrize(
    "payload",
    [
        {"nope": 1},
        {"alpha": 2},
        {"alpha": -0.1},
        {"stroke": "zigzag"},
        {"color": [1, 2]},
    ],
)
def test_style_rejects_bad_input(payload):
    with pytest.raises(ShapeError):
        Style.from_dict(payload)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"kind": "point", "x": 1, "y": 2}, Point),
        ({"points": [[1, 2], [3, 4]]}, Line),
        ({"x1": 1, "y1": 2, "x2": 3, "y2": 4}, Box),
        ({"points": [[1, 1], [5, 1], [5, 5]]}, Polygon),
        ({"cx": 5, "cy": 5, "r": 3}, Circle),
        ({"text": "hi", "x": 2, "y": 3}, Text),
    ],
)
def test_parse_shape_infers_kind(spec, expected):
    anno = parse_shape(spec)
    assert isinstance(anno, Anno)
    assert isinstance(anno.shape, expected)


def test_parse_shape_keeps_label_and_conf():
    anno = parse_shape(
        {
            "kind": "box",
            "x1": 1,
            "y1": 2,
            "x2": 3,
            "y2": 4,
            "label": "obj",
            "conf": 0.5,
            "category_idx": 2,
        }
    )
    assert anno.label == "obj"
    assert anno.shape.conf == 0.5
    assert anno.shape.category_idx == 2


@pytest.mark.parametrize(
    "spec",
    [
        {"kind": "star", "x": 1, "y": 1},
        {"kind": "circle", "cx": 0, "cy": 0, "r": 0},
        {"kind": "polygon", "points": [[0, 0], [1, 1]]},
        {"kind": "text", "text": "  ", "x": 0, "y": 0},
        {"kind": "box", "x1": 0, "y1": 0},
        {},
    ],
)
def test_parse_shape_rejects_bad_input(spec):
    with pytest.raises(ShapeError):
        parse_shape(spec)


def test_render_dispatches_mixed_shapes():
    result = render(
        blank(),
        [
            Anno(Box(Point(5, 5), Point(80, 60)), label="X"),
            Text("T", Point(10, 150), 14),
            Circle(Point(200, 100), 25),
        ],
    )
    assert changed(result)


def test_render_rejects_unknown_object():
    with pytest.raises(ShapeError):
        Canvas(blank()).draw("not-a-shape")


def test_image_roundtrip_with_chinese_path(tmp_path: Path):
    source = tmp_path / "原图.png"
    imwrite(source, blank(), overwrite=True)
    info = image_info(source)
    assert (info.width, info.height, info.channels) == (320, 240, 3)

    target = tmp_path / "输出" / "结果.jpg"
    saved = Canvas(imread(source)).box(Box(Point(10, 10), Point(100, 100))).save(target)
    assert saved.is_file()
    assert imread(saved).shape == (240, 320, 3)


def test_imwrite_refuses_overwrite_without_flag(tmp_path: Path):
    target = tmp_path / "out.png"
    imwrite(target, blank(), overwrite=True)
    with pytest.raises(ImageError):
        imwrite(target, blank())
    assert imwrite(target, blank(), overwrite=True).is_file()


def test_imread_missing_file(tmp_path: Path):
    with pytest.raises(ImageError):
        imread(tmp_path / "缺失.png")


def test_mcp_server_exposes_tools():
    pytest.importorskip("fastmcp")
    import asyncio

    from fastmcp import Client

    from cs.cv_draw.mcp_server import create_server

    async def names() -> list[str]:
        async with Client(create_server()) as client:
            return [tool.name for tool in await client.list_tools()]

    assert asyncio.run(names()) == [
        "cv_draw_info",
        "cv_draw_shapes",
        "cv_draw_text",
        "cv_draw_mask",
        "cv_draw_grid",
    ]
