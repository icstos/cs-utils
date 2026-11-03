"""cs.cv 的几何、数据模型、标注与推理封装测试。

真实模型相关用例在缺少 ``yolo11n.pt`` 或 ultralytics 时自动跳过 ——
其余用例不需要任何权重文件，可在离线环境跑。
"""

import asyncio
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs.cv import (
    FORMATS,
    AnnoDataset,
    AnnoImg,
    Box,
    Category,
    Circle,
    Cv,
    CVError,
    Geometry,
    ImgSize,
    Line,
    Metrics,
    Model,
    ModelError,
    ModelFormat,
    ModelNotFoundError,
    Obb,
    Point,
    Polygon,
    Pose,
    Probs,
    Result,
    Task,
    format_of,
    imwrite,
    resolve_device,
)
from cs.my_types import TaskType

SIZE = ImgSize(200, 100)
WEIGHTS = ROOT / "yolo11n.pt"


def _runtime_ready() -> bool:
    """真实推理需要本地权重 + 可用的 torchvision（部分环境的 TV 与 torch 版本不匹配）。"""
    if not WEIGHTS.exists():
        return False
    try:
        from torchvision.ops import nms  # noqa: F401
    except Exception:  # noqa: BLE001 - 版本不匹配时可能是任意 RuntimeError
        return False
    return True


RUNTIME_READY = _runtime_ready()
NO_RUNTIME = pytest.mark.skipif(
    not RUNTIME_READY, reason="需要本地 yolo11n.pt 且 torchvision 可用"
)


# ============================================================
# 几何
# ============================================================


def test_point_bounds_use_radius():
    point = Point(50, 40)
    assert (point.x_min, point.y_min, point.x_max, point.y_max) == (46, 36, 54, 44)
    assert point.width == point.height == 8
    assert point.xy_int == (50, 40)


def test_box_normalizes_corners():
    box = Box(Point(80, 60), Point(10, 10))
    assert (box.x_min, box.y_min, box.x_max, box.y_max) == (10, 10, 80, 60)
    assert box.width == 70 and box.height == 50


def test_box_iou_and_iou_max():
    left = Box(Point(0, 0), Point(100, 100))
    right = Box(Point(50, 0), Point(150, 100))
    assert left.iou(right) == pytest.approx(1 / 3)
    assert left.iou_max(right) == pytest.approx(0.5)
    assert left.iou(Box(Point(200, 200), Point(300, 300))) == 0.0


def test_box_from_xyxy_and_bounding():
    box = Box.from_xyxy(10, 20, 30, 40, category_idx=2, conf=0.5)
    assert box.xyxy == (10, 20, 30, 40)
    assert box.category_idx == 2

    union = Box.bounding([box, Box(Point(50, 0), Point(60, 10))])
    assert union.xyxy == (10, 0, 60, 40)


def test_box_yolo_roundtrip():
    box = Box(Point(10, 10), Point(50, 30), category_idx=1, conf=0.5)
    line = box.to_yolo(SIZE)
    assert line.startswith("1 ")
    back = Box.from_yolo(line, SIZE)
    assert back.xyxy == pytest.approx(box.xyxy)
    assert Box.from_yolo(box.to_yolo(SIZE, with_conf=True), SIZE).conf == 0.5


def test_polygon_area_perimeter_centroid():
    triangle = Polygon([Point(0, 0), Point(10, 0), Point(0, 10)])
    assert triangle.area == pytest.approx(50)
    assert triangle.perimeter == pytest.approx(10 + 10 + 14.142136, rel=1e-5)
    assert triangle.centroid.xy == pytest.approx((10 / 3, 10 / 3))


def test_polygon_simplify_reduces_vertices():
    circle = Circle(Point(100, 50), 40)
    polygon = Polygon(circle.points)
    simplified = polygon.simplify(epsilon_ratio=0.01, min_vertices=4, max_vertices=40)
    assert len(simplified.points) < len(polygon.points)
    assert simplified.area > polygon.area * 0.5


def test_polygon_yolo_roundtrip():
    polygon = Polygon([Point(10, 10), Point(50, 10), Point(50, 40)], category_idx=3)
    back = Polygon.from_yolo(polygon.to_yolo(SIZE), SIZE)
    assert back.category_idx == 3
    assert [p.xy for p in back.points] == pytest.approx([p.xy for p in polygon.points])


def test_circle_geometry():
    circle = Circle(Point(100, 50), 10)
    assert circle.area == pytest.approx(3.14159265 * 100)
    assert circle.circumference == pytest.approx(2 * 3.14159265 * 10)
    assert len(circle.points) == 64
    # 采样点包含四个极点，外接框与解析值一致
    assert (circle.x_min, circle.y_min) == (90, 40)
    assert circle.contains(Point(105, 50))
    assert not circle.contains(Point(120, 50))
    assert circle.distance_to_boundary(Point(120, 50)) == pytest.approx(10)


def test_circle_radius_normalized_by_short_side():
    line = Circle(Point(100, 50), 10).to_yolo(SIZE)
    # 100 为短边 → 归一化半径 0.1
    assert line.split()[3] == "0.1"
    assert Circle.from_yolo(line, SIZE).radius == pytest.approx(10)


def test_line_geometry_and_normalization():
    line = Line(Point(50, 60), Point(10, 20))
    assert line.start.xy == (10, 20)
    assert line.length == pytest.approx(np.hypot(40, 40))
    assert line.angle == pytest.approx(45)
    assert line.slope == pytest.approx(1)
    assert Line(Point(10, 10), Point(10, 50)).slope == float("inf")
    assert line.center.xy == pytest.approx((30, 40))


def test_obb_corners_and_roundtrip():
    obb = Obb(Point(100, 50), (40, 20), 30.0, category_idx=1, conf=0.6)
    assert len(obb.points) == 4
    assert obb.area == pytest.approx(800)
    # 旋转后外接框一定不小于原始尺寸
    assert obb.to_box().width >= 40

    back = Obb.from_yolo(obb.to_yolo(SIZE), SIZE)
    assert back.center.xy == pytest.approx(obb.center.xy)
    assert back.size == pytest.approx(obb.size, rel=1e-4)
    assert back.angle == pytest.approx(obb.angle, abs=1e-3)
    assert back.category_idx == 1


def test_pose_bounds_and_visibility_roundtrip():
    pose = Pose(
        points=(Point(10, 20, conf=1.0), Point(30, 40, conf=0.5)),
        category_idx=1,
        conf=0.8,
        box=Box(Point(0, 0), Point(50, 60)),
    )
    assert pose.xyxy == (0, 0, 50, 60)
    assert len(pose) == 2
    assert pose.keypoints == pose.points

    line = pose.to_yolo(SIZE)
    assert line.split()[3] == "2" and line.split()[6] == "1"
    assert len(Pose.from_yolo(line, SIZE).points) == 2


def test_geometry_is_frozen_and_hashable():
    box = Box(Point(0, 0), Point(10, 10))
    with pytest.raises(FrozenInstanceError):
        box.start = Point(5, 5)  # type: ignore[misc]
    moved = box.offset(5, 5)
    assert moved is not box and box.xyxy == (0, 0, 10, 10)
    assert len({box, moved, box}) == 2


def test_clamp_and_validate():
    box = Box(Point(-10, -10), Point(50, 50))
    assert not box.validate(SIZE)
    clamped = box.clamp(SIZE)
    assert clamped.xyxy == (0, 0, 60, 60)
    assert clamped.validate(ImgSize(200, 200))

    outside = Box(Point(150, 0), Point(250, 10)).clamp(ImgSize(200, 100))
    assert outside.x_max == 200


# ============================================================
# 数据模型
# ============================================================


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("detect", Task.DETECT),
        ("SEGMENT", Task.SEGMENT),
        (2, Task.DETECT),
        (4, Task.POSE),
        (TaskType.OBB, Task.OBB),
        (None, Task.DETECT),
        (Task.CLASSIFY, Task.CLASSIFY),
    ],
)
def test_task_parse_variants(value, expected):
    assert Task.parse(value) is expected


def test_task_parse_rejects_unknown():
    with pytest.raises(ValueError, match="未知任务"):
        Task.parse("tracking")
    assert Task.DETECT.task_type is TaskType.DETECT


def test_category_parse_and_lookup():
    category = Category(["cat", "dog"])
    assert len(category) == 2
    assert category[1] == "dog"
    assert "cat" in category
    assert category.index("dog") == 1
    assert category.get(5) == ""
    assert category.get(5, "?") == "?"

    from_mapping = Category.parse({1: "b", 0: "a"})
    assert from_mapping.names == ("a", "b")
    with pytest.raises(ValueError):
        Category.parse("abc")


def test_probs_top():
    probs = Probs(names=("a", "b", "c"), values=(0.1, 0.7, 0.2))
    assert probs.top1 == ("b", 0.7)
    assert [name for name, _ in probs.top(2)] == ["b", "c"]
    with pytest.raises(ValueError):
        probs.top(0)


def test_result_views_and_transforms():
    result = Result(
        source=Path("a.jpg"),
        objs=[
            Box(Point(10, 10), Point(20, 20), category_idx=0, conf=0.9),
            Polygon([Point(30, 30), Point(40, 30), Point(35, 40)], category_idx=1),
            Obb(Point(60, 60), (10, 5), 15.0, category_idx=0),
        ],
        names=Category(["cat", "dog"]),
    )
    assert len(result) == 3
    assert len(result.boxes) == 1
    assert len(result.polygons) == 1
    assert len(result.obbs) == 1
    assert result.poses == []
    assert result.category_indices == [0, 1]
    assert result.label(0, 0.9) == "cat: 0.90"
    assert result.label(9, 0.9) == ""

    moved = result.shifted(10, 20)
    assert moved.boxes[0].xyxy == (20, 30, 30, 40)
    assert moved.polygons[0].points[0].xy == (40, 50)
    # 原结果不受影响
    assert result.boxes[0].xyxy == (10, 10, 20, 20)

    scaled = result.scaled(2, 2)
    assert scaled.boxes[0].xyxy == (20, 20, 40, 40)
    assert scaled.obbs[0].size == pytest.approx((20, 10))


def test_result_to_dict_and_render():
    image = np.full((60, 80, 3), 20, np.uint8)
    result = Result(
        source=Path("a.jpg"),
        objs=[Box(Point(10, 10), Point(30, 30), conf=0.75)],
        image=image,
        names=Category(["cat"]),
    )
    payload = result.to_dict()
    assert payload["count"] == 1 and payload["objs"][0]["kind"] == "box"
    assert payload["source"] == "a.jpg" and payload["path"] is None

    drawn = result.render()
    assert drawn.shape == image.shape
    assert not np.array_equal(drawn, image)

    empty = Result()
    with pytest.raises(ValueError, match="没有原始图像"):
        empty.render()


# ============================================================
# 标注
# ============================================================


@pytest.mark.parametrize(
    ("task", "line"),
    [
        ("detect", "0 0.5 0.5 0.2 0.2"),
        ("segment", "0 0.1 0.1 0.5 0.1 0.5 0.5"),
        ("obb", "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6"),
        ("pose", "0 0.1 0.2 2 0.3 0.4 1"),
    ],
)
def test_anno_img_roundtrip(task, line):
    anno = AnnoImg.parse(line, SIZE, task)
    assert len(anno) == 1
    assert anno.to_yolo() == line
    assert anno.to_dict()["task"] == task


def test_anno_img_rejects_classify():
    with pytest.raises(CVError, match="行文本"):
        AnnoImg.parse("0", SIZE, "classify")


def test_anno_img_save_and_load(tmp_path: Path):
    label = tmp_path / "a.txt"
    AnnoImg.parse("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1", SIZE).save(label)
    loaded = AnnoImg.load(label, SIZE)
    assert len(loaded) == 2
    assert loaded.label == label
    assert loaded.filter(min_conf=0.0) == []


def test_anno_dataset_scan_summary_compare(tmp_path: Path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    imwrite(images / "a.jpg", np.zeros((100, 200, 3), np.uint8), overwrite=True)
    imwrite(images / "b.jpg", np.zeros((100, 200, 3), np.uint8), overwrite=True)
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")

    dataset = AnnoDataset(images, labels, category=Category(["cat"]))
    summary = dataset.summary()
    assert (summary.images, summary.labels, summary.objects) == (2, 1, 1)
    assert summary.categories == {"cat": 1}
    assert summary.coverage == 0.5
    assert summary.to_dict()["coverage"] == 0.5
    assert len(dataset.labels) == 1 and len(dataset.images) == 2

    # 预测与原标签完全重合 → 全部命中
    predicted = AnnoDataset(images, labels)
    predicted.items["a.jpg"].objs = [
        Box(Point(80, 40), Point(120, 60), category_idx=0, conf=0.9)
    ]
    report = dataset.compare(predicted, iou=0.5)
    assert report.matched == 1 and report.recall == 1.0 and report.precision == 1.0
    assert report.to_dict()["missed"] == 0

    # 类别不同 → 不匹配
    predicted.items["a.jpg"].objs[0] = Box(
        Point(80, 40), Point(120, 60), category_idx=1
    )
    assert dataset.compare(predicted, iou=0.5).matched == 0


def test_anno_dataset_missing_dir(tmp_path: Path):
    with pytest.raises(CVError, match="图像目录不存在"):
        AnnoDataset(tmp_path / "nope", tmp_path / "labels")


# ============================================================
# 推理封装（不需要权重）
# ============================================================


def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device(0) == "cuda:0"
    assert resolve_device("1") == "cuda:1"
    assert resolve_device("cuda:2") == "cuda:2"
    assert resolve_device(None) in {"cpu", "cuda:0", "mps"}
    with pytest.raises(ModelError, match="无法识别的设备"):
        resolve_device("tpu")


def test_format_table_matches_legacy():
    assert [item.format for item in FORMATS] == [
        ModelFormat.PT,
        ModelFormat.TORCHSCRIPT,
        ModelFormat.ONNX,
        ModelFormat.OPENVINO,
        ModelFormat.ENGINE,
    ]
    assert [item.suffix for item in FORMATS] == [
        ".pt",
        ".torchscript",
        ".onnx",
        "_openvino_model",
        ".engine",
    ]
    assert FORMATS[4].cpu is False and FORMATS[4].gpu is True


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("yolo11n.pt", ModelFormat.PT),
        ("model.onnx", ModelFormat.ONNX),
        ("model.torchscript", ModelFormat.TORCHSCRIPT),
        ("model.engine", ModelFormat.ENGINE),
        ("model_openvino_model", ModelFormat.OPENVINO),
    ],
)
def test_format_of(source, expected):
    info = format_of(source)
    assert info is not None and info.format is expected


def test_format_of_unknown():
    assert format_of("model.tflite") is None


def test_model_missing_weights(tmp_path: Path):
    model = Model(tmp_path / "nope" / "best.pt")
    with pytest.raises(ModelNotFoundError):
        model.load()
    assert not model.loaded


def test_model_task_normalized():
    assert Model("yolo11n.pt", task="segment").task is Task.SEGMENT
    assert Model("yolo11n.pt", task=4).task is Task.POSE
    assert Model("yolo11n.pt").task is None


def test_describe_dispatches_by_suffix(tmp_path: Path):
    image = tmp_path / "a.png"
    imwrite(image, np.zeros((8, 12, 3), np.uint8))
    assert Cv.describe(image).size == ImgSize(12, 8)
    # 权重后缀走模型分支，因此报 ModelNotFoundError 而不是 ImageError
    with pytest.raises(ModelNotFoundError):
        Cv.describe(tmp_path / "nope" / "best.pt")


def test_metrics_properties():
    metrics = Metrics(
        task=Task.DETECT,
        values={
            "metrics/mAP50-95(B)": 0.42,
            "metrics/mAP50(B)": 0.66,
            "top1": 0.9,
        },
    )
    assert metrics.map == 0.42
    assert metrics.map50 == 0.66
    assert metrics.top1 == 0.9
    assert metrics.top5 == 0.0
    assert metrics.get("nope", default=-1) == -1
    assert metrics.to_dict()["task"] == "detect"


# ============================================================
# 真实模型（缺少权重或 ultralytics 时跳过）
# ============================================================


@pytest.fixture(scope="module")
def bus_image() -> Path:
    ultralytics = pytest.importorskip("ultralytics")
    return Path(ultralytics.__file__).parent / "assets" / "bus.jpg"


@NO_RUNTIME
def test_predict_detect_end_to_end(bus_image: Path, tmp_path: Path):
    with Model(WEIGHTS, device="cpu") as model:
        assert model.loaded
        assert model.info().task is Task.DETECT
        assert len(model.names) == 80

        results = model.predict(bus_image, conf=0.3, output_dir=tmp_path)
        assert len(results) == 1

        result = results[0]
        assert result.source == bus_image
        assert len(result.boxes) >= 1
        assert all(isinstance(obj, Box) for obj in result.objs)
        assert all(0.0 <= obj.conf <= 1.0 for obj in result.boxes)
        assert set(result.speed) >= {"preprocess", "inference", "postprocess"}

        # 渲染结果已经落盘
        assert result.path is not None and result.path.is_file()
        assert result.to_dict()["count"] == len(result.boxes)

    assert not model.loaded


@NO_RUNTIME
def test_predict_accepts_numpy(bus_image: Path):
    pytest.importorskip("cv2")
    from cs.cv import imread

    model = Model(WEIGHTS, device="cpu")
    results = model.predict(imread(bus_image), conf=0.5)
    assert len(results) == 1
    assert len(results[0].boxes) >= 1


@NO_RUNTIME
def test_mcp_predict_tool(bus_image: Path, tmp_path: Path):
    fastmcp = pytest.importorskip("fastmcp")
    _ = fastmcp
    from fastmcp import Client

    from cs.cv.mcp_server import create_server

    async def call() -> dict:
        async with Client(create_server()) as client:
            payload = await client.call_tool(
                "cv_predict",
                {
                    "model": str(WEIGHTS),
                    "source": str(bus_image),
                    "device": "cpu",
                    "conf": 0.4,
                    "output_dir": str(tmp_path),
                },
            )
            return payload.data

    data = asyncio.run(call())
    assert data["ok"] is True
    assert data["count"] == 1
    assert data["results"][0]["count"] >= 1


# ============================================================
# ultralytics 结果 → 本包模型（替身对象，不需要 torchvision）
# ============================================================


class _Array:
    """模拟 torch.Tensor 的 ``tolist()``。"""

    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class _Boxes:
    def __init__(self, boxes, conf, cls, ids=None):
        self.xyxy = _Array(boxes)
        self.conf = _Array(conf)
        self.cls = _Array(cls)
        self.id = _Array(ids) if ids is not None else None


class _Masks:
    def __init__(self, xy):
        self.xy = xy


class _Keypoints:
    def __init__(self, xy, conf=None):
        self.xy = _Array(xy)
        self.conf = _Array(conf) if conf is not None else None


class _Obb:
    def __init__(self, corners, conf, cls):
        self.xyxyxyxy = _Array(corners)
        self.conf = _Array(conf)
        self.cls = _Array(cls)


class _Probs:
    def __init__(self, data):
        self.data = _Array(data)


class _Raw:
    """模拟 ultralytics.engine.results.Results 的字段。"""

    def __init__(self, **fields):
        self.path = fields.get("path", "a.jpg")
        self.names = fields.get("names", {0: "cat", 1: "dog"})
        self.speed = fields.get("speed", {"inference": 2.0, "postprocess": 0.5})
        self.orig_img = fields.get("orig_img")
        for name in ("boxes", "masks", "keypoints", "probs", "obb"):
            setattr(self, name, fields.get(name))


def test_to_result_detect():
    model = Model("yolo11n.pt", task="detect")
    result = model._to_result(_Raw(boxes=_Boxes([[10, 20, 30, 40]], [0.8], [1])))
    assert len(result) == 1
    box = result.boxes[0]
    assert box.xyxy == (10, 20, 30, 40)
    assert (box.category_idx, box.conf) == (1, 0.8)
    assert result.names[1] == "dog"
    assert result.source == Path("a.jpg")
    assert result.label(1, 0.8) == "dog: 0.80"
    assert result.to_dict()["speed"]["inference"] == 2.0


def test_to_result_segment():
    model = Model("yolo11n.pt", task="segment")
    raw = _Raw(
        boxes=_Boxes([[0, 0, 10, 10]], [0.9], [0]),
        masks=_Masks([[[0, 0], [10, 0], [10, 10]]]),
    )
    polygon = model._to_result(raw).polygons[0]
    assert isinstance(polygon, Polygon)
    assert len(polygon.points) == 3
    assert polygon.conf == 0.9


def test_to_result_pose():
    model = Model("yolo11n.pt", task="pose")
    raw = _Raw(
        boxes=_Boxes([[0, 0, 20, 30]], [0.7], [0]),
        keypoints=_Keypoints([[[1, 2], [3, 4]]], [[0.9, 0.4]]),
    )
    pose = model._to_result(raw).poses[0]
    assert len(pose.points) == 2
    assert pose.points[0].conf == 0.9
    assert pose.box is not None and pose.box.xyxy == (0, 0, 20, 30)
    assert pose.conf == 0.7


def test_to_result_obb():
    model = Model("yolo11n.pt", task="obb")
    corners = [[0, 0], [10, 0], [10, 5], [0, 5]]
    raw = _Raw(obb=_Obb([corners], [0.6], [1]))
    obb = model._to_result(raw).obbs[0]
    assert obb.size == pytest.approx((10, 5))
    assert obb.angle == pytest.approx(0)
    assert obb.category_idx == 1


def test_to_result_classify():
    model = Model("yolo11n.pt", task="classify")
    raw = _Raw(names={0: "a", 1: "b"}, probs=_Probs([0.2, 0.8]))
    result = model._to_result(raw)
    assert result.objs == []
    assert result.probs is not None
    assert result.probs.top1 == ("b", 0.8)


def test_to_result_keeps_track_id():
    model = Model("yolo11n.pt", task="detect")
    raw = _Raw(boxes=_Boxes([[0, 0, 10, 10]], [0.9], [0], ids=[7]))
    result = model._to_result(raw)
    assert result.track_ids == [7]
    assert result.to_dict()["objs"][0]["track_id"] == 7

    plain = model._to_result(_Raw(boxes=_Boxes([[0, 0, 10, 10]], [0.9], [0])))
    assert plain.track_ids == [-1]
    assert "track_id" not in plain.to_dict()["objs"][0]


def test_to_result_without_predictions():
    model = Model("yolo11n.pt", task="detect")
    result = model._to_result(_Raw(boxes=_Boxes([], [], [])))
    assert len(result) == 0 and result.to_dict()["count"] == 0


def test_geometry_union_covers_all_types():
    members = [Point, Line, Box, Obb, Polygon, Circle, Pose]
    union = str(Geometry.__value__)
    assert all(member.__name__ in union for member in members)
