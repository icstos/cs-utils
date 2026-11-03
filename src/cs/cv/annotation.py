"""YOLO 标注文件（``labels/*.txt``）与数据集目录的读写、统计与比对。

标签格式见 :mod:`cs.cv.geometry` 的模块文档；分类任务不使用标签文件
（YOLO 用目录结构表示类别），因此在 :class:`AnnoImg` 中直接拒绝。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from ..my_types import ImgSize
from .errors import CVError
from .geometry import Box, Geometry, Obb, Polygon, Pose
from .models import Category, JsonModel, Task

__all__ = ["AnnoDataset", "AnnoImg", "CompareReport", "DatasetSummary", "image_size"]

type StrPath = str | Path

#: 训练/标注常见的图像后缀
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})

#: 每种任务对应的标签行解析器
_PARSERS: dict[Task, Callable[[str, ImgSize], Geometry]] = {
    Task.DETECT: Box.from_yolo,
    Task.SEGMENT: Polygon.from_yolo,
    Task.OBB: Obb.from_yolo,
    Task.POSE: Pose.from_yolo,
}


def image_size(source: StrPath) -> ImgSize:
    """读取图像宽高（只读文件头，不解码像素）。"""
    from PIL import Image

    with Image.open(source) as image:
        return ImgSize(*image.size)


# ============================================================
# 单张图像的标注
# ============================================================


@dataclass
class AnnoImg:
    """一张图像的标注集合，与一个 YOLO 标签文件一一对应。"""

    label: Path | None = None
    img_size: ImgSize = field(default_factory=ImgSize)
    task: Task = Task.DETECT
    objs: list[Geometry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.objs)

    def __iter__(self) -> Iterator[Geometry]:
        return iter(self.objs)

    # ---------------------------------------------------------- 读取
    @classmethod
    def parse(
        cls, text: str, img_size: ImgSize, task: Task | str = Task.DETECT
    ) -> Self:
        """解析标签文件内容。"""
        resolved = Task.parse(task)
        parser = _PARSERS.get(resolved)
        if parser is None:
            raise CVError(f"{resolved.value} 任务的标签不是行文本，无法从文件解析")
        objs = [parser(line, img_size) for line in text.splitlines() if line.strip()]
        return cls(img_size=img_size, task=resolved, objs=objs)

    @classmethod
    def load(
        cls,
        label: StrPath,
        img_size: ImgSize | None = None,
        task: Task | str = Task.DETECT,
    ) -> Self:
        """从磁盘读取标签；``img_size`` 省略时按同名图像推断。"""
        path = Path(label)
        if not path.is_file():
            raise CVError(f"标签文件不存在：{path}")
        size = img_size or _guess_size(path)
        anno = cls.parse(path.read_text(encoding="utf-8"), size, task)
        anno.label = path
        return anno

    # ---------------------------------------------------------- 写出
    def to_yolo(self, *, with_conf: bool = False) -> str:
        """导出为 YOLO 文本（不含结尾换行）。"""
        if not self.objs:
            return ""
        if not self.img_size.width > 0 or not self.img_size.height > 0:
            raise CVError("导出标签需要已知的图像尺寸")
        lines: list[str] = []
        for obj in self.objs:
            if not obj.validate(self.img_size):
                continue
            # 姿态行用可见性标记代替置信度，因此不接受 with_conf
            match obj:
                case Pose():
                    lines.append(obj.to_yolo(self.img_size))
                case _:
                    lines.append(obj.to_yolo(self.img_size, with_conf=with_conf))
        return "\n".join(lines)

    def save(self, target: StrPath | None = None, *, with_conf: bool = False) -> Path:
        """写入标签文件（默认写回 :attr:`label`）。"""
        path = Path(target) if target is not None else self.label
        if path is None:
            raise CVError("未指定标签文件路径")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yolo(with_conf=with_conf), encoding="utf-8")
        return path

    # ---------------------------------------------------------- 工具
    def filter(self, *, min_conf: float = 0.0) -> list[Geometry]:
        """按置信度过滤（用于剔除低分预测）。"""
        return [obj for obj in self.objs if obj.conf >= min_conf]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": str(self.label) if self.label else None,
            "task": self.task.value,
            "size": [self.img_size.width, self.img_size.height],
            "count": len(self.objs),
        }


def _guess_size(label: Path) -> ImgSize:
    """按 ``labels/a.txt`` → ``images/a.jpg`` 的约定查找同名图像。"""
    for parent in label.parents:
        images = parent / "images"
        if not images.is_dir():
            continue
        for suffix in IMAGE_SUFFIXES:
            candidate = images / f"{label.stem}{suffix}"
            if candidate.is_file():
                return image_size(candidate)
    raise CVError(f"无法推断图像尺寸，请显式传入 img_size：{label}")


# ============================================================
# 数据集
# ============================================================


@dataclass(frozen=True)
class DatasetSummary(JsonModel):
    """数据集统计。"""

    task: Task
    images: int
    labels: int
    objects: int
    categories: dict[str, int] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        """有标签的图像占比。"""
        return self.labels / self.images if self.images else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "images": self.images,
            "labels": self.labels,
            "objects": self.objects,
            "coverage": round(self.coverage, 4),
            "categories": self.categories,
        }


@dataclass(frozen=True)
class CompareReport(JsonModel):
    """两个标注数据集（通常是人工标注 vs 模型预测）的比对结果。"""

    gt: int
    pred: int
    matched: int
    iou: float

    @property
    def missed(self) -> int:
        return self.gt - self.matched

    @property
    def extra(self) -> int:
        return self.pred - self.matched

    @property
    def recall(self) -> float:
        return self.matched / self.gt if self.gt else 0.0

    @property
    def precision(self) -> float:
        return self.matched / self.pred if self.pred else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "gt": self.gt,
            "pred": self.pred,
            "matched": self.matched,
            "missed": self.missed,
            "extra": self.extra,
            "iou": self.iou,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
        }


@dataclass
class AnnoDataset:
    """图像目录 + 标签目录构成的 YOLO 数据集。"""

    img_dir: Path
    anno_dir: Path
    task: Task | str = Task.DETECT
    category: Category = field(default_factory=Category)
    items: dict[str, AnnoImg | None] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.img_dir = Path(self.img_dir)
        self.anno_dir = Path(self.anno_dir)
        self.task = Task.parse(self.task)
        if not self.img_dir.is_dir():
            raise CVError(f"图像目录不存在：{self.img_dir}")
        self.scan()

    # ---------------------------------------------------------- 扫描
    def scan(self) -> Self:
        """重新扫描目录（标签缺失的图像记为 ``None``）。"""
        self.items = {}
        for image in sorted(self.img_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = self.anno_dir / f"{image.stem}.txt"
            self.items[image.name] = (
                AnnoImg.load(label, image_size(image), self.task)
                if label.is_file()
                else None
            )
        return self

    # ---------------------------------------------------------- 视图
    @property
    def images(self) -> list[str]:
        return list(self.items)

    @property
    def labels(self) -> list[str]:
        return [name for name, anno in self.items.items() if anno is not None]

    @property
    def objs(self) -> list[Geometry]:
        return [
            obj for anno in self.items.values() if anno is not None for obj in anno.objs
        ]

    def summary(self) -> DatasetSummary:
        """图像 / 标签 / 目标数量与各类别分布。"""
        counts: dict[str, int] = {}
        for obj in self.objs:
            name = self.category.get(obj.category_idx) or str(obj.category_idx)
            counts[name] = counts.get(name, 0) + 1
        return DatasetSummary(
            task=self.task,
            images=len(self.items),
            labels=len(self.labels),
            objects=len(self.objs),
            categories=dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        )

    # ---------------------------------------------------------- 比对
    def compare(
        self, other: AnnoDataset, *, iou: float = 0.5, by_category: bool = True
    ) -> CompareReport:
        """逐图贪心匹配（默认同类 + IoU ≥ 阈值），用于评估召回与误检。"""
        matched = total_gt = 0
        for name, anno in self.items.items():
            if anno is None:
                continue
            predicted = other.items.get(name)
            pool = [obj.to_box() for obj in (predicted.objs if predicted else [])]
            used: set[int] = set()
            for gt in anno.objs:
                total_gt += 1
                hit = _match(gt.to_box(), pool, used, iou, by_category)
                if hit is None:
                    continue
                used.add(hit)
                matched += 1
        total_pred = sum(len(anno.objs) for anno in other.items.values() if anno)
        return CompareReport(gt=total_gt, pred=total_pred, matched=matched, iou=iou)


def _match(
    gt: Box, pool: list[Box], used: set[int], threshold: float, by_category: bool
) -> int | None:
    """在未使用的候选框里找 IoU 最高且过阈值的那个。"""
    best, best_iou = None, threshold
    for index, box in enumerate(pool):
        if index in used:
            continue
        if by_category and box.category_idx != gt.category_idx:
            continue
        score = gt.iou(box)
        if score >= best_iou:
            best, best_iou = index, score
    return best
