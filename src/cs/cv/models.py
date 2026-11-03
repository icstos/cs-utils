"""跨模块共享的数据模型：任务类型、类别表、分类概率、推理结果、图像信息。

这些类型既不依赖 ultralytics，也不依赖 OpenCV —— 只有 :class:`Result` 的两个便捷
方法（``render`` / ``save``）在调用时才用到绘制模块，避免循环导入。
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import numpy as np

from ..my_types import ImgSize, TaskType
from .geometry import Box, Geometry, Obb, Polygon, Pose

__all__ = ["Category", "ImageInfo", "JsonModel", "Probs", "Result", "Task"]


class JsonModel:
    """提供 ``to_dict`` / ``to_json`` 的轻量基类。"""

    __slots__ = ()

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, indent=indent, default=str
        )


# ============================================================
# 任务与类别
# ============================================================


class Task(StrEnum):
    """Ultralytics 的五类视觉任务（值即 API 中的 slug）。"""

    CLASSIFY = "classify"
    DETECT = "detect"
    SEGMENT = "segment"
    POSE = "pose"
    OBB = "obb"

    @classmethod
    def parse(cls, value: Task | str | int | TaskType | None) -> Task:
        """接受 slug、旧枚举 :class:`~cs.my_types.TaskType`、序号或 ``None``（默认检测）。"""
        match value:
            case None:
                return cls.DETECT
            case cls():
                return value
            case TaskType() as legacy:
                return cls[legacy.name]
            case int() as number:
                return cls[TaskType(number).name]
            case str() as text:
                try:
                    return cls(text.strip().lower())
                except ValueError:
                    raise ValueError(
                        f"未知任务 {text!r}，可选：{[item.value for item in cls]}"
                    ) from None
            case _:
                raise ValueError(f"无法识别的任务类型：{value!r}")

    @property
    def task_type(self) -> TaskType:
        """旧版枚举，便于与数据库层（存整数）互操作。"""
        return TaskType[self.name]


@dataclass(frozen=True)
class Category(JsonModel):
    """类别名称表：支持下标、名称、迭代与 ``in`` 判断。"""

    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.names, tuple):
            object.__setattr__(self, "names", tuple(str(name) for name in self.names))

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __getitem__(self, idx: int) -> str:
        return self.names[idx]

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def index(self, name: str) -> int:
        """名称 → 下标，未找到抛 ``ValueError``。"""
        return self.names.index(name)

    def get(self, idx: int, default: str = "") -> str:
        """安全取名称（越界返回 ``default``）。"""
        return self.names[idx] if 0 <= idx < len(self.names) else default

    def to_list(self) -> list[str]:
        return list(self.names)

    def to_dict(self) -> dict[str, Any]:
        return {"names": list(self.names)}

    @classmethod
    def parse(cls, value: Category | Sequence[str] | Mapping[int, str] | None) -> Self:
        """从列表、``{下标: 名称}``（ultralytics 的 ``names``）或 ``None`` 构造。"""
        match value:
            case None:
                return cls()
            case cls():
                return value
            case str() as text:
                raise ValueError(f"类别表需要列表或映射，收到字符串 {text!r}")
            case Mapping() as mapping:
                return cls(tuple(str(mapping[key]) for key in sorted(mapping)))
            case Sequence() as sequence:
                return cls(tuple(str(name) for name in sequence))
            case _:
                raise ValueError(f"无法识别的类别表：{value!r}")


# ============================================================
# 分类概率
# ============================================================


@dataclass(frozen=True)
class Probs(JsonModel):
    """分类任务的概率分布。"""

    names: tuple[str, ...] = ()
    values: tuple[float, ...] = ()

    def __len__(self) -> int:
        return len(self.values)

    @property
    def top1(self) -> tuple[str, float]:
        """最高置信度的 ``(名称, 概率)``；空结果返回 ``("", 0.0)``。"""
        return self.top(1)[0] if self.values else ("", 0.0)

    def top(self, k: int = 5) -> list[tuple[str, float]]:
        """按概率降序取前 ``k`` 项。"""
        if k < 1:
            raise ValueError("k 必须为正整数")
        order = sorted(
            range(len(self.values)), key=lambda i: self.values[i], reverse=True
        )
        return [
            (self.names[i] if i < len(self.names) else str(i), self.values[i])
            for i in order[:k]
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "top": [
                {"name": name, "conf": round(value, 4)} for name, value in self.top(5)
            ],
            "count": len(self.values),
        }


# ============================================================
# 推理结果
# ============================================================


@dataclass
class Result:
    """单张图像的推理结果：图元列表 + 可选分类概率 + 耗时。"""

    source: Path | None = None
    objs: list[Geometry] = field(default_factory=list)
    probs: Probs | None = None
    speed: dict[str, float] = field(default_factory=dict)
    names: Category = field(default_factory=Category)
    image: np.ndarray | None = field(default=None, repr=False)
    path: Path | None = None

    def __len__(self) -> int:
        return len(self.objs)

    def __iter__(self) -> Iterator[Geometry]:
        return iter(self.objs)

    # ---------------------------------------------------------- 分类视图
    @property
    def boxes(self) -> list[Box]:
        return [obj for obj in self.objs if isinstance(obj, Box)]

    @property
    def polygons(self) -> list[Polygon]:
        return [obj for obj in self.objs if isinstance(obj, Polygon)]

    @property
    def obbs(self) -> list[Obb]:
        return [obj for obj in self.objs if isinstance(obj, Obb)]

    @property
    def poses(self) -> list[Pose]:
        return [obj for obj in self.objs if isinstance(obj, Pose)]

    @property
    def track_ids(self) -> list[int]:
        """跟踪场景下的目标 ID（无 ID 的框记 -1）。"""
        return [box.track_id for box in self.boxes]

    @property
    def category_indices(self) -> list[int]:
        return sorted({obj.category_idx for obj in self.objs})

    def label(self, category_idx: int, conf: float = -1.0) -> str:
        """类别名（给出置信度时拼成 ``名称: 0.87``）。"""
        name = self.names.get(category_idx)
        return f"{name}: {conf:.2f}" if name and conf >= 0 else name

    # ---------------------------------------------------------- 变换
    def shifted(self, delta_x: float, delta_y: float) -> Result:
        """整体平移（裁剪推理后还原到原图坐标）。"""
        return self._mapped(offset=(delta_x, delta_y))

    def scaled(self, scale_x: float, scale_y: float) -> Result:
        """整体缩放（回归到原图尺寸）。"""
        return self._mapped(scale=(scale_x, scale_y))

    def _mapped(
        self,
        *,
        offset: tuple[float, float] | None = None,
        scale: tuple[float, float] | None = None,
    ) -> Result:
        def move(obj: Geometry) -> Geometry:
            if scale is not None:
                obj = obj.scale(*scale)
            if offset is not None:
                obj = obj.offset(*offset)
            return obj

        return Result(
            source=self.source,
            objs=[move(obj) for obj in self.objs],
            probs=self.probs,
            speed=self.speed,
            names=self.names,
            image=self.image,
            path=self.path,
        )

    # ---------------------------------------------------------- 渲染
    def render(
        self, *, style: Any = None, category: Category | None = None
    ) -> np.ndarray:
        """把结果画到原图副本上（需要结果里保留 ``image``）。"""
        from .draw import Canvas

        if self.image is None:
            raise ValueError("结果中没有原始图像，无法渲染")
        canvas = Canvas(self.image, style=style, category=category or self.names)
        canvas.draw_all(self.objs)
        return canvas.image

    def save(
        self, target: str | Path, *, style: Any = None, overwrite: bool = False
    ) -> Path:
        """渲染并另存为图片，同时更新 :attr:`path`。"""
        from .draw import imwrite

        self.path = imwrite(target, self.render(style=style), overwrite=overwrite)
        return self.path

    # ---------------------------------------------------------- 序列化
    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source) if self.source else None,
            "count": len(self.objs),
            "objs": [obj.to_dict() for obj in self.objs],
            "probs": self.probs.to_dict() if self.probs else None,
            "speed": {key: round(value, 2) for key, value in self.speed.items()},
            "path": str(self.path) if self.path else None,
        }


# ============================================================
# 图像信息
# ============================================================


@dataclass(frozen=True)
class ImageInfo(JsonModel):
    """图像基本信息。"""

    path: Path
    width: int
    height: int
    channels: int
    dtype: str

    @property
    def is_gray(self) -> bool:
        return self.channels == 1

    @property
    def size(self) -> ImgSize:
        return ImgSize(self.width, self.height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "dtype": self.dtype,
            "gray": self.is_gray,
        }
