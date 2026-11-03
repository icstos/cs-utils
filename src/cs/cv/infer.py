"""Ultralytics 模型封装：加载、预测、跟踪、训练、验证、导出。

ultralytics（含 torch）是**可选**依赖 —— 本模块在没装它的环境里照常导入，
只有真正加载模型时才抛 :class:`~cs.cv.errors.DependencyError` 并给出安装命令。

>>> from cs.cv import Model
>>> Model("yolo11n.pt").predict("bus.jpg")[0].boxes[:2]
[Box([...], conf=0.87, cat=5), ...]
>>> Model("yolo11n.pt").train("coco8.yaml", epochs=10, log="train.log")
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from .errors import DependencyError, ModelError, ModelNotFoundError
from .geometry import Box, Obb, Point, Polygon, Pose
from .models import Category, JsonModel, Probs, Result, Task

__all__ = [
    "FORMATS",
    "FormatInfo",
    "Metrics",
    "Model",
    "ModelFormat",
    "ModelInfo",
    "TrainResult",
    "devices",
    "format_of",
    "load",
    "predict",
    "resolve_device",
]

type StrPath = str | Path


# ============================================================
# 可选依赖与设备
# ============================================================


def _import_ultralytics() -> Any:
    try:
        import ultralytics
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise DependencyError(
            "需要 ultralytics：pip install ultralytics（或 pip install 'cs[cv]'）"
        ) from exc
    return ultralytics


def _auto_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device: str | int | None = None) -> str:
    """归一化设备写法：``None``/``"auto"`` 自动挑选，``0`` 等价 ``"cuda:0"``。"""
    match device:
        case None | "" | "auto":
            return _auto_device()
        case int() as index:
            return f"cuda:{index}"
        case str() as text:
            cleaned = text.strip().lower()
            if cleaned.isdigit():
                return f"cuda:{cleaned}"
            if (
                cleaned in {"cpu", "mps", "cuda"}
                or cleaned.startswith("cuda:")
                or "," in cleaned
            ):
                return cleaned
            raise ModelError(
                f"无法识别的设备：{device!r}（可用 cpu / mps / cuda:0 / 0）"
            )
        case _:
            raise ModelError(f"无法识别的设备：{device!r}")


def devices() -> list[str]:
    """当前机器上可选的计算设备。"""
    available = ["cpu"]
    try:
        import torch
    except ImportError:
        return available
    if torch.cuda.is_available():
        available.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        available.append("mps")
    return available


# ============================================================
# 模型格式
# ============================================================


class ModelFormat(StrEnum):
    """ultralytics 的导出格式（值与 ``export(format=...)`` 一致）。"""

    PT = "pt"
    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"
    OPENVINO = "openvino"
    ENGINE = "engine"


@dataclass(frozen=True)
class FormatInfo(JsonModel):
    """导出格式说明：参数名、产物后缀、可用平台。"""

    format: ModelFormat
    name: str
    suffix: str
    cpu: bool
    gpu: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format.value,
            "name": self.name,
            "suffix": self.suffix,
            "cpu": self.cpu,
            "gpu": self.gpu,
        }


#: 常用导出格式
FORMATS: tuple[FormatInfo, ...] = (
    FormatInfo(ModelFormat.PT, "PyTorch", ".pt", True, True),
    FormatInfo(ModelFormat.TORCHSCRIPT, "TorchScript", ".torchscript", True, True),
    FormatInfo(ModelFormat.ONNX, "ONNX", ".onnx", True, True),
    FormatInfo(ModelFormat.OPENVINO, "OpenVINO", "_openvino_model", True, False),
    FormatInfo(ModelFormat.ENGINE, "TensorRT", ".engine", False, True),
)

_SUFFIX_FORMATS = {info.suffix: info for info in FORMATS}


def format_of(source: StrPath) -> FormatInfo | None:
    """按权重路径推断格式（目录形式如 ``xxx_openvino_model`` 也能识别）。"""
    text = str(source)
    for suffix, info in _SUFFIX_FORMATS.items():
        if text.endswith(suffix):
            return info
    return None


# ============================================================
# 结果模型
# ============================================================


@dataclass(frozen=True)
class ModelInfo(JsonModel):
    """模型概要。"""

    path: Path
    task: Task
    names: Category
    device: str
    format: FormatInfo | None = None
    params: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "task": self.task.value,
            "device": self.device,
            "format": self.format.format.value if self.format else None,
            "params": self.params,
            "classes": len(self.names),
            "names": self.names.to_list(),
        }


@dataclass(frozen=True)
class Metrics(JsonModel):
    """验证 / 训练产出的指标集合（键名沿用 ultralytics 的 ``results_dict``）。"""

    task: Task
    values: dict[str, float] = field(default_factory=dict)

    def get(self, *keys: str, default: float = 0.0) -> float:
        """按顺序返回第一个存在的指标。"""
        for key in keys:
            if key in self.values:
                return self.values[key]
        return default

    @property
    def map(self) -> float:
        """mAP@0.5:0.95（分割/姿态/OBB 任务回退到对应后缀）。"""
        return self.get(
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95(M)",
            "metrics/mAP50-95(P)",
            "metrics/mAP50-95",
        )

    @property
    def map50(self) -> float:
        return self.get(
            "metrics/mAP50(B)", "metrics/mAP50(M)", "metrics/mAP50(P)", "metrics/mAP50"
        )

    @property
    def top1(self) -> float:
        return self.get("top1", "metrics/accuracy_top1")

    @property
    def top5(self) -> float:
        return self.get("top5", "metrics/accuracy_top5")

    def to_dict(self) -> dict[str, Any]:
        return {"task": self.task.value, **self.values}


@dataclass(frozen=True)
class TrainResult(JsonModel):
    """训练产出。"""

    weights: Path
    save_dir: Path
    epochs: int
    metrics: Metrics | None = None
    log: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": str(self.weights),
            "save_dir": str(self.save_dir),
            "epochs": self.epochs,
            "log": str(self.log) if self.log else None,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


# ============================================================
# 模型
# ============================================================


@dataclass
class Model:
    """一个 YOLO 模型：懒加载，可反复预测，结束时释放显存。

    ``path`` 可以是本地权重、导出的目录（``*_openvino_model``），
    也可以是官方模型名（``yolo11n.pt``，首次使用会自动下载）。
    """

    path: StrPath = "yolo11n.pt"
    task: Task | str | None = None
    device: str | None = None
    imgsz: int = 640
    verbose: bool = False

    _yolo: Any = field(default=None, init=False, repr=False)
    _names: Category = field(default_factory=Category, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.task is not None:
            self.task = Task.parse(self.task)

    # ---------------------------------------------------------- 生命周期
    @property
    def loaded(self) -> bool:
        return self._yolo is not None

    @property
    def names(self) -> Category:
        """类别名称表（未加载时为空）。"""
        return self._names

    def load(self) -> Self:
        """加载模型（幂等）。"""
        if self._yolo is not None:
            return self
        if not self.path.exists() and len(self.path.parts) > 1:
            raise ModelNotFoundError(f"权重文件不存在：{self.path}")
        ultralytics = _import_ultralytics()
        try:
            self._yolo = ultralytics.YOLO(
                str(self.path),
                task=self.task.value if self.task else None,
                verbose=self.verbose,
            )
        except Exception as exc:  # 第三方库异常类型不稳定，统一包装
            raise ModelError(f"加载模型失败（{self.path}）：{exc}") from exc
        self.device = resolve_device(self.device)
        self._names = Category.parse(getattr(self._yolo, "names", None))
        if self.task is None:
            self.task = Task.parse(getattr(self._yolo, "task", None))
        return self

    def unload(self) -> Self:
        """释放模型与显存。"""
        self._yolo = None
        self._names = Category()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return self

    def __enter__(self) -> Self:
        return self.load()

    def __exit__(self, *_: object) -> None:
        self.unload()

    # ---------------------------------------------------------- 信息
    def info(self) -> ModelInfo:
        """模型概要：任务、类别、设备、格式与参数量。"""
        self.load()
        return ModelInfo(
            path=self.path,
            task=Task.parse(self.task),
            names=self._names,
            device=resolve_device(self.device),
            format=format_of(self.path),
            params=self._param_count(),
        )

    def _param_count(self) -> int:
        inner = getattr(self._yolo, "model", None)
        try:
            return int(sum(parameter.numel() for parameter in inner.parameters()))
        except (AttributeError, TypeError):
            return 0

    # ---------------------------------------------------------- 推理
    def predict(
        self,
        source: Any,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int | None = None,
        max_det: int = 300,
        classes: list[int] | None = None,
        agnostic_nms: bool = False,
        retina_masks: bool = False,
        output_dir: StrPath | None = None,
        style: Any = None,
        overwrite: bool = False,
        **extra: Any,
    ) -> list[Result]:
        """对图片 / 目录 / 视频 / numpy 数组做推理。

        ``output_dir`` 不为空时，把每张结果渲染并保存到该目录（同名 ``.jpg``）。
        其余未知参数原样透传给 ultralytics（如 ``augment``、``vid_stride``）。
        """
        self.load()
        options = {
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz or self.imgsz,
            "max_det": max_det,
            "classes": classes,
            "agnostic_nms": agnostic_nms,
            "retina_masks": retina_masks,
            "device": resolve_device(self.device),
            "verbose": self.verbose,
            **extra,
        }
        try:
            raw = self._yolo.predict(source=source, **options)
        except Exception as exc:
            raise ModelError(f"推理失败：{exc}") from exc
        return self._collect(
            raw, output_dir=output_dir, style=style, overwrite=overwrite
        )

    def track(
        self,
        source: Any,
        *,
        tracker: str = "bytetrack.yaml",
        persist: bool = True,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int | None = None,
        output_dir: StrPath | None = None,
        style: Any = None,
        overwrite: bool = False,
        **extra: Any,
    ) -> list[Result]:
        """多目标跟踪（视频逐帧返回；跟踪 ID 存在 ``Box.track_id``，-1 表示无）。"""
        self.load()
        options = {
            "tracker": tracker,
            "persist": persist,
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz or self.imgsz,
            "device": resolve_device(self.device),
            "verbose": self.verbose,
            **extra,
        }
        try:
            raw = self._yolo.track(source=source, **options)
        except Exception as exc:
            raise ModelError(f"跟踪失败：{exc}") from exc
        return self._collect(
            raw, output_dir=output_dir, style=style, overwrite=overwrite
        )

    # ---------------------------------------------------------- 训练 / 验证 / 导出
    def train(
        self,
        data: StrPath,
        *,
        epochs: int = 100,
        patience: int = 50,
        batch: int = -1,
        imgsz: int | None = None,
        device: str | None = None,
        workers: int = 8,
        project: StrPath = "runs/train",
        name: str = "exp",
        exist_ok: bool = False,
        resume: bool = False,
        seed: int = 0,
        log: StrPath | None = None,
        **extra: Any,
    ) -> TrainResult:
        """训练模型，``log`` 给出时把 ultralytics 日志同时落盘。"""
        self.load()
        options = {
            "data": str(data),
            "epochs": epochs,
            "patience": patience,
            "batch": batch,
            "imgsz": imgsz or self.imgsz,
            "device": resolve_device(device or self.device),
            "workers": workers,
            "project": str(project),
            "name": name,
            "exist_ok": exist_ok,
            "resume": resume,
            "seed": seed,
            **extra,
        }
        with _log_to(log):
            try:
                trained = self._yolo.train(**options)
            except Exception as exc:
                raise ModelError(f"训练失败：{exc}") from exc
        save_dir = Path(
            str(getattr(getattr(self._yolo, "trainer", None), "save_dir", project))
        )
        return TrainResult(
            weights=save_dir / "weights" / "best.pt",
            save_dir=save_dir,
            epochs=epochs,
            metrics=_metrics(trained, Task.parse(self.task)),
            log=Path(log) if log else None,
        )

    def val(
        self,
        data: StrPath | None = None,
        *,
        split: str = "val",
        batch: int = 16,
        conf: float = 0.001,
        iou: float = 0.6,
        imgsz: int | None = None,
        device: str | None = None,
        plots: bool = False,
        **extra: Any,
    ) -> Metrics:
        """在验证集上评估，返回 mAP / 准确率等指标。"""
        self.load()
        options = {
            "data": str(data) if data is not None else None,
            "split": split,
            "batch": batch,
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz or self.imgsz,
            "device": resolve_device(device or self.device),
            "plots": plots,
            **extra,
        }
        try:
            raw = self._yolo.val(**{k: v for k, v in options.items() if v is not None})
        except Exception as exc:
            raise ModelError(f"验证失败：{exc}") from exc
        metrics = _metrics(raw, Task.parse(self.task))
        if metrics is None:
            raise ModelError("验证未返回任何指标")
        return metrics

    def export(
        self,
        fmt: ModelFormat | str = ModelFormat.ONNX,
        *,
        imgsz: int | None = None,
        half: bool = False,
        int8: bool = False,
        dynamic: bool = False,
        simplify: bool = True,
        opset: int | None = None,
        workspace: float | None = None,
        **extra: Any,
    ) -> Path:
        """把 ``.pt`` 导出为部署格式，返回产物路径。"""
        self.load()
        options = {
            "format": fmt.value if isinstance(fmt, ModelFormat) else str(fmt),
            "imgsz": imgsz or self.imgsz,
            "half": half,
            "int8": int8,
            "dynamic": dynamic,
            "simplify": simplify,
            "opset": opset,
            "workspace": workspace,
            "device": resolve_device(self.device),
            **extra,
        }
        try:
            exported = self._yolo.export(
                **{k: v for k, v in options.items() if v is not None}
            )
        except Exception as exc:
            raise ModelError(f"导出失败（{fmt}）：{exc}") from exc
        if not exported:
            raise ModelError(f"导出失败（{fmt}）：未返回产物路径")
        return Path(str(exported))

    # ---------------------------------------------------------- 结果转换
    def _collect(
        self,
        raw: Any,
        *,
        output_dir: StrPath | None = None,
        style: Any = None,
        overwrite: bool = False,
    ) -> list[Result]:
        results = [self._to_result(item) for item in raw]
        if output_dir is not None:
            target_dir = Path(output_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            used: set[str] = set()
            for index, result in enumerate(results):
                stem = result.source.stem if result.source else f"result_{index}"
                if stem in used:  # 视频 / 流：同一来源有多帧，追加序号避免互相覆盖
                    stem = f"{stem}_{index:05d}"
                used.add(stem)
                result.save(
                    target_dir / f"{stem}.jpg", style=style, overwrite=overwrite
                )
        return results

    def _to_result(self, raw: Any) -> Result:
        task = Task.parse(self.task)
        result = Result(
            source=Path(raw.path) if getattr(raw, "path", None) else None,
            names=Category.parse(getattr(raw, "names", None)) or self._names,
            speed=dict(getattr(raw, "speed", None) or {}),
            image=getattr(raw, "orig_img", None),
        )
        match task:
            case Task.DETECT:
                result.objs = list(_boxes(raw))
            case Task.SEGMENT:
                result.objs = list(_polygons(raw))
            case Task.OBB:
                result.objs = list(_obbs(raw))
            case Task.POSE:
                result.objs = list(_poses(raw))
            case Task.CLASSIFY:
                result.probs = _probs(raw, result.names)
        return result


# ============================================================
# ultralytics 结果 → 本包数据模型
# ============================================================


def _boxes(raw: Any) -> Iterator[Box]:
    boxes = getattr(raw, "boxes", None)
    if boxes is None:
        return
    # 跟踪模式下 ultralytics 会额外给出 boxes.id
    tracks = boxes.id.tolist() if getattr(boxes, "id", None) is not None else []
    for index, (coords, conf, category) in enumerate(
        zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=False)
    ):
        yield Box.from_xyxy(
            *coords,
            category_idx=int(category),
            conf=float(conf),
            track_id=int(tracks[index]) if index < len(tracks) else -1,
        )


def _polygons(raw: Any) -> Iterator[Polygon]:
    masks = getattr(raw, "masks", None)
    if masks is None:
        return
    boxes = getattr(raw, "boxes", None)
    confs = boxes.conf.tolist() if boxes is not None else []
    classes = boxes.cls.tolist() if boxes is not None else []
    for index, contour in enumerate(masks.xy):
        yield Polygon(
            points=tuple(Point(float(x), float(y)) for x, y in contour),
            category_idx=int(classes[index]) if index < len(classes) else 0,
            conf=float(confs[index]) if index < len(confs) else -1.0,
        )


def _obbs(raw: Any) -> Iterator[Obb]:
    obb = getattr(raw, "obb", None)
    if obb is None:
        return
    for corners, conf, category in zip(
        obb.xyxyxyxy.tolist(), obb.conf.tolist(), obb.cls.tolist(), strict=False
    ):
        yield Obb.from_corners(
            [Point(float(x), float(y)) for x, y in corners],
            category_idx=int(category),
            conf=float(conf),
        )


def _poses(raw: Any) -> Iterator[Pose]:
    keypoints = getattr(raw, "keypoints", None)
    if keypoints is None:
        return
    boxes = getattr(raw, "boxes", None)
    classes = boxes.cls.tolist() if boxes is not None else []
    confs = boxes.conf.tolist() if boxes is not None else []
    frames = boxes.xyxy.tolist() if boxes is not None else []
    confidences = (
        keypoints.conf.tolist()
        if getattr(keypoints, "conf", None) is not None
        else None
    )
    for index, instance in enumerate(keypoints.xy.tolist()):
        points = tuple(
            Point(
                float(x),
                float(y),
                conf=(
                    float(confidences[index][order])
                    if confidences is not None
                    else -1.0
                ),
            )
            for order, (x, y) in enumerate(instance)
        )
        category = int(classes[index]) if index < len(classes) else 0
        conf = float(confs[index]) if index < len(confs) else -1.0
        box = (
            Box.from_xyxy(*frames[index], category_idx=category, conf=conf)
            if index < len(frames)
            else None
        )
        yield Pose(points=points, category_idx=category, conf=conf, box=box)


def _probs(raw: Any, names: Category) -> Probs | None:
    probs = getattr(raw, "probs", None)
    if probs is None:
        return None
    values = probs.data.tolist()
    return Probs(names=names.names, values=tuple(float(value) for value in values))


def _metrics(raw: Any, task: Task) -> Metrics | None:
    """从 ultralytics 的 metrics 对象里抽取指标（键名保持原样）。"""
    if raw is None:
        return None
    values: dict[str, float] = {}
    if isinstance(results := getattr(raw, "results_dict", None), dict):
        values.update({str(key): float(value) for key, value in results.items()})
    for key in ("top1", "top5"):
        if (value := getattr(raw, key, None)) is not None:
            values[key] = float(value)
    return Metrics(task=task, values=values) if values else None


# ============================================================
# 日志与快捷函数
# ============================================================


@contextmanager
def _log_to(target: StrPath | None) -> Iterator[None]:
    """把 ultralytics 的日志同时写入文件（训练/导出耗时较长，便于回溯）。"""
    if target is None:
        yield
        return
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger = logging.getLogger("ultralytics")
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()


def load(path: StrPath = "yolo11n.pt", **options: Any) -> Model:
    """加载模型（等价于 ``Model(path, **options).load()``）。"""
    return Model(path, **options).load()


def predict(model: Model | StrPath, source: Any, **options: Any) -> list[Result]:
    """一次性推理：可用模型实例或权重路径。"""
    return (model if isinstance(model, Model) else Model(model)).predict(
        source, **options
    )
