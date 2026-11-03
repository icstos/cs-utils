"""计算机视觉模块的异常层级。

与 :mod:`cs.text` / :mod:`cs.env` 一致：领域异常统一继承 :class:`CVError`，
MCP 层再转换成 ``ToolError``，避免 traceback 泄漏给模型。
"""

from __future__ import annotations

__all__ = [
    "CVError",
    "DependencyError",
    "ImageError",
    "ModelError",
    "ModelNotFoundError",
    "ShapeError",
]


class CVError(Exception):
    """本包所有错误的基类。"""


class ImageError(CVError):
    """图像读写、解码或格式问题。"""


class ShapeError(CVError):
    """图元规格或几何参数不合法。"""


class ModelError(CVError):
    """模型加载、推理、训练、验证或导出失败。"""


class ModelNotFoundError(ModelError):
    """权重文件或数据集配置不存在。"""


class DependencyError(CVError):
    """缺少可选依赖（ultralytics / torch / onnxruntime 等）。"""
