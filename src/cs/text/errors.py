"""``cs.text`` 的异常层级。

全部继承 :class:`TextError`；属于「入参不合法」的同时继承 ``ValueError``，
方便调用方统一兜底。
"""

from __future__ import annotations

__all__ = [
    "CodecError",
    "CryptoError",
    "DecryptError",
    "DigestError",
    "SecretError",
    "TextError",
    "TransformError",
    "ValidationError",
]


class TextError(Exception):
    """text 模块的异常基类。"""


# ---------------------------------------------------------------- 变换


class TransformError(TextError, ValueError):
    """文本变换参数非法（宽度为负、未知对齐方式、未知归一化形式等）。"""


# ---------------------------------------------------------------- 校验


class ValidationError(TextError, ValueError):
    """校验请求本身不合法（未知规则名、规则参数缺失等）。"""


# ---------------------------------------------------------------- 编解码


class CodecError(TextError, ValueError):
    """编码/解码/压缩失败（非法 base64、未知编码名等）。"""


# ---------------------------------------------------------------- 摘要


class DigestError(TextError, ValueError):
    """摘要计算失败（未知算法、文件不存在等）。"""


# ---------------------------------------------------------------- 加解密


class CryptoError(TextError, ValueError):
    """加解密参数或环境错误（缺少密钥、密钥与密文模式不匹配等）。"""


class DecryptError(CryptoError):
    """密文无法解密（被篡改、密钥或密码错误、格式损坏）。"""


class SecretError(CryptoError):
    """密钥/密码来源缺失或不合法（含环境变量不存在）。"""
