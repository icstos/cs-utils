"""``cs.text`` 的数据模型。

全部是 ``frozen`` + ``slots`` 的 dataclass：创建后不可变、字段紧凑，
``to_dict()`` / ``to_json()`` 可直接交给 MCP 工具或日志。

>>> Chunk(OpTag.REPLACE, 0, 1, 0, 1, "a", "b").to_dict()["tag"]
'replace'
"""

from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "Check",
    "CheckResult",
    "Chunk",
    "DerivedKey",
    "DiffResult",
    "Encoding",
    "Granularity",
    "HashAlgorithm",
    "JsonModel",
    "Match",
    "OpTag",
    "TextLike",
]

#: 入参允许的文本形态（``None`` 由 :func:`cs.text.transform.to_str` 兜底）
type TextLike = str | bytes | bytearray | memoryview


class JsonModel:
    """可序列化为 JSON 对象的最小协议。"""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------- 枚举


class HashAlgorithm(StrEnum):
    """可用的摘要算法；名字与 :mod:`hashlib` 一致。"""

    MD5 = "md5"
    SHA1 = "sha1"
    SHA224 = "sha224"
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    SHA3_256 = "sha3_256"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"

    @classmethod
    def coerce(cls, value: str | HashAlgorithm) -> HashAlgorithm:
        """接受 ``SHA-256`` / ``sha256`` / ``sha3_256`` / 枚举本身。"""
        if isinstance(value, cls):
            return value
        key = value.strip().lower().replace("-", "").replace("_", "")
        aliases = {item.value.replace("_", ""): item for item in cls}
        if key in aliases:
            return aliases[key]
        allowed = ", ".join(item.value for item in cls)
        raise ValueError(f"未知摘要算法 {value!r}，可选：{allowed}")

    @property
    def digest_size(self) -> int:
        """摘要字节数。"""
        return hashlib.new(self.value).digest_size

    @property
    def hex_length(self) -> int:
        """十六进制摘要字符数。"""
        return self.digest_size * 2

    @property
    def weak(self) -> bool:
        """MD5 / SHA1 已不适合做安全用途，仅用于兼容校验。"""
        return self in {HashAlgorithm.MD5, HashAlgorithm.SHA1}


class Encoding(StrEnum):
    """文本与字节串互转时使用的编码方案。"""

    BASE64 = "base64"
    BASE64URL = "base64url"
    BASE32 = "base32"
    HEX = "hex"
    URL = "url"
    AUTO = "auto"


class Granularity(StrEnum):
    """差异比较的粒度。"""

    CHAR = "char"
    WORD = "word"
    LINE = "line"


class OpTag(StrEnum):
    """差异片段的动作，取值与 :meth:`difflib.SequenceMatcher.get_opcodes` 一致。"""

    EQUAL = "equal"
    REPLACE = "replace"
    DELETE = "delete"
    INSERT = "insert"


# ---------------------------------------------------------------- 比较


@dataclass(frozen=True, slots=True)
class Chunk(JsonModel):
    """一段差异：``a[a_start:a_end]`` 与 ``b[b_start:b_end]`` 的对应关系。

    ``a_text`` / ``b_text`` 是切片后的原始文本；若比较时忽略了某些字符，
    这里的文本是剔除忽略字符后的结果。
    """

    tag: OpTag
    a_start: int
    a_end: int
    b_start: int
    b_end: int
    a_text: str = ""
    b_text: str = ""

    @property
    def changed(self) -> bool:
        return self.tag is not OpTag.EQUAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag.value,
            "a": [self.a_start, self.a_end],
            "b": [self.b_start, self.b_end],
            "a_text": self.a_text,
            "b_text": self.b_text,
        }


@dataclass(frozen=True, slots=True)
class DiffResult(JsonModel):
    """一次比较的完整结果。

    ``added`` / ``removed`` 以比较粒度（字符/词/行）计数，``distance`` 是二者之和，
    等价于把 ``a`` 改写为 ``b`` 所需的最小增删量。
    """

    ratio: float
    chunks: tuple[Chunk, ...] = ()
    added: int = 0
    removed: int = 0
    granularity: Granularity = Granularity.CHAR
    ignore: tuple[str, ...] = ()
    case_sensitive: bool = True
    a: str = field(default="", repr=False)
    b: str = field(default="", repr=False)

    @property
    def equal(self) -> bool:
        return not any(chunk.changed for chunk in self.chunks)

    @property
    def changes(self) -> tuple[Chunk, ...]:
        return tuple(chunk for chunk in self.chunks if chunk.changed)

    @property
    def distance(self) -> int:
        return self.added + self.removed

    def unified(
        self, *, fromfile: str = "a", tofile: str = "b", context: int = 3
    ) -> str:
        """按原文行渲染统一差异（``difflib.unified_diff``）。"""
        lines = difflib.unified_diff(
            self.a.splitlines(),
            self.b.splitlines(),
            fromfile=fromfile,
            tofile=tofile,
            n=context,
            lineterm="",
        )
        return "\n".join(lines)

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ratio": round(self.ratio, 4),
            "equal": self.equal,
            "added": self.added,
            "removed": self.removed,
            "distance": self.distance,
            "granularity": self.granularity.value,
            "ignore": list(self.ignore),
            "case_sensitive": self.case_sensitive,
            "chunks": [chunk.to_dict() for chunk in self.changes],
        }
        if include_text:
            payload |= {"a": self.a, "b": self.b}
        return payload

    def __str__(self) -> str:
        if self.equal:
            return "完全一致"
        return (
            f"相似度 {self.ratio:.1%}，新增 {self.added} / 删除 {self.removed} "
            f"{'字符' if self.granularity is Granularity.CHAR else self.granularity.value}"
        )


@dataclass(frozen=True, slots=True)
class Match(JsonModel):
    """候选文本与查询的相似度。"""

    text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "score": round(self.score, 4)}


# ---------------------------------------------------------------- 校验


@dataclass(frozen=True, slots=True)
class Check(JsonModel):
    """单项校验结果。"""

    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}

    def __str__(self) -> str:
        mark = "通过" if self.passed else "不通过"
        return f"{self.name}: {mark}" + (f"（{self.detail}）" if self.detail else "")


@dataclass(frozen=True, slots=True)
class CheckResult(JsonModel):
    """一组校验的汇总结果。"""

    value: str
    checks: tuple[Check, ...] = ()

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    #: 兼容旧 ``Validate.response.valid``
    @property
    def valid(self) -> bool:
        return self.ok

    @property
    def passed(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if check.passed)

    @property
    def failed(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if not check.passed)

    #: 兼容旧 ``Validate.response.errors``
    @property
    def errors(self) -> tuple[str, ...]:
        return self.failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "ok": self.ok,
            "passed": list(self.passed),
            "failed": list(self.failed),
            "checks": [check.to_dict() for check in self.checks],
        }

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        if not self.checks:
            return "未执行任何校验"
        state = "全部通过" if self.ok else f"未通过：{', '.join(self.failed)}"
        return f"{len(self.checks)} 项校验，{state}"


# ---------------------------------------------------------------- 密钥派生


@dataclass(frozen=True, slots=True)
class DerivedKey(JsonModel):
    """PBKDF2 派生结果；``salt`` 与参数需随密文一起保存才能复现。"""

    key: str
    salt: str
    iterations: int
    algorithm: str = HashAlgorithm.SHA256.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "salt": self.salt,
            "iterations": self.iterations,
            "algorithm": self.algorithm,
        }
