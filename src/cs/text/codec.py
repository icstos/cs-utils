"""文本编解码：base64 / base64url / base32 / hex / URL 百分号编码，以及 gzip/zlib 压缩。

解码默认宽松（补齐 base64 缺失的 ``=``、忽略空白），编码默认不换行。
"""

from __future__ import annotations

import base64
import binascii
import gzip
import re
import zlib
from typing import Literal
from urllib.parse import quote, unquote

from .errors import CodecError
from .models import Encoding

__all__ = [
    "compress",
    "decode",
    "decode_bytes",
    "decompress",
    "encode",
    "encode_bytes",
    "guess",
]

_WHITESPACE = re.compile(r"\s+")
_HEX_CHARS = re.compile(r"[0-9a-fA-F]+")
_B32_CHARS = re.compile(r"[A-Z2-7]+=*")

type Algorithm = Literal["gzip", "zlib"]


def encode(
    data: str | bytes | bytearray | memoryview,
    encoding: Encoding | str = Encoding.BASE64,
    *,
    charset: str = "utf-8",
    padding: bool = True,
) -> str:
    """把文本或字节编码成字符串。"""
    raw = data.encode(charset) if isinstance(data, str) else bytes(data)
    return encode_bytes(raw, encoding, charset=charset, padding=padding)


def encode_bytes(
    data: bytes,
    encoding: Encoding | str = Encoding.BASE64,
    *,
    charset: str = "utf-8",
    padding: bool = True,
) -> str:
    """把字节编码成字符串；``charset`` 仅用于 URL 编码。"""
    kind = _as_encoding(encoding)
    match kind:
        case Encoding.BASE64:
            text = base64.b64encode(data).decode("ascii")
        case Encoding.BASE64URL:
            text = base64.urlsafe_b64encode(data).decode("ascii")
        case Encoding.BASE32:
            text = base64.b32encode(data).decode("ascii")
        case Encoding.HEX:
            return data.hex()
        case Encoding.URL:
            return quote(data.decode(charset, "strict"), safe="")
        case Encoding.AUTO:
            raise CodecError("编码方向不支持 auto，请指定具体编码")
    return text if padding else text.rstrip("=")


def decode(
    data: str | bytes,
    encoding: Encoding | str = Encoding.BASE64,
    *,
    charset: str = "utf-8",
    errors: str = "replace",
) -> str:
    """把编码后的字符串还原为文本。"""
    return decode_bytes(data, encoding).decode(charset, errors)


def decode_bytes(
    data: str | bytes,
    encoding: Encoding | str = Encoding.BASE64,
) -> bytes:
    """把编码后的字符串还原为字节；缺失的 ``=`` 会自动补齐。"""
    text = data.decode("ascii", "strict") if isinstance(data, bytes) else data
    text = _WHITESPACE.sub("", text)
    kind = _as_encoding(encoding)
    if kind is Encoding.AUTO:
        candidates = guess(text)
        if not candidates:
            raise CodecError("无法自动识别编码，请显式指定")
        for candidate in candidates:
            try:
                return decode_bytes(text, candidate)
            except CodecError:
                continue
        raise CodecError(f"无法自动识别编码，候选 {candidates} 均解码失败")

    try:
        match kind:
            case Encoding.BASE64:
                return base64.b64decode(_pad(text), validate=True)
            case Encoding.BASE64URL:
                return base64.urlsafe_b64decode(_pad(text))
            case Encoding.BASE32:
                return base64.b32decode(_pad(text, block=8))
            case Encoding.HEX:
                return binascii.unhexlify(text)
            case Encoding.URL:
                return unquote(text).encode("utf-8")
    except (binascii.Error, ValueError) as exc:
        raise CodecError(f"{kind.value} 解码失败：{exc}") from exc
    raise CodecError(f"不支持的解码方式 {kind.value}")


def guess(text: str) -> tuple[Encoding, ...]:
    """猜测文本可能的编码，按可信度从高到低返回（仅供 ``AUTO`` 参考）。"""
    stripped = _WHITESPACE.sub("", text)
    if not stripped:
        return ()
    candidates: list[Encoding] = []
    if len(stripped) % 2 == 0 and _HEX_CHARS.fullmatch(stripped):
        candidates.append(Encoding.HEX)
    if "%" in stripped:
        candidates.append(Encoding.URL)
    if _B32_CHARS.fullmatch(stripped) and len(stripped) % 8 in {0, 2, 4, 5, 7}:
        candidates.append(Encoding.BASE32)
    if "-" in stripped or "_" in stripped:
        candidates.append(Encoding.BASE64URL)
    candidates.append(Encoding.BASE64)
    return tuple(dict.fromkeys(candidates))


def compress(
    data: str | bytes | bytearray | memoryview,
    *,
    algorithm: Algorithm = "gzip",
    level: int = 6,
    charset: str = "utf-8",
) -> bytes:
    """压缩文本或字节；``algorithm`` 取 ``gzip`` / ``zlib``。"""
    if not -1 <= level <= 9:
        raise CodecError(f"压缩级别需在 -1~9 之间，收到 {level}")
    raw = data.encode(charset) if isinstance(data, str) else bytes(data)
    match algorithm:
        case "gzip":
            return gzip.compress(raw, compresslevel=max(level, 1) if level >= 0 else 6)
        case "zlib":
            return zlib.compress(raw, level)
        case _:
            raise CodecError(f"未知压缩算法 {algorithm!r}，可选 gzip / zlib")


def decompress(
    data: bytes,
    *,
    algorithm: Algorithm = "gzip",
    charset: str = "utf-8",
    errors: str = "replace",
) -> str:
    """解压并解码为文本。"""
    try:
        match algorithm:
            case "gzip":
                raw = gzip.decompress(data)
            case "zlib":
                raw = zlib.decompress(data)
            case _:
                raise CodecError(f"未知压缩算法 {algorithm!r}，可选 gzip / zlib")
    except (OSError, zlib.error) as exc:
        raise CodecError(f"{algorithm} 解压失败：{exc}") from exc
    return raw.decode(charset, errors)


def _as_encoding(value: Encoding | str) -> Encoding:
    if isinstance(value, Encoding):
        return value
    key = value.strip().lower().replace("-", "").replace("_", "")
    match key:
        case "base64" | "b64":
            return Encoding.BASE64
        case "base64url" | "b64url" | "urlsafe":
            return Encoding.BASE64URL
        case "base32" | "b32":
            return Encoding.BASE32
        case "hex" | "base16":
            return Encoding.HEX
        case "url" | "percent" | "quote":
            return Encoding.URL
        case "auto":
            return Encoding.AUTO
    allowed = ", ".join(item.value for item in Encoding)
    raise CodecError(f"未知编码 {value!r}，可选：{allowed}")


def _pad(text: str, *, block: int = 4) -> str:
    """补齐块长度；输入已带 ``=`` 或长度已是整数倍时原样返回。"""
    if not text:
        return text
    return text + "=" * (-len(text) % block)
