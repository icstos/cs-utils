"""摘要与密钥派生：哈希、HMAC、PBKDF2、文件指纹、恒等比较。

MD5 / SHA1 仅用于兼容既有数据（如旧系统的校验值），新场景请用 SHA-256 及以上；
密码存储请用 :func:`derive_key`（PBKDF2），不要直接哈希。
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import time
from pathlib import Path
from secrets import token_bytes

from .errors import DigestError
from .models import DerivedKey, HashAlgorithm

__all__ = [
    "DEFAULT_ITERATIONS",
    "derive_bytes",
    "derive_key",
    "file_hash",
    "fingerprint",
    "hash_bytes",
    "hash_text",
    "hmac_bytes",
    "hmac_text",
    "token",
    "verify",
]

#: PBKDF2 默认迭代次数（OWASP 2023 对 PBKDF2-HMAC-SHA256 的建议下限）
DEFAULT_ITERATIONS = 200_000

_CHUNK = 1 << 20
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _coerce(value: str | HashAlgorithm) -> HashAlgorithm:
    try:
        return HashAlgorithm.coerce(value)
    except ValueError as exc:
        raise DigestError(str(exc)) from exc


def hash_text(
    data: str,
    algorithm: str | HashAlgorithm = HashAlgorithm.SHA256,
    *,
    charset: str = "utf-8",
    errors: str = "strict",
    uppercase: bool = False,
) -> str:
    """文本摘要的十六进制串。"""
    return hash_bytes(data.encode(charset, errors), algorithm, uppercase=uppercase)


def hash_bytes(
    data: bytes | bytearray | memoryview,
    algorithm: str | HashAlgorithm = HashAlgorithm.SHA256,
    *,
    uppercase: bool = False,
) -> str:
    """字节摘要的十六进制串。"""
    kind = _coerce(algorithm)
    digest = hashlib.new(kind.value, bytes(data)).hexdigest()
    return digest.upper() if uppercase else digest


def hmac_text(
    data: str,
    key: str | bytes,
    algorithm: str | HashAlgorithm = HashAlgorithm.SHA256,
    *,
    charset: str = "utf-8",
) -> str:
    """带密钥的消息认证码（HMAC）十六进制串。"""
    raw_key = key.encode(charset) if isinstance(key, str) else key
    return hmac_bytes(data.encode(charset), raw_key, algorithm)


def hmac_bytes(
    data: bytes | bytearray | memoryview,
    key: bytes | bytearray | memoryview,
    algorithm: str | HashAlgorithm = HashAlgorithm.SHA256,
) -> str:
    """字节级 HMAC。"""
    if not key:
        raise DigestError("HMAC 的 key 不能为空")
    kind = _coerce(algorithm)
    return hmac.new(bytes(key), bytes(data), kind.value).hexdigest()


def derive_key(
    password: str,
    *,
    salt: str | bytes | None = None,
    iterations: int = DEFAULT_ITERATIONS,
    length: int = 32,
    algorithm: str | HashAlgorithm = HashAlgorithm.SHA256,
) -> DerivedKey:
    """PBKDF2 派生密钥；不传 ``salt`` 时随机生成并随结果返回。

    返回的 ``salt`` / ``iterations`` 必须与密文一起保存，否则无法复现。
    """
    if iterations < 1:
        raise DigestError(f"iterations 必须为正整数，收到 {iterations}")
    if length < 1:
        raise DigestError(f"length 必须为正整数，收到 {length}")
    kind = _coerce(algorithm)
    raw_salt = _salt_bytes(salt)
    raw = derive_bytes(
        password,
        raw_salt,
        iterations=iterations,
        length=length,
        algorithm=kind,
    )
    return DerivedKey(
        key=_b64(raw),
        salt=_b64(raw_salt),
        iterations=iterations,
        algorithm=kind.value,
    )


def derive_bytes(
    password: str,
    salt: bytes,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    length: int = 32,
    algorithm: str | HashAlgorithm = HashAlgorithm.SHA256,
) -> bytes:
    """PBKDF2 原始字节结果。"""
    kind = _coerce(algorithm)
    return hashlib.pbkdf2_hmac(
        kind.value, password.encode("utf-8"), salt, iterations, length
    )


def verify(left: str, right: str) -> bool:
    """恒定时间比较两个摘要/令牌，避免时序侧信道。

    十六进制摘要大小写不同也视为相等（只有两边都是纯十六进制时才放宽）。
    非 ASCII 文本会比较其 UTF-8 字节（``hmac.compare_digest`` 不接受非 ASCII 字符串）。
    """
    if not left or not right:
        return False
    if hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8")):
        return True
    if set(left) <= _HEX_DIGITS and set(right) <= _HEX_DIGITS:
        return hmac.compare_digest(
            left.lower().encode("ascii"), right.lower().encode("ascii")
        )
    return False


def token(
    value: str,
    *,
    salt: str = "",
    timestamp: float | None = None,
    algorithm: str | HashAlgorithm = HashAlgorithm.SHA256,
    length: int | None = None,
) -> str:
    """生成「值 + 盐 + 时间戳」的指纹，用于缓存键或一次性令牌。

    这不是安全令牌：它不含随机性、长度可预测，签名场景请用
    :func:`hmac_text`。
    """
    stamp = int(time.time()) if timestamp is None else int(timestamp)
    digest = hash_text(f"{value}{salt}{stamp}", algorithm)
    if length is not None:
        if length < 1:
            raise DigestError(f"length 必须为正整数，收到 {length}")
        return digest[:length]
    return digest


def file_hash(
    path: str | Path,
    algorithm: str | HashAlgorithm = HashAlgorithm.SHA256,
    *,
    chunk_size: int = _CHUNK,
) -> str:
    """分块计算文件摘要，避免大文件整体读入内存。"""
    target = Path(path)
    if not target.is_file():
        raise DigestError(f"文件不存在：{target}")
    if chunk_size < 1:
        raise DigestError(f"chunk_size 必须为正整数，收到 {chunk_size}")
    kind = _coerce(algorithm)
    digest = hashlib.new(kind.value)
    with target.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(
    text: str,
    *,
    b64: bool = True,
    algorithm: str | HashAlgorithm = HashAlgorithm.MD5,
) -> str:
    """旧系统 ``md5(base64(utf-8))`` 校验值的等价实现（兼容用途）。"""
    raw = text.encode("utf-8")
    if b64:
        raw = base64.b64encode(raw)
    return hash_bytes(raw, algorithm)


def _salt_bytes(salt: str | bytes | None) -> bytes:
    if salt is None:
        return token_bytes(16)
    if isinstance(salt, bytes):
        return salt
    try:
        return base64.b64decode(salt + "=" * (-len(salt) % 4), validate=True)
    except (binascii.Error, ValueError):
        return salt.encode("utf-8")


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")
