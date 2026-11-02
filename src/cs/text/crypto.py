"""对称加解密：Fernet（AES-128-CBC + HMAC-SHA256）之上的一层封装。

两种模式，互斥：

- **密钥模式**：调用方持有 Fernet 密钥（:func:`new_key` 生成或已有密钥），
  密文就是一个标准 Fernet token。
- **口令模式**：只记住口令，盐与迭代次数随密文一起写在信封里，
  下次用一个 ``cs1.`` 前缀的字符串即可解密。

信封格式（口令模式）::

    cs1.<base64url(json)>  →  {"v":1,"kdf":"pbkdf2-sha256","iter":200000,
                              "salt":"...","ct":"<fernet token>"}
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from pathlib import Path
from secrets import token_bytes
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from .digest import DEFAULT_ITERATIONS, derive_bytes
from .errors import CryptoError, DecryptError, SecretError
from .models import HashAlgorithm

__all__ = [
    "decrypt",
    "decrypt_bytes",
    "decrypt_file",
    "encrypt",
    "encrypt_bytes",
    "encrypt_file",
    "key_from",
    "new_key",
    "token_info",
]

_PREFIX = "cs1."
_VERSION = 1
_KDF = "pbkdf2-sha256"


def new_key() -> str:
    """生成一个 Fernet 密钥（urlsafe base64，44 字符）。"""
    return Fernet.generate_key().decode("ascii")


def key_from(material: str | bytes) -> str:
    """把任意口令/字节收敛成 Fernet 密钥。

    已是合法 Fernet 密钥（32 字节 base64）时原样返回，否则按其 SHA-256 派生，
    因此 ``"my-secret"`` 也能直接用。
    """
    raw = material.encode("utf-8") if isinstance(material, str) else bytes(material)
    if not raw:
        raise SecretError("密钥不能为空")
    if len(raw) == 44:
        try:
            if len(base64.urlsafe_b64decode(raw)) == 32:
                return raw.decode("ascii")
        except (binascii.Error, ValueError):
            pass
    return base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode("ascii")


def encrypt(
    text: str,
    *,
    key: str | bytes | None = None,
    password: str | None = None,
    iterations: int = DEFAULT_ITERATIONS,
    charset: str = "utf-8",
) -> str:
    """加密文本，返回可安全放进 JSON/URL 的字符串。"""
    return encrypt_bytes(
        text.encode(charset),
        key=key,
        password=password,
        iterations=iterations,
    )


def encrypt_bytes(
    data: bytes | bytearray | memoryview,
    *,
    key: str | bytes | None = None,
    password: str | None = None,
    iterations: int = DEFAULT_ITERATIONS,
) -> str:
    """加密字节。``key`` 与 ``password`` 必须且只能给一个。"""
    payload = bytes(data)
    if (key is None) == (password is None):
        raise CryptoError("key 与 password 必须且只能提供一个")
    if password is not None:
        return _seal_with_password(payload, password, iterations)
    return Fernet(key_from(key).encode("ascii")).encrypt(payload).decode("ascii")


def decrypt(
    token: str,
    *,
    key: str | bytes | None = None,
    password: str | None = None,
    charset: str = "utf-8",
    errors: str = "strict",
) -> str:
    """解密文本；口令模式需提供 ``password``，密钥模式需提供 ``key``。"""
    return decrypt_bytes(token, key=key, password=password).decode(charset, errors)


def decrypt_bytes(
    token: str,
    *,
    key: str | bytes | None = None,
    password: str | None = None,
) -> bytes:
    """解密字节。"""
    text = token.strip()
    if not text:
        raise DecryptError("密文为空")
    if text.startswith(_PREFIX):
        if password is None:
            raise CryptoError("该密文使用口令加密，请提供 password")
        return _open_with_password(text, password)
    if key is None:
        if password is not None:
            raise CryptoError("该密文使用密钥加密，请提供 key（不是 password）")
        raise CryptoError("解密需要 key 或 password")
    try:
        return Fernet(key_from(key).encode("ascii")).decrypt(text.encode("ascii"))
    except (InvalidToken, ValueError, binascii.Error) as exc:
        raise DecryptError("解密失败：密钥不匹配或密文被篡改") from exc


def encrypt_file(
    path: str | Path,
    *,
    key: str | bytes | None = None,
    password: str | None = None,
    out: str | Path | None = None,
    iterations: int = DEFAULT_ITERATIONS,
    overwrite: bool = False,
) -> Path:
    """加密文件，默认写到 ``<原文件名>.enc``。"""
    source = Path(path)
    if not source.is_file():
        raise CryptoError(f"文件不存在：{source}")
    target = (
        Path(out) if out is not None else source.with_suffix(source.suffix + ".enc")
    )
    if target.exists() and not overwrite:
        raise CryptoError(f"目标已存在：{target}（需显式 overwrite=True）")
    token = encrypt_bytes(
        source.read_bytes(),
        key=key,
        password=password,
        iterations=iterations,
    )
    target.write_text(token, encoding="ascii")
    return target


def decrypt_file(
    path: str | Path,
    *,
    key: str | bytes | None = None,
    password: str | None = None,
    out: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """解密文件；``.enc`` 后缀会被去掉，其他后缀追加 ``.dec``。"""
    source = Path(path)
    if not source.is_file():
        raise CryptoError(f"文件不存在：{source}")
    if out is not None:
        target = Path(out)
    elif source.suffix == ".enc":
        target = source.with_suffix("")
    else:
        target = source.with_name(source.name + ".dec")
    if target.exists() and not overwrite:
        raise CryptoError(f"目标已存在：{target}（需显式 overwrite=True）")
    target.write_bytes(
        decrypt_bytes(source.read_text(encoding="ascii"), key=key, password=password)
    )
    return target


def token_info(token: str) -> dict[str, Any]:
    """读取密文的元信息（不解密）；不泄露任何密钥材料。"""
    text = token.strip()
    if not text:
        raise DecryptError("密文为空")
    if not text.startswith(_PREFIX):
        return {"mode": "key", "version": 0, "algorithm": "fernet"}
    envelope = _unpack(text)
    return {
        "mode": "password",
        "version": envelope["v"],
        "algorithm": "fernet",
        "kdf": envelope["kdf"],
        "iterations": envelope["iter"],
        "salt": envelope["salt"],
    }


def _seal_with_password(data: bytes, password: str, iterations: int) -> str:
    if iterations < 1:
        raise CryptoError(f"iterations 必须为正整数，收到 {iterations}")
    salt = token_bytes(16)
    derived = derive_bytes(
        password,
        salt,
        iterations=iterations,
        length=32,
        algorithm=HashAlgorithm.SHA256,
    )
    token = Fernet(base64.urlsafe_b64encode(derived)).encrypt(data).decode("ascii")
    envelope = {
        "v": _VERSION,
        "kdf": _KDF,
        "iter": iterations,
        "salt": base64.b64encode(salt).decode("ascii"),
        "ct": token,
    }
    packed = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
    return _PREFIX + base64.urlsafe_b64encode(packed).decode("ascii").rstrip("=")


def _open_with_password(token: str, password: str) -> bytes:
    envelope = _unpack(token)
    try:
        salt = base64.b64decode(envelope["salt"], validate=True)
        iterations = int(envelope["iter"])
    except (KeyError, ValueError, binascii.Error) as exc:
        raise DecryptError(f"密文信封损坏：{exc}") from exc
    derived = derive_bytes(
        password,
        salt,
        iterations=iterations,
        length=32,
        algorithm=HashAlgorithm.SHA256,
    )
    try:
        return Fernet(base64.urlsafe_b64encode(derived)).decrypt(
            envelope["ct"].encode("ascii")
        )
    except (InvalidToken, ValueError) as exc:
        raise DecryptError("解密失败：口令错误或密文被篡改") from exc


def _unpack(token: str) -> dict[str, Any]:
    body = token[len(_PREFIX) :]
    try:
        raw = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
        envelope = json.loads(raw)
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        raise DecryptError(f"密文格式无法解析：{exc}") from exc
    if not isinstance(envelope, dict) or "ct" not in envelope:
        raise DecryptError("密文格式无法解析：缺少 ct 字段")
    if int(envelope.get("v", 0)) > _VERSION:
        raise DecryptError(f"密文版本 {envelope.get('v')} 高于当前实现 {_VERSION}")
    return envelope
