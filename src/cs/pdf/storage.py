"""路径归一化、原子写入、可选依赖加载等底层设施。"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import IO, Any

from .errors import (
    MissingDependencyError,
    OutputExistsError,
    PDFOpenError,
    PDFPasswordError,
)

__all__ = [
    "PathLike",
    "as_path",
    "atomic_writer",
    "ensure_dir",
    "ensure_input",
    "open_reader",
    "require",
    "write_pdf",
]

type PathLike = str | os.PathLike[str]


def as_path(value: PathLike) -> Path:
    """把 ``str`` / ``Path`` / ``os.PathLike`` 统一为 ``Path`` 并展开 ``~``。"""
    if isinstance(value, Path):
        return value.expanduser()
    if isinstance(value, (str, os.PathLike)):
        return Path(os.fspath(value)).expanduser()
    raise TypeError(f"需要路径（str 或 Path），收到 {type(value).__name__}")


def ensure_input(value: PathLike, *, role: str = "文件") -> Path:
    """校验输入路径存在且是普通文件，返回绝对路径。"""
    path = as_path(value)
    if not path.exists():
        raise FileNotFoundError(f"{role}不存在: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"{role}是目录而非文件: {path}")
    return path.resolve()


def ensure_dir(value: PathLike) -> Path:
    """确保输出目录存在。"""
    path = as_path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def atomic_writer(target: Path, *, overwrite: bool = True) -> Iterator[IO[bytes]]:
    """写入同目录临时文件，成功后原子替换，避免半成品残留。"""
    target = as_path(target)
    if target.exists() and not overwrite:
        raise OutputExistsError(target)
    parent = target.parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)

    handle, temp_name = tempfile.mkstemp(
        dir=parent, prefix=f".{target.name}.", suffix=".part"
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            yield stream
        os.replace(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def write_pdf(writer: Any, target: Path, *, overwrite: bool = True) -> Path:
    """把 ``PdfWriter`` 原子地落盘。"""
    path = as_path(target)
    with atomic_writer(path, overwrite=overwrite) as stream:
        writer.write(stream)
    return path


def open_reader(source: PathLike, password: str | None = None) -> Any:
    """打开 PDF 并完成解密，失败时抛出 :class:`PDFOpenError` / :class:`PDFPasswordError`。"""
    from pypdf import PdfReader

    path = ensure_input(source)
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pypdf 对损坏文件会抛出多种异常
        raise PDFOpenError(path, f"{type(exc).__name__}: {exc}") from exc

    if reader.is_encrypted:
        if not password:
            raise PDFPasswordError(path, "文件已加密，请提供 password")
        try:
            unlocked = bool(reader.decrypt(password))
        except Exception as exc:
            raise PDFPasswordError(path, f"解密失败: {exc}") from exc
        if not unlocked:
            raise PDFPasswordError(path, "密码错误")

    if len(reader.pages) == 0:
        raise PDFOpenError(path, "文档不包含任何页面")
    return reader


def require(module: str, *, extra: str, purpose: str) -> ModuleType:
    """惰性导入可选依赖，缺失时给出可执行的安装提示。"""
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(module, extra=extra, purpose=purpose) from exc
