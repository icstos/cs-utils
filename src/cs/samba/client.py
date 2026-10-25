"""高层客户端：把 SMB 原语组合成日常操作。

:class:`SMBClient` 是 :mod:`cs.samba` 的主入口，负责所有"业务语义"——
路径校验、递归、覆盖策略、续传、断点、统计——而把协议细节交给
:class:`cs.samba.transport.SmbprotocolBackend`。

两条用法::

    from cs.samba import SMBClient

    # 1) 类方法：一次调用拿到已连接的客户端
    client = SMBClient.open(
        server="nas", share="share", username="u", password="p"
    )

    # 2) 上下文管理：退出时自动断开
    with SMBClient.open(server="nas", share="share", username="u", password="p") as c:
        for entry in c.list_dir("docs"):
            print(entry.path, entry.mtime_local)
        c.download("docs/report.pdf", "report.pdf")
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Self

from .backend import OpenMode, SmbBackend
from .errors import (
    SambaError,
    SambaExistsError,
    SambaIsADirectoryError,
    SambaNotADirectoryError,
    SambaNotFoundError,
    SambaOperationError,
)
from .models import (
    ServerInfo,
    SMBEntry,
    SMBOptions,
    StrPath,
    TransferResult,
    as_utc,
    human_size,
)
from .paths import PathSpec, RemotePath
from .transport import CHUNK, SmbprotocolBackend

__all__ = ["ProgressHook", "SMBClient", "SortKey"]

#: 传输进度回调：``(已传输字节, 总字节)``
type ProgressHook = Callable[[int, int], None]
#: 目录排序方式
type SortKey = Literal["name", "size", "mtime"] | None

#: 递归遍历的硬上限，防御 reparse point / 软链接成环
_MAX_DEPTH = 64


class SMBClient:
    """一次 SMB 会话（一条 TCP 连接 + 一个共享）。

    * 构造后默认立即连接（``auto_connect=False`` 可推迟到首次调用）；
    * 支持 ``with`` 自动关闭；连接在实例内部串行化，可跨线程复用，
      但高并发场景建议每个线程一个实例；
    * 路径参数统一接受 ``a/b``、``a\\b``、``\\\\server\\share\\a`` 三种写法，
      统一解析成共享内相对路径。
    """

    def __init__(
        self,
        options: SMBOptions,
        *,
        backend: SmbBackend | None = None,
        auto_connect: bool = True,
        default_path: str = "",
    ) -> None:
        self.options = options
        self.default_path = RemotePath.parse(default_path).posix
        self._backend: SmbBackend = backend or SmbprotocolBackend(options)
        if auto_connect:
            self.connect()

    # ------------------------------------------------------------ 构造
    @classmethod
    def open(
        cls,
        options: SMBOptions | None = None,
        *,
        auto_connect: bool = True,
        **kwargs: Any,
    ) -> Self:
        """类方法构造器：接受 :class:`SMBOptions` 或关键字参数，默认立即连接。"""
        if options is None:
            options = SMBOptions(**kwargs)
        elif kwargs:
            options = options.replace(**kwargs)
        return cls(options, auto_connect=auto_connect)

    @classmethod
    def from_url(cls, url: str, *, auto_connect: bool = True, **overrides: Any) -> Self:
        """从 ``smb://[DOMAIN;]user:pass@host:445/share/sub/dir`` 构造。

        URL 中的子路径会记在 :attr:`default_path`，不改变方法的路径语义。
        """
        options, sub_path = SMBOptions.from_url(url, **overrides)
        return cls(options, auto_connect=auto_connect, default_path=sub_path)

    @classmethod
    def from_env(cls, *, auto_connect: bool = True, **overrides: Any) -> Self:
        """从 ``SMB_HOST`` / ``SMB_SHARE`` / ``SMB_USER`` / ``SMB_PASSWORD`` 构造。"""
        return cls(SMBOptions.from_env(**overrides), auto_connect=auto_connect)

    @classmethod
    def with_backend(cls, options: SMBOptions, backend: SmbBackend) -> Self:
        """注入自定义后端（测试、或换成别的 SMB 实现）。"""
        return cls(options, backend=backend, auto_connect=False)

    # ------------------------------------------------------------ 连接
    @property
    def connected(self) -> bool:
        return self._backend.connected

    @property
    def share(self) -> str:
        return self.options.share

    @property
    def server(self) -> str:
        return self.options.server

    def __repr__(self) -> str:
        state = "connected" if self.connected else "disconnected"
        return f"<{type(self).__name__} {self.options.display} {state}>"

    def __enter__(self) -> Self:
        if not self.connected:
            self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def connect(self) -> ServerInfo:
        """建立连接（带指数退避重试）；认证失败不重试。"""
        attempts = self.options.retries + 1
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                return self._backend.connect()
            except SambaError as exc:
                last = exc
                # 认证/配置错重试也不会变好，直接抛出
                if not isinstance(exc, (ConnectionError, TimeoutError)):
                    raise
                if attempt + 1 < attempts:
                    time.sleep(self.options.backoff * (attempt + 1))
        assert last is not None
        raise last

    def close(self) -> None:
        """断开连接（可重复调用）。"""
        self._backend.close()

    def reconnect(self) -> ServerInfo:
        """断开后重连，用于网络抖动后的自愈。"""
        self.close()
        return self.connect()

    def info(self) -> ServerInfo:
        """当前连接信息（未连接时先连接）。"""
        return self._backend.connect()

    @property
    def unc(self) -> str:
        return self.options.unc

    def use_share(self, share: str) -> Self:
        """切换到同服务器上的另一个共享，返回自身便于链式调用。"""
        self.options = self.options.with_share(share)
        self._backend.use_share(share)
        return self

    # ------------------------------------------------------------ 路径
    def resolve(self, path: PathSpec = "") -> RemotePath:
        """把任意写法解析成共享内相对路径（不做存在性检查）。"""
        return RemotePath.parse(path, server=self.server, share=self.share)

    # ------------------------------------------------------------ 属性读取
    def stat(self, path: PathSpec = "") -> SMBEntry:
        """读取文件/目录属性（大小、最后修改时间、属性位）。"""
        return self._backend.stat(self.resolve(path))

    def exists(self, path: PathSpec) -> bool:
        """路径是否存在（文件或目录）。"""
        try:
            self.stat(path)
        except SambaNotFoundError:
            return False
        return True

    def is_file(self, path: PathSpec) -> bool:
        try:
            return self.stat(path).is_file
        except SambaNotFoundError:
            return False

    def is_dir(self, path: PathSpec) -> bool:
        try:
            return self.stat(path).is_dir
        except SambaNotFoundError:
            return False

    def mtime(self, path: PathSpec) -> datetime:
        """获取文件最后更新时间（UTC，带时区）。"""
        entry = self.stat(path)
        if entry.mtime is None:
            raise SambaOperationError(self.resolve(path).posix, "服务端未提供修改时间")
        return entry.mtime

    def age(self, path: PathSpec, *, now: datetime | None = None) -> timedelta:
        """距最后更新时间过了多久。"""
        entry = self.stat(path)
        age = entry.age(now=now)
        if age is None:
            raise SambaOperationError(self.resolve(path).posix, "服务端未提供修改时间")
        return age

    def size(self, path: PathSpec) -> int:
        """文件大小（字节）；目录返回 0。"""
        entry = self.stat(path)
        return 0 if entry.is_dir else entry.size

    # ------------------------------------------------------------ 目录浏览
    def list_dir(
        self,
        path: PathSpec = "",
        *,
        pattern: str | None = None,
        include_hidden: bool = False,
        sort: SortKey = "name",
    ) -> list[SMBEntry]:
        """列出目录直接子项。

        :param pattern: 只保留名字匹配该 glob 的条目（``*.pdf``）
        :param include_hidden: 是否包含点开头/带 HIDDEN 属性的条目
        :param sort: ``name``（默认，目录在前）/ ``size`` / ``mtime`` / ``None``
        """
        entries = self._backend.list_dir(self.resolve(path))
        if not include_hidden:
            entries = [entry for entry in entries if not entry.is_hidden]
        if pattern:
            wanted = pattern.casefold()
            entries = [
                entry
                for entry in entries
                if fnmatch.fnmatchcase(entry.name.casefold(), wanted)
            ]
        match sort:
            case "name":
                entries.sort(key=lambda e: (not e.is_dir, e.name.casefold()))
            case "size":
                entries.sort(key=lambda e: (not e.is_dir, e.size), reverse=False)
            case "mtime":
                entries.sort(key=lambda e: (not e.is_dir, e.mtime or 0), reverse=True)
            case None:
                pass
            case _:
                raise ValueError(f"不支持的排序方式：{sort!r}")
        return entries

    def iter_entries(
        self,
        path: PathSpec = "",
        *,
        recursive: bool = True,
        max_depth: int | None = None,
        include_hidden: bool = False,
    ) -> Iterator[SMBEntry]:
        """深度优先遍历，逐条产出（目录本身先于其内容产出）。"""
        root = self.resolve(path)
        if not recursive:
            limit = 0
        elif max_depth is None:
            limit = _MAX_DEPTH
        else:
            limit = min(max_depth, _MAX_DEPTH)

        def walk(current: RemotePath, depth: int) -> Iterator[SMBEntry]:
            for entry in self.list_dir(current, include_hidden=include_hidden):
                yield entry
                if entry.is_dir and depth < limit:
                    yield from walk(RemotePath(entry.path), depth + 1)

        yield from walk(root, 0)

    def find(
        self,
        pattern: str,
        path: PathSpec = "",
        *,
        recursive: bool = True,
        include_dirs: bool = False,
        include_hidden: bool = False,
    ) -> list[SMBEntry]:
        """按文件名 glob 查找（``*.log``、``report-2024-??.pdf``）。"""
        wanted = pattern.casefold()
        hits: list[SMBEntry] = []
        for entry in self.iter_entries(
            path, recursive=recursive, include_hidden=include_hidden
        ):
            if entry.is_dir and not include_dirs:
                continue
            if fnmatch.fnmatchcase(entry.name.casefold(), wanted):
                hits.append(entry)
        return hits

    def du(self, path: PathSpec = "") -> int:
        """递归统计占用字节数。"""
        total = 0
        for entry in self.iter_entries(path):
            if entry.is_file:
                total += entry.size
        return total

    # ------------------------------------------------------------ 上传下载
    def download(
        self,
        remote: PathSpec,
        local: StrPath,
        *,
        overwrite: bool = False,
        resume: bool = False,
        verify: bool = False,
        progress: ProgressHook | None = None,
    ) -> TransferResult:
        """下载文件到本地。

        本地写入先落到 ``.part`` 临时文件再原子替换，失败不会留下半个文件；
        ``resume=True`` 时从本地已有长度续传（目标文件保留并追加）。
        """
        source = self.resolve(remote)
        entry = self.stat(source)
        if entry.is_dir:
            raise SambaIsADirectoryError(source.posix)

        target = Path(os.fspath(local)).expanduser()
        if target.is_dir():
            target = target / (entry.name or "download.bin")
        target.parent.mkdir(parents=True, exist_ok=True)

        resumed = 0
        if target.exists():
            if not target.is_file():
                raise SambaIsADirectoryError(str(target))
            existing = target.stat().st_size
            if resume and existing <= entry.size:
                resumed = existing
            elif not overwrite:
                raise SambaExistsError(str(target))

        started = time.perf_counter()
        temp = target.with_name(f".{target.name}.{os.getpid()}.part")
        handle = self._backend.open_read(source)
        try:
            if resumed:
                # 先复制已有前缀，剩下的字节追加到临时文件，最后整体原子替换
                shutil.copyfile(target, temp)
            handle.seek(resumed)
            with open(temp, "ab" if resumed else "wb") as sink:
                written = _pump(handle, sink, entry.size, resumed, progress)
                sink.flush()
                os.fsync(sink.fileno())
            os.replace(temp, target)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        finally:
            handle.close()

        if verify:
            actual = target.stat().st_size
            if actual != entry.size:
                raise SambaOperationError(
                    str(target),
                    f"下载校验失败：期望 {entry.size} 字节，实际 {actual} 字节",
                )
        return TransferResult(
            action="download",
            source=source.posix,
            target=str(target),
            size=written,
            seconds=time.perf_counter() - started,
            resumed_bytes=resumed,
        )

    def upload(
        self,
        local: StrPath,
        remote: PathSpec,
        *,
        overwrite: bool = False,
        resume: bool = False,
        verify: bool = False,
        mkdirs: bool = True,
        progress: ProgressHook | None = None,
    ) -> TransferResult:
        """上传本地文件到共享。

        :param mkdirs: 远端父目录不存在时自动补建（多级）
        :param resume: 远端已有部分内容时从断点续传（按字节数对齐）
        """
        source = Path(os.fspath(local)).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"本地文件不存在: {source}")
        if source.is_dir():
            raise SambaIsADirectoryError(str(source))

        target = self.resolve(remote)
        if target.is_root or str(remote).endswith(("/", "\\")) or self.is_dir(target):
            target = target.child(source.name)

        total = source.stat().st_size
        offset = 0
        mode: OpenMode = "create"
        if resume and self.exists(target):
            offset = self.size(target)
            if offset > total:
                offset, mode = 0, "overwrite"
            elif offset == total:
                return TransferResult(
                    action="upload",
                    source=str(source),
                    target=target.posix,
                    size=0,
                    seconds=0.0,
                    resumed_bytes=offset,
                )
            else:
                mode = "append"
        elif overwrite:
            mode = "overwrite"
        elif self.exists(target):
            raise SambaExistsError(target.posix)

        if mkdirs and not target.parent.is_root:
            self.mkdir(target.parent, parents=True, exist_ok=True)

        started = time.perf_counter()
        with source.open("rb") as stream:
            stream.seek(offset)
            handle = self._backend.open_write(target, mode=mode)
            try:
                written = _pump(stream, handle, total, offset, progress)
                handle.flush()
            finally:
                handle.close()

        if verify:
            actual = self.size(target)
            if actual != total:
                raise SambaOperationError(
                    target.posix,
                    f"上传校验失败：期望 {total} 字节，实际 {actual} 字节",
                )
        return TransferResult(
            action="upload",
            source=str(source),
            target=target.posix,
            size=written,
            seconds=time.perf_counter() - started,
            resumed_bytes=offset,
        )

    def upload_dir(
        self,
        local: StrPath,
        remote: PathSpec = "",
        *,
        overwrite: bool = False,
        pattern: str | None = None,
    ) -> list[TransferResult]:
        """递归上传目录，保留相对层级。"""
        root = Path(os.fspath(local)).expanduser()
        if not root.is_dir():
            raise SambaNotADirectoryError(str(root))
        base = self.resolve(remote).join(root.name)
        self.mkdir(base, parents=True, exist_ok=True)
        results: list[TransferResult] = []
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            target = base.join(*relative.parts)
            if path.is_dir():
                self.mkdir(target, parents=True, exist_ok=True)
                continue
            if pattern and not fnmatch.fnmatch(
                path.name.casefold(), pattern.casefold()
            ):
                continue
            results.append(self.upload(path, target, overwrite=overwrite))
        return results

    def download_dir(
        self,
        remote: PathSpec,
        local: StrPath,
        *,
        overwrite: bool = False,
        pattern: str | None = None,
    ) -> list[TransferResult]:
        """递归下载目录，保留相对层级。"""
        source = self.resolve(remote)
        if not self.stat(source).is_dir:
            raise SambaNotADirectoryError(source.posix)
        root = Path(os.fspath(local)).expanduser() / (source.name or ".")
        results: list[TransferResult] = []
        for entry in self.iter_entries(source):
            relative = RemotePath(entry.path).parts[len(source.parts) :]
            target = root.joinpath(*relative)
            if entry.is_dir:
                target.mkdir(parents=True, exist_ok=True)
                continue
            if pattern and not fnmatch.fnmatch(
                entry.name.casefold(), pattern.casefold()
            ):
                continue
            results.append(
                self.download(entry.path, target, overwrite=overwrite, verify=False)
            )
        return results

    def copy(
        self,
        source: PathSpec,
        target: PathSpec,
        *,
        overwrite: bool = False,
        chunk_size: int = CHUNK,
    ) -> TransferResult:
        """共享内复制（数据经客户端内存中转，不落本地磁盘）。"""
        src = self.resolve(source)
        dst = self.resolve(target)
        entry = self.stat(src)
        if entry.is_dir:
            raise SambaIsADirectoryError(src.posix)
        if self.exists(dst) and not overwrite:
            raise SambaExistsError(dst.posix)

        started = time.perf_counter()
        written = 0
        reader = self._backend.open_read(src)
        try:
            writer = self._backend.open_write(
                dst, mode="overwrite" if overwrite else "create"
            )
            try:
                while True:
                    chunk = reader.read(chunk_size)
                    if not chunk:
                        break
                    writer.write(chunk)
                    written += len(chunk)
                writer.flush()
            finally:
                writer.close()
        finally:
            reader.close()
        return TransferResult(
            action="copy",
            source=src.posix,
            target=dst.posix,
            size=written,
            seconds=time.perf_counter() - started,
        )

    # ------------------------------------------------------------ 文本 / 字节
    def read_bytes(self, path: PathSpec, *, max_bytes: int | None = None) -> bytes:
        """读取整个文件（``max_bytes`` 用于防止误读超大文件）。"""
        entry = self.stat(path)
        if entry.is_dir:
            raise SambaIsADirectoryError(entry.path)
        limit = entry.size if max_bytes is None else min(max_bytes, entry.size)
        handle = self._backend.open_read(self.resolve(path))
        try:
            return handle.read(limit)
        finally:
            handle.close()

    def read_text(
        self,
        path: PathSpec,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
        max_bytes: int | None = None,
    ) -> str:
        """读取文本文件（自动去掉 UTF-8 BOM）。"""
        data = self.read_bytes(path, max_bytes=max_bytes)
        return data.decode(encoding, errors).lstrip("\ufeff")

    def write_bytes(
        self, path: PathSpec, data: bytes, *, overwrite: bool = True
    ) -> int:
        """写入字节（返回写入长度）。"""
        target = self.resolve(path)
        if target.is_root:
            raise SambaIsADirectoryError(self.unc)
        if not overwrite and self.exists(target):
            raise SambaExistsError(target.posix)
        if not target.parent.is_root:
            self.mkdir(target.parent, parents=True, exist_ok=True)
        handle = self._backend.open_write(
            target, mode="overwrite" if overwrite else "create"
        )
        try:
            handle.write(data)
            handle.flush()
        finally:
            handle.close()
        return len(data)

    def write_text(
        self,
        path: PathSpec,
        text: str,
        *,
        encoding: str = "utf-8",
        newline: str | None = None,
        overwrite: bool = True,
    ) -> int:
        """写入文本（``newline`` 可显式指定换行符，便于写 Windows 风格文件）。"""
        payload = text if newline is None else text.replace("\n", newline)
        return self.write_bytes(path, payload.encode(encoding), overwrite=overwrite)

    # ------------------------------------------------------------ 修改
    def mkdir(
        self, path: PathSpec, *, parents: bool = True, exist_ok: bool = True
    ) -> int:
        """创建目录，返回本次新建的层数。

        逐层创建（SMB 没有 ``mkdir -p``），已存在且 ``exist_ok=True`` 时跳过。
        """
        target = self.resolve(path)
        if target.is_root:
            return 0

        missing: list[RemotePath] = []
        cursor = target
        while not cursor.is_root:
            try:
                found = self.stat(cursor)
            except SambaNotFoundError:
                missing.append(cursor)
                cursor = cursor.parent
                continue
            if not found.is_dir:
                raise SambaNotADirectoryError(cursor.posix)
            break

        if not missing:
            if not exist_ok:
                raise SambaExistsError(target.posix)
            return 0
        if not parents and len(missing) > 1:
            raise SambaNotFoundError(
                missing[-1].parent.posix, "父目录不存在（parents=False 时不会自动创建）"
            )
        created = 0
        for item in reversed(missing):
            self._backend.mkdir(item)
            created += 1
        return created

    def remove(self, path: PathSpec, *, missing_ok: bool = False) -> bool:
        """删除文件；返回是否真的删除了。"""
        target = self.resolve(path)
        try:
            entry = self.stat(target)
        except SambaNotFoundError:
            if missing_ok:
                return False
            raise
        if entry.is_dir:
            raise SambaIsADirectoryError(target.posix)
        try:
            self._backend.remove(target)
        except SambaNotFoundError:
            if not missing_ok:
                raise
            return False
        return True

    def rmdir(
        self,
        path: PathSpec,
        *,
        recursive: bool = False,
        missing_ok: bool = False,
    ) -> int:
        """删除目录，返回删除的条目总数。

        ``recursive=True`` 时先自底向上清空（目录非空时服务端会拒绝）。
        """
        target = self.resolve(path)
        if target.is_root:
            raise SambaOperationError(self.unc, "不能删除共享根目录")
        try:
            entry = self.stat(target)
        except SambaNotFoundError:
            if missing_ok:
                return 0
            raise
        if not entry.is_dir:
            raise SambaNotADirectoryError(target.posix)

        removed = 0
        if recursive:
            children = list(self.iter_entries(target))
            for item in reversed(children):
                if item.is_dir:
                    self._backend.rmdir(RemotePath(item.path))
                else:
                    self._backend.remove(RemotePath(item.path))
                removed += 1
        self._backend.rmdir(target)
        return removed + 1

    def rename(
        self, source: PathSpec, target: PathSpec, *, overwrite: bool = False
    ) -> RemotePath:
        """重命名/移动（同一共享内），返回目标路径。"""
        src = self.resolve(source)
        dst = self.resolve(target)
        if src.is_root:
            raise SambaOperationError(self.unc, "不能重命名共享根目录")
        if dst.is_root:
            raise SambaOperationError(self.unc, "目标不能是共享根目录")
        if self.exists(dst) and not overwrite:
            raise SambaExistsError(dst.posix)
        self._backend.rename(src, dst, overwrite=overwrite)
        return dst

    def move(
        self, source: PathSpec, target: PathSpec, *, overwrite: bool = False
    ) -> RemotePath:
        """``rename`` 的别名，语义更贴近"移动"。"""
        return self.rename(source, target, overwrite=overwrite)

    def touch(self, path: PathSpec, when: datetime | None = None) -> RemotePath:
        """把最后修改时间改为指定时间（默认当前时间）。"""
        target = self.resolve(path)
        self.stat(target)  # 不存在时报错，避免静默成功
        self._backend.set_mtime(target, as_utc(when) or datetime.now().astimezone())
        return target

    def ensure_dir(self, path: PathSpec) -> RemotePath:
        """确保目录存在（等价 ``mkdir -p``）。"""
        target = self.resolve(path)
        self.mkdir(target, parents=True, exist_ok=True)
        return target

    # ------------------------------------------------------------ 批量辅助
    def latest(self, path: PathSpec = "", *, pattern: str | None = None) -> SMBEntry:
        """取目录下最近修改的文件，常用于"同步最新产物"。"""
        candidates = [
            entry
            for entry in self.list_dir(path, pattern=pattern)
            if entry.is_file and entry.mtime is not None
        ]
        if not candidates:
            raise SambaNotFoundError(self.resolve(path).posix, "目录下没有匹配的文件")
        return max(candidates, key=lambda entry: entry.mtime)  # type: ignore[arg-type,return-value]

    def summary(self, path: PathSpec = "") -> str:
        """目录概览：文件数、目录数、总大小，适合直接打印给人看。"""
        files = dirs = total = 0
        for entry in self.iter_entries(path):
            if entry.is_dir:
                dirs += 1
            else:
                files += 1
                total += entry.size
        where = self.resolve(path).posix or "/"
        return f"{self.unc}/{where}: {files} 个文件, {dirs} 个目录, 合计 {human_size(total)}"


# ---------------------------------------------------------------- 传输助手
def _pump(
    source: Any,
    sink: Any,
    total: int,
    offset: int,
    progress: ProgressHook | None,
) -> int:
    """在源与目标之间搬运数据，返回本次搬运的字节数。"""
    moved = 0
    while True:
        chunk = source.read(CHUNK)
        if not chunk:
            break
        sink.write(chunk)
        moved += len(chunk)
        if progress is not None:
            progress(offset + moved, total)
    return moved
