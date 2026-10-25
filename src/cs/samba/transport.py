"""基于 ``smbprotocol`` 的 SMB2/3 后端实现。

这是唯一直接接触 SMB2 协议的模块，承担三件事：

1. **连接生命周期**：TCP → 协商 → 会话 → 挂载共享，逐层显式关闭；
2. **异常翻译**：把 ``SMBResponseException`` 家族翻译成 :mod:`cs.samba.errors`
   里的领域异常，让调用方凭 ``FileNotFoundError`` / ``PermissionError`` 就能处理；
3. **句柄抽象**：把 SMB2 的"偏移量读写"包装成 ``io.RawIOBase`` 文件对象，
   于是上传下载可以直接复用 ``shutil.copyfileobj`` 的语义。

线程安全：内部一把 ``RLock`` 串行化所有 SMB 调用（SMB2 会话本身是有状态的，
并发复用同一连接需要排队；要真并发就开多个客户端）。
"""

from __future__ import annotations

import io
import threading
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, BinaryIO

from smbprotocol.connection import Connection
from smbprotocol.exceptions import (
    AccessDenied,
    BadNetworkName,
    CannotDelete,
    DeletePending,
    DirectoryNotEmpty,
    DiskFull,
    FileIsADirectory,
    InsufficientResources,
    IOTimeout,
    LogonFailure,
    NetworkNameDelegated,
    NoSuchFile,
    NotSupported,
    NtStatus,
    ObjectNameCollision,
    ObjectNameNotFound,
    ObjectPathNotFound,
    PasswordExpired,
    PrivilegeNotHeld,
    ServerUnavailable,
    SharingViolation,
    SMBConnectionClosed,
    SMBException,
    SMBResponseException,
    SMBUnsupportedFeature,
    StatusNotADirectory,
    WrongPassword,
)
from smbprotocol.file_info import (
    FileBasicInformation,
    FileInformationClass,
    FileRenameInformation,
    InfoType,
)
from smbprotocol.open import (
    CreateDisposition,
    CreateOptions,
    DirectoryAccessMask,
    FileAttributes,
    FilePipePrinterAccessMask,
    ImpersonationLevel,
    Open,
    QueryDirectoryFlags,
    ShareAccess,
    SMB2SetInfoRequest,
)
from smbprotocol.session import Session
from smbprotocol.tree import TreeConnect

from .errors import (
    SambaAccessError,
    SambaAuthError,
    SambaConfigError,
    SambaConnectionError,
    SambaError,
    SambaExistsError,
    SambaIsADirectoryError,
    SambaNotADirectoryError,
    SambaNotFoundError,
    SambaOperationError,
    SambaProtocolError,
    SambaShareError,
    SambaTimeoutError,
    SambaUnsupportedError,
)
from .models import (
    FILE_ATTRIBUTE,
    ServerInfo,
    SMBEntry,
    SMBOptions,
    as_utc,
    dialect_name,
)
from .paths import RemotePath

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["RemoteFile", "SmbprotocolBackend"]

#: 单次 SMB2 READ/WRITE 的最大载荷
CHUNK = 1024 * 1024
#: 目录枚举的翻页上限，防御性保护（正常目录远小于这个数）
_MAX_PAGES = 4096
#: FILETIME 纪元：1601-01-01 UTC
_FILETIME_EPOCH = datetime(1601, 1, 1, tzinfo=UTC)
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

_ACCESS = FilePipePrinterAccessMask
_DIR_ACCESS = DirectoryAccessMask
_SHARE_ALL = (
    ShareAccess.FILE_SHARE_READ
    | ShareAccess.FILE_SHARE_WRITE
    | ShareAccess.FILE_SHARE_DELETE
)

# 异常分组：按语义归类，比逐个 status 判断更耐版本升级
_AUTH_ERRORS = (LogonFailure, WrongPassword, PasswordExpired)
_NOT_FOUND_ERRORS = (ObjectNameNotFound, ObjectPathNotFound, NoSuchFile)
_SHARE_ERRORS = (BadNetworkName, NetworkNameDelegated)
_CONNECTION_ERRORS = (SMBConnectionClosed, ServerUnavailable)
#: 权限类：ACL 拒绝，或缺少所需特权（如创建目录时的 SE_RESTORE 类特权）
_ACCESS_ERRORS = (AccessDenied, PrivilegeNotHeld)
#: 操作类：容量/占用/删除受阻
_OPERATION_ERRORS = (
    DiskFull,
    InsufficientResources,
    SharingViolation,
    CannotDelete,
    DeletePending,
)
#: 目录枚举的结束状态（服务端用 Status 而非异常类型表达"没有更多"）
_STATUS_NO_MORE_FILES = NtStatus.STATUS_NO_MORE_FILES
#: 表示"这个信息类我不认"的状态码，可据此降级重试
_UNSUPPORTED_INFO_STATUS = frozenset(
    {
        NtStatus.STATUS_INVALID_INFO_CLASS,
        NtStatus.STATUS_INVALID_PARAMETER,
        NtStatus.STATUS_INVALID_DEVICE_REQUEST,
        NtStatus.STATUS_NOT_SUPPORTED,
    }
)
#: 目录枚举信息类的降级顺序（信息量从多到少）
_INFO_CLASSES = (
    FileInformationClass.FILE_ID_BOTH_DIRECTORY_INFORMATION,
    FileInformationClass.FILE_FULL_DIRECTORY_INFORMATION,
    FileInformationClass.FILE_DIRECTORY_INFORMATION,
)


def _unsupported_info_class(exc: BaseException) -> bool:
    return getattr(exc, "status", None) in _UNSUPPORTED_INFO_STATUS


# ---------------------------------------------------------------- 工具函数
def to_filetime(value: datetime) -> int:
    """``datetime`` → FILETIME（100ns 自 1601-01-01）。"""
    moment = as_utc(value) or datetime.now(UTC)
    delta = moment - _FILETIME_EPOCH
    return delta.days * 86_400 * 10**7 + delta.seconds * 10**7 + delta.microseconds * 10


def from_filetime(value: int) -> datetime | None:
    """FILETIME → ``datetime``；``0`` 与 ``0xFFFFFFFFFFFFFFFF`` 表示"未设置"。"""
    if value in (0, 0xFFFFFFFFFFFFFFFF):
        return None
    return datetime.fromtimestamp(value / 10**7 - 11_644_473_600, tz=UTC)


def _describe(exc: BaseException) -> str:
    text = getattr(exc, "message", None) or str(exc) or type(exc).__name__
    return " ".join(text.split())


def _translate(
    exc: BaseException, options: SMBOptions, *, path: str | None, action: str
) -> SambaError:
    """把 smbprotocol 的异常翻译成领域异常（返回而非抛出，便于 ``raise ... from``）。"""
    label = path if path is not None else options.unc
    if isinstance(exc, _AUTH_ERRORS):
        return SambaAuthError(options.server, options.effective_user, _describe(exc))
    if isinstance(exc, _ACCESS_ERRORS):
        return SambaAccessError(label, "拒绝访问（检查共享权限与文件 ACL）")
    if isinstance(exc, _SHARE_ERRORS):
        return SambaShareError(options.server, options.share, _describe(exc))
    if isinstance(exc, _NOT_FOUND_ERRORS):
        return SambaNotFoundError(label, f"{action}失败，路径不存在")
    if isinstance(exc, ObjectNameCollision):
        return SambaExistsError(label)
    if isinstance(exc, DirectoryNotEmpty):
        return SambaOperationError(label, "目录非空（删除目录请用 recursive=True）")
    if isinstance(exc, FileIsADirectory):
        return SambaIsADirectoryError(label)
    if isinstance(exc, StatusNotADirectory):
        return SambaNotADirectoryError(label)
    if isinstance(exc, (NotSupported, SMBUnsupportedFeature)):
        return SambaUnsupportedError(f"{action} 不受支持（{label}）：{_describe(exc)}")
    if isinstance(exc, _OPERATION_ERRORS):
        return SambaOperationError(label, _describe(exc))
    if isinstance(exc, (TimeoutError, IOTimeout)):
        return SambaTimeoutError(options.server, options.port, _describe(exc))
    if isinstance(exc, _CONNECTION_ERRORS):
        return SambaConnectionError(options.server, options.port, _describe(exc))
    if isinstance(exc, OSError) and not isinstance(exc, SMBException):
        return SambaConnectionError(options.server, options.port, _describe(exc))
    return SambaProtocolError(f"{action} 失败（{label}）：{_describe(exc)}")


def _session_error(exc: BaseException, options: SMBOptions) -> SambaError:
    """会话建立失败：认证阶段掉线按认证错误上报。

    Windows 拒绝登录时经常不回 ``STATUS_LOGON_FAILURE``，而是直接把 TCP 掐了
    （``socket aborted by peer`` / WSAESHUTDOWN）。这种"认证阶段中断"在现实中
    几乎只有两类原因：凭据不对，或账号被策略/锁定挡住。报成认证失败比笼统的
    协议错误更准确，也能让上层跳过无意义的重试。
    """
    translated = _translate(exc, options, path=None, action="建立会话")
    if isinstance(translated, SambaTimeoutError):
        return translated
    if isinstance(translated, (SambaConnectionError, SambaProtocolError)):
        return SambaAuthError(
            options.server,
            options.effective_user,
            "服务端在认证阶段中断连接：凭据错误、账号被策略限制，"
            f"或该端口并非 SMB 服务（{_describe(exc)}）",
        )
    return translated


def _connect_error(exc: BaseException, options: SMBOptions) -> SambaError:
    """TCP 层失败：区分超时与其它错误，给出更可操作的提示。"""
    detail = _describe(exc)
    lowered = detail.lower()
    if isinstance(exc, TimeoutError) or "timed out" in lowered or "timeout" in lowered:
        return SambaTimeoutError(options.server, options.port, detail)
    if "getaddrinfo" in lowered or "name or service not known" in lowered:
        return SambaConnectionError(
            options.server, options.port, f"主机名无法解析（{detail}）"
        )
    if "refused" in lowered:
        return SambaConnectionError(
            options.server, options.port, "端口未监听或服务未启动（连接被拒绝）"
        )
    return SambaConnectionError(options.server, options.port, detail)


# ---------------------------------------------------------------- 文件对象
class RemoteFile(io.RawIOBase):
    """把 SMB2 的偏移读写包装成 ``io.RawIOBase``。

    支持 ``read`` / ``readinto`` / ``write`` / ``seek`` / ``tell``，
    因此 ``shutil.copyfileobj``、``json.load``、``csv.reader`` 都能直接吃。
    """

    def __init__(
        self,
        backend: SmbprotocolBackend,
        handle: Open,
        *,
        readable: bool,
        writable: bool,
        size: int,
        position: int = 0,
    ) -> None:
        super().__init__()
        self._backend = backend
        self._handle = handle
        self._readable = readable
        self._writable = writable
        self._size = size
        self._pos = position

    # -- io 接口
    def _require_open(self) -> None:
        """等价于标准库的 ``_checkClosed``，但显式实现，不依赖私有名。"""
        if self.closed:
            raise ValueError("I/O operation on closed file")

    def readable(self) -> bool:
        return self._readable

    def writable(self) -> bool:
        return self._writable

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        self._require_open()
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._require_open()
        match whence:
            case io.SEEK_SET:
                target = offset
            case io.SEEK_CUR:
                target = self._pos + offset
            case io.SEEK_END:
                target = self._size + offset
            case _:
                raise ValueError(f"非法的 whence: {whence}")
        if target < 0:
            raise ValueError(f"负的偏移量: {target}")
        self._pos = target
        return target

    def readinto(self, buffer: Any) -> int:
        self._require_open()
        if not self._readable:
            raise io.UnsupportedOperation("文件以只写方式打开")
        want = min(len(buffer), CHUNK)
        if want <= 0:
            return 0
        data = self._backend.invoke(
            "读取文件", self._path, lambda: self._handle.read(self._pos, want)
        )
        if not data:
            return 0
        buffer[: len(data)] = data
        self._pos += len(data)
        return len(data)

    def write(self, data: Any) -> int:
        self._require_open()
        if not self._writable:
            raise io.UnsupportedOperation("文件以只读方式打开")
        payload = bytes(data)
        written = 0
        while written < len(payload):
            chunk = payload[written : written + CHUNK]
            sent = self._backend.invoke(
                "写入文件",
                self._path,
                lambda chunk=chunk, at=self._pos + written: self._handle.write(
                    chunk, at
                ),
            )
            if sent <= 0:  # 服务端没有进展，避免死循环
                raise SambaProtocolError(f"写入 {self._path} 时服务端未接受任何数据")
            written += sent
        self._pos += written
        self._size = max(self._size, self._pos)
        return written

    def close(self) -> None:
        if self.closed:
            return
        try:
            self._backend.invoke("关闭文件", self._path, self._handle.close)
        except SambaError:
            if not self._backend.closing:
                raise
        finally:
            super().close()

    # -- 附加信息
    @property
    def size(self) -> int:
        """文件当前大小（字节）。"""
        return self._size

    @property
    def _path(self) -> str:
        return self._handle.file_name.replace("\\", "/")


# ---------------------------------------------------------------- 后端实现
class SmbprotocolBackend:
    """``smbprotocol`` 后端：一个实例对应一条 TCP 连接 + 一个共享。"""

    def __init__(self, options: SMBOptions) -> None:
        self.options = options
        self._lock = threading.RLock()
        self._connection: Connection | None = None
        self._session: Session | None = None
        self._tree: TreeConnect | None = None
        self._info: ServerInfo | None = None
        self._closing = False

    # -- 状态
    @property
    def connected(self) -> bool:
        return self._tree is not None

    @property
    def closing(self) -> bool:
        """正在关闭：此时忽略句柄关闭期的错误，避免掩盖真正的问题。"""
        return self._closing

    def __repr__(self) -> str:
        state = "connected" if self.connected else "disconnected"
        return f"<{type(self).__name__} {self.options.display} {state}>"

    # -- 连接
    def connect(self) -> ServerInfo:
        """建立 TCP → 会话 → 共享三层连接，返回协商结果。"""
        with self._lock:
            if self._info is not None:
                return self._info
            options = self.options
            if not options.share:
                raise SambaConfigError("未指定共享名（share），无法挂载")

            connection = Connection(
                uuid.uuid4(),
                options.server,
                port=options.port,
                require_signing=options.require_signing and not options.allow_guest,
            )
            try:
                connection.connect(timeout=options.timeout)
            except (OSError, ValueError) as exc:
                # smbprotocol 把 socket 错误包成了 ValueError，这里统一成连接类异常
                raise _connect_error(exc, options) from exc
            except Exception as exc:
                raise _translate(exc, options, path=None, action="连接服务器") from exc

            session: Session | None = None
            tree: TreeConnect | None = None
            try:
                session = Session(
                    connection,
                    options.effective_user or None,
                    options.password or None,
                    require_encryption=options.encrypt,
                    auth_protocol="negotiate",
                )
                try:
                    session.connect()
                except Exception as exc:
                    raise _session_error(exc, options) from exc

                tree = TreeConnect(session, options.unc)
                try:
                    # guest/匿名会话没有签名密钥，跳过 SMB3 协商信息校验
                    tree.connect(require_secure_negotiate=not options.allow_guest)
                except Exception as exc:
                    raise _translate(
                        exc, options, path=None, action="挂载共享"
                    ) from exc
            except BaseException:
                _quiet(tree.disconnect if tree else None)
                _quiet(session.disconnect if session else None)
                _quiet(connection.disconnect)
                raise

            self._connection = connection
            self._session = session
            self._tree = tree
            self._info = ServerInfo(
                server=options.server,
                port=options.port,
                share=options.share,
                dialect=dialect_name(int(connection.dialect)),
                server_guid=str(connection.server_guid or ""),
                username=options.effective_user,
                signing=bool(getattr(session, "signing_required", False)),
                encrypted=bool(getattr(session, "encrypt_data", False)),
                connected_at=datetime.now(UTC),
            )
            return self._info

    def close(self) -> None:
        """按 共享 → 会话 → 连接 的顺序优雅断开，重复调用无副作用。"""
        with self._lock:
            self._closing = True
            try:
                _quiet(self._tree.disconnect if self._tree else None)
                _quiet(self._session.disconnect if self._session else None)
                _quiet(self._connection.disconnect if self._connection else None)
            finally:
                self._tree = None
                self._session = None
                self._connection = None
                self._info = None
                self._closing = False

    def use_share(self, share: str) -> ServerInfo:
        """切换共享：重建连接（共享属于会话下的树连接，重建最省心也最可靠）。"""
        options = self.options.with_share(share)
        with self._lock:
            self.close()
            self.options = options
            return self.connect()

    # -- 统一调用入口
    def invoke[T](self, action: str, path: str | None, func: Callable[[], T]) -> T:
        """串行化执行一次 SMB 调用，并把异常翻译成领域异常。"""
        with self._lock:
            if not self.connected:
                self.connect()
            try:
                return func()
            except SambaError:
                raise
            except BaseException as exc:
                raise _translate(exc, self.options, path=path, action=action) from exc

    # -- 内部助手
    def _handle(
        self,
        path: RemotePath,
        *,
        access: int,
        attributes: int,
        disposition: int,
        options: int,
    ) -> Open:
        handle = Open(self._tree, path.wire)
        handle.create(
            ImpersonationLevel.Impersonation,
            access,
            attributes,
            _SHARE_ALL,
            disposition,
            options,
        )
        return handle

    def _open_dir(self, path: RemotePath) -> Open:
        return self._handle(
            path,
            access=_DIR_ACCESS.FILE_LIST_DIRECTORY | _DIR_ACCESS.FILE_READ_ATTRIBUTES,
            attributes=FileAttributes.FILE_ATTRIBUTE_DIRECTORY,
            disposition=CreateDisposition.FILE_OPEN,
            options=CreateOptions.FILE_DIRECTORY_FILE,
        )

    def _query(self, handle: Open, pattern: str = "*") -> list[Any]:
        """翻页枚举目录项。

        信息类按兼容性从强到弱降级：``FILE_ID_BOTH_DIRECTORY_INFORMATION``
        信息最全，但个别服务端（Windows 的 IPC$ 伪共享等）会以
        ``STATUS_INVALID_INFO_CLASS`` 拒绝，此时退回到更基础的目录信息类。
        """
        last_error: BaseException | None = None
        for info_class in _INFO_CLASSES:
            try:
                return self._query_with(handle, pattern, info_class)
            except SMBResponseException as exc:
                if _unsupported_info_class(exc):
                    last_error = exc
                    continue
                if exc.status == _STATUS_NO_MORE_FILES:
                    return []
                raise
        raise SambaUnsupportedError(
            f"服务端不支持任何目录枚举信息类：{_describe(last_error) if last_error else ''}"
        ) from last_error

    @staticmethod
    def _query_with(handle: Open, pattern: str, info_class: int) -> list[Any]:
        entries: list[Any] = []
        for page in range(_MAX_PAGES):
            try:
                chunk = handle.query_directory(
                    pattern,
                    info_class,
                    flags=QueryDirectoryFlags.SMB2_RESTART_SCANS if page == 0 else None,
                )
            except SMBResponseException as exc:
                if exc.status == _STATUS_NO_MORE_FILES:
                    return entries
                raise
            if not chunk:
                return entries
            entries.extend(chunk)
        raise SambaProtocolError(
            f"目录枚举超过 {_MAX_PAGES} 页仍未结束（{pattern}），已中止"
        )

    def _entry(self, parent: RemotePath, info: Any) -> SMBEntry:
        raw = info["file_name"].get_value()
        name = (
            raw.decode("utf-16-le", "replace") if isinstance(raw, bytes) else str(raw)
        )
        name = name.rstrip("\x00")
        attributes = _int_field(info, "file_attributes")
        return SMBEntry(
            name=name,
            path=(parent.join(name) if name else parent).posix,
            is_dir=bool(attributes & FILE_ATTRIBUTE["directory"]),
            size=_int_field(info, "end_of_file"),
            mtime=_time_field(info, "last_write_time"),
            atime=_time_field(info, "last_access_time"),
            ctime=_time_field(info, "change_time"),
            attributes=attributes,
        )

    def _set_info(self, handle: Open, payload: Any) -> None:
        request = SMB2SetInfoRequest()
        request["info_type"] = InfoType.SMB2_0_INFO_FILE
        request["file_info_class"] = payload.INFO_CLASS
        request["additional_information"] = 0
        request["file_id"] = handle.file_id
        request["buffer"] = payload.pack()
        sent = self._connection.send(  # type: ignore[union-attr]
            request,
            self._session.session_id,  # type: ignore[union-attr]
            self._tree.tree_connect_id,  # type: ignore[union-attr]
        )
        self._connection.receive(sent)  # type: ignore[union-attr]

    # -- 只读操作
    def stat(self, path: RemotePath) -> SMBEntry:
        if path.is_root:
            return SMBEntry(
                name="",
                path="",
                is_dir=True,
                attributes=FILE_ATTRIBUTE["directory"],
                mtime=None,
            )
        name = path.name

        def run() -> list[Any]:
            handle = self._open_dir(path.parent)
            try:
                return self._query(handle, name)
            finally:
                _quiet(handle.close)

        entries = self.invoke("读取属性", path.posix, run)
        for item in entries:
            entry = self._entry(path.parent, item)
            if entry.name.casefold() == name.casefold():
                return entry
        raise SambaNotFoundError(path.posix)

    def list_dir(self, path: RemotePath) -> list[SMBEntry]:
        def run() -> list[SMBEntry]:
            handle = self._open_dir(path)
            try:
                return [self._entry(path, item) for item in self._query(handle)]
            finally:
                _quiet(handle.close)

        entries = self.invoke("列出目录", path.posix or "/", run)
        return [e for e in entries if e.name not in {".", ".."}]

    # -- 读写
    def open_read(self, path: RemotePath) -> BinaryIO:
        def run() -> RemoteFile:
            handle = self._handle(
                path,
                access=_ACCESS.GENERIC_READ | _ACCESS.FILE_READ_ATTRIBUTES,
                attributes=FileAttributes.FILE_ATTRIBUTE_NORMAL,
                disposition=CreateDisposition.FILE_OPEN,
                options=CreateOptions.FILE_NON_DIRECTORY_FILE,
            )
            return RemoteFile(
                self,
                handle,
                readable=True,
                writable=False,
                size=int(handle.end_of_file or 0),
            )

        return self.invoke("打开文件", path.posix, run)

    def open_write(self, path: RemotePath, *, mode: str) -> BinaryIO:
        disposition = {
            "create": CreateDisposition.FILE_CREATE,
            "overwrite": CreateDisposition.FILE_OVERWRITE_IF,
            "append": CreateDisposition.FILE_OPEN_IF,
        }.get(mode)
        if disposition is None:
            raise SambaConfigError(f"未知写入模式：{mode!r}")

        def run() -> RemoteFile:
            handle = self._handle(
                path,
                access=_ACCESS.GENERIC_WRITE | _ACCESS.FILE_READ_ATTRIBUTES,
                attributes=FileAttributes.FILE_ATTRIBUTE_NORMAL,
                disposition=disposition,
                options=CreateOptions.FILE_NON_DIRECTORY_FILE,
            )
            size = int(handle.end_of_file or 0)
            return RemoteFile(
                self,
                handle,
                readable=False,
                writable=True,
                size=size,
                position=size if mode == "append" else 0,
            )

        return self.invoke("打开文件", path.posix, run)

    # -- 目录与文件管理
    def mkdir(self, path: RemotePath) -> None:
        def run() -> None:
            handle = self._handle(
                path,
                access=_DIR_ACCESS.FILE_LIST_DIRECTORY | _DIR_ACCESS.FILE_ADD_FILE,
                attributes=FileAttributes.FILE_ATTRIBUTE_DIRECTORY,
                disposition=CreateDisposition.FILE_CREATE,
                options=CreateOptions.FILE_DIRECTORY_FILE,
            )
            _quiet(handle.close)

        self.invoke("创建目录", path.posix, run)

    def remove(self, path: RemotePath) -> None:
        def run() -> None:
            handle = self._handle(
                path,
                access=_ACCESS.DELETE,
                attributes=FileAttributes.FILE_ATTRIBUTE_NORMAL,
                disposition=CreateDisposition.FILE_OPEN,
                options=CreateOptions.FILE_NON_DIRECTORY_FILE
                | CreateOptions.FILE_DELETE_ON_CLOSE,
            )
            _quiet(handle.close)

        self.invoke("删除文件", path.posix, run)

    def rmdir(self, path: RemotePath) -> None:
        def run() -> None:
            handle = self._handle(
                path,
                access=_DIR_ACCESS.DELETE,
                attributes=FileAttributes.FILE_ATTRIBUTE_DIRECTORY,
                disposition=CreateDisposition.FILE_OPEN,
                options=CreateOptions.FILE_DIRECTORY_FILE
                | CreateOptions.FILE_DELETE_ON_CLOSE,
            )
            _quiet(handle.close)

        self.invoke("删除目录", path.posix, run)

    def rename(
        self, source: RemotePath, target: RemotePath, *, overwrite: bool
    ) -> None:
        def run() -> None:
            handle = self._handle(
                source,
                access=_ACCESS.DELETE,
                attributes=FileAttributes.FILE_ATTRIBUTE_NORMAL,
                disposition=CreateDisposition.FILE_OPEN,
                options=CreateOptions.FILE_NON_DIRECTORY_FILE,
            )
            try:
                payload = FileRenameInformation()
                payload["replace_if_exists"] = overwrite
                payload["root_directory"] = 0
                payload["file_name"] = target.wire
                self._set_info(handle, payload)
            finally:
                _quiet(handle.close)

        self.invoke("重命名", source.posix, run)

    def set_mtime(self, path: RemotePath, when: datetime) -> None:
        def run() -> None:
            handle = self._handle(
                path,
                access=_ACCESS.FILE_WRITE_ATTRIBUTES,
                attributes=FileAttributes.FILE_ATTRIBUTE_NORMAL,
                disposition=CreateDisposition.FILE_OPEN,
                options=CreateOptions.FILE_NON_DIRECTORY_FILE,
            )
            try:
                payload = FileBasicInformation()
                payload["last_write_time"] = to_filetime(when)
                self._set_info(handle, payload)
            finally:
                _quiet(handle.close)

        self.invoke("修改时间戳", path.posix, run)


def _quiet(func: Callable[[], Any] | None) -> None:
    """尽力而为地调用清理函数：清理期的异常不应掩盖原始错误。"""
    if func is None:
        return
    try:
        func()
    except Exception:  # noqa: BLE001, S110 - 清理阶段吞掉一切
        pass


def _int_field(info: Any, key: str) -> int:
    """读取结构里的整数字段；不同信息类字段不完全一致，缺失按 0 处理。"""
    try:
        value = info[key].get_value()
    except KeyError:
        return 0
    return value if isinstance(value, int) else 0


def _time_field(info: Any, key: str) -> datetime | None:
    """读取时间字段：``DateTimeField`` 已转成 datetime，整数则按 FILETIME 解释。"""
    try:
        value = info[key].get_value()
    except KeyError:
        return None
    if isinstance(value, datetime):
        moment = as_utc(value)
    elif isinstance(value, int):
        moment = from_filetime(value)
    else:
        return None
    # 1601 年之类的哨兵值（命名管道很常见）在 Windows 上无法转本地时间，按未设置处理
    if moment is not None and moment.year < 1970:
        return None
    return moment
