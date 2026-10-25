"""``cs.samba`` 的值对象：连接参数、目录项、服务器信息、传输结果。

全部使用 ``@dataclass(frozen=True, slots=True)``，可哈希、可比较、无副作用；
每个模型都提供 ``to_dict()``，方便直接喂给 JSON / MCP / CLI。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from .errors import SambaConfigError, SambaPathError

__all__ = [
    "FILE_ATTRIBUTE",
    "SMBEntry",
    "SMBOptions",
    "ServerInfo",
    "StrPath",
    "TransferResult",
    "as_utc",
    "dialect_name",
    "ensure_remote_path",
    "human_age",
    "human_size",
]

type StrPath = str | os.PathLike[str]

_NUMERIC = re.compile(r"\d+(?:\.\d+)?")

#: ``FILE_ATTRIBUTE_*`` 位掩码（[MS-FSCC] 2.6）
FILE_ATTRIBUTE: dict[str, int] = {
    "readonly": 0x0001,
    "hidden": 0x0002,
    "system": 0x0004,
    "directory": 0x0010,
    "archive": 0x0020,
    "normal": 0x0080,
    "temporary": 0x0100,
    "sparse": 0x0200,
    "reparse_point": 0x0400,
    "compressed": 0x0800,
    "offline": 0x1000,
    "not_indexed": 0x2000,
    "encrypted": 0x4000,
}

_DIALECTS: dict[int, str] = {
    0x0202: "SMB 2.0.2",
    0x0210: "SMB 2.1",
    0x0300: "SMB 3.0",
    0x0302: "SMB 3.0.2",
    0x0311: "SMB 3.1.1",
    0x02FF: "SMB 1.0",
}


def dialect_name(dialect: int | str) -> str:
    """把协商得到的方言编号翻译成 ``SMB 3.1.1`` 这类可读名称。"""
    if isinstance(dialect, str):
        return dialect
    return _DIALECTS.get(dialect, f"未知方言 0x{dialect:04x}")


def as_utc(value: datetime | None) -> datetime | None:
    """补齐时区信息：SMB 的时间戳都是 UTC，naive 值按 UTC 解释。"""
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def human_size(size: int | None, *, digits: int = 1) -> str:
    """把字节数格式化为 ``1.5 MiB``。"""
    if size is None:
        return "?"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(value) < 1024 or unit == "PiB":
            return f"{value:.0f} B" if unit == "B" else f"{value:.{digits}f} {unit}"
        value /= 1024
    return f"{value:.{digits}f} PiB"


def human_age(delta: timedelta | None) -> str:
    """把时间差说成人话：``3 天前`` / ``2 小时后``。"""
    if delta is None:
        return "未知"
    seconds = int(delta.total_seconds())
    suffix = "前" if seconds >= 0 else "后"
    seconds = abs(seconds)
    for limit, unit, size in (
        (60, "秒", 1),
        (3600, "分钟", 60),
        (86_400, "小时", 3600),
        (86_400 * 30, "天", 86_400),
        (86_400 * 365, "个月", 86_400 * 30),
    ):
        if seconds < limit:
            return f"{seconds // size} {unit}{suffix}"
    return f"{seconds // (86_400 * 365)} 年{suffix}"


# ---------------------------------------------------------------- 连接参数
@dataclass(frozen=True, slots=True)
class SMBOptions:
    """连接一个 SMB 共享所需的全部参数。

    :param server: 主机名或 IP（不含 ``\\\\`` 前缀）
    :param share: 共享名；留空表示只建会话、稍后用 :meth:`SMBClient.use_share` 切换
    :param username: 用户名；空字符串表示匿名/guest
    :param password: 密码（``repr`` 中隐藏）
    :param domain: NTLM 域，等价于 ``DOMAIN\\user`` 写法
    :param port: 默认 445（直连 TCP）
    :param timeout: 连接/读写超时（秒）
    :param encrypt: 是否要求 SMB3 加密（服务端需支持）
    :param require_signing: 是否要求消息签名（更安全，个别 NAS 不支持）
    :param allow_guest: 允许 guest/匿名会话（会放宽签名与协商校验）
    :param retries: 连接失败后的重试次数
    :param backoff: 重试退避基数（秒），按次数线性增长
    """

    server: str
    share: str = ""
    username: str = ""
    password: str = field(default="", repr=False)
    domain: str = ""
    port: int = 445
    timeout: float = 30.0
    encrypt: bool = False
    require_signing: bool = True
    allow_guest: bool = False
    retries: int = 2
    backoff: float = 0.4

    def __post_init__(self) -> None:
        server = self.server.strip().strip("\\/")
        if not server:
            raise SambaConfigError("server 不能为空")
        if "\\" in server or "/" in server:
            raise SambaConfigError(f"server 只能是主机名或 IP，收到 {self.server!r}")
        share = self.share.strip().strip("\\/")
        if "\\" in share or "/" in share:
            raise SambaConfigError(
                f"share 只能是共享名，不能带路径分隔符：{self.share!r}"
            )
        if not 1 <= self.port <= 65535:
            raise SambaConfigError(f"port 必须在 1~65535 之间，收到 {self.port}")
        if not 0 < self.timeout <= 3600:
            raise SambaConfigError(
                f"timeout 必须在 (0, 3600] 秒之间，收到 {self.timeout}"
            )
        if self.retries < 0:
            raise SambaConfigError(f"retries 不能为负：{self.retries}")
        if self.backoff < 0:
            raise SambaConfigError(f"backoff 不能为负：{self.backoff}")
        object.__setattr__(self, "server", server)
        object.__setattr__(self, "share", share)

    # -- 派生属性
    @property
    def is_anonymous(self) -> bool:
        """是否以匿名/guest 身份连接。"""
        return not self.username

    @property
    def effective_user(self) -> str:
        """实际传给 NTLM 的用户名（带上域前缀）。"""
        if self.domain and self.username:
            return f"{self.domain}\\{self.username}"
        return self.username

    @property
    def unc(self) -> str:
        """UNC 路径，如 ``\\\\nas\\share``。"""
        return (
            f"\\\\{self.server}\\{self.share}" if self.share else f"\\\\{self.server}"
        )

    @property
    def display(self) -> str:
        who = self.username or "anonymous"
        target = f"{self.server}:{self.port}/{self.share or '-'}"
        return f"{who}@{target}"

    # -- 构造与变换
    def replace(self, **changes: Any) -> SMBOptions:
        """派生一份新参数（例如切换共享、补上密码）。"""
        return replace(self, **changes)

    def with_share(self, share: str) -> SMBOptions:
        return self.replace(share=share)

    def to_dict(self, *, mask_password: bool = True) -> dict[str, Any]:
        """序列化为字典；默认不泄露密码，仅给出 ``has_password``。"""
        payload: dict[str, Any] = {
            "server": self.server,
            "port": self.port,
            "share": self.share,
            "username": self.username,
            "domain": self.domain,
            "timeout": self.timeout,
            "encrypt": self.encrypt,
            "require_signing": self.require_signing,
            "allow_guest": self.allow_guest,
            "retries": self.retries,
            "anonymous": self.is_anonymous,
            "has_password": bool(self.password),
            "unc": self.unc,
        }
        if not mask_password:
            payload["password"] = self.password
        return payload

    # -- 解析
    @classmethod
    def from_url(cls, url: str, **overrides: Any) -> tuple[SMBOptions, str]:
        """解析 ``smb://[DOMAIN;]user:pass@host:445/share/sub/dir``。

        返回 ``(options, 子路径)``；子路径是共享内的相对目录（可能为空）。
        """
        parsed = urlsplit(url)
        if parsed.scheme not in {"smb", "cifs"}:
            raise SambaConfigError(
                f"URL 必须以 smb:// 开头，收到 {parsed.scheme or url!r}"
            )
        if not parsed.hostname:
            raise SambaConfigError(f"URL 缺少主机名：{url!r}")

        username = unquote(parsed.username or "")
        domain = ""
        if ";" in username:  # DOMAIN;user 是 SMB URL 的约定写法
            domain, _, username = username.partition(";")
        password = unquote(parsed.password or "")

        segments = [part for part in parsed.path.split("/") if part]
        share = segments[0] if segments else ""
        sub_path = "/".join(segments[1:])

        options = cls(
            server=parsed.hostname,
            share=share,
            username=username,
            password=password,
            domain=domain,
            port=parsed.port or 445,
        )
        return (options.replace(**overrides) if overrides else options), sub_path

    @classmethod
    def from_env(
        cls, env: dict[str, str] | None = None, prefix: str = "SMB_", **overrides: Any
    ) -> SMBOptions:
        """从环境变量读取（``SMB_HOST`` / ``SMB_SHARE`` / ``SMB_USER`` …）。"""
        source = os.environ if env is None else env

        def string(name: str) -> str:
            return source.get(f"{prefix}{name}", "").strip()

        def number(name: str, default: str) -> str:
            raw = string(name)
            if raw and _NUMERIC.fullmatch(raw) is None:
                raise SambaConfigError(f"环境变量 {prefix}{name}={raw!r} 不是数字")
            return raw or default

        host = string("HOST") or string("SERVER")
        if not host:
            raise SambaConfigError(f"缺少环境变量 {prefix}HOST")

        options = cls(
            server=host,
            share=string("SHARE"),
            username=string("USER") or string("USERNAME"),
            password=string("PASSWORD"),
            domain=string("DOMAIN"),
            port=int(number("PORT", "445")),
            timeout=float(number("TIMEOUT", "30")),
            encrypt=string("ENCRYPT").lower() in {"1", "true", "yes"},
        )
        return options.replace(**overrides) if overrides else options


# ---------------------------------------------------------------- 目录项
@dataclass(frozen=True, slots=True)
class SMBEntry:
    """目录项（文件或子目录）的元数据快照。"""

    name: str
    path: str
    is_dir: bool
    size: int = 0
    mtime: datetime | None = None
    atime: datetime | None = None
    ctime: datetime | None = None
    attributes: int = 0

    # -- 名称拆分（沿用 pathlib 语义：``.env`` 的 stem 是 ``.env``、suffix 为空）
    @property
    def stem(self) -> str:
        return PurePosixPath(self.name).stem

    @property
    def suffix(self) -> str:
        return PurePosixPath(self.name).suffix

    # -- 属性判定
    @property
    def is_file(self) -> bool:
        return not self.is_dir

    @property
    def is_hidden(self) -> bool:
        """隐藏文件：带 HIDDEN 属性，或名字以点开头（Samba 习惯）。"""
        return bool(self.attributes & FILE_ATTRIBUTE["hidden"]) or self.name.startswith(
            "."
        )

    @property
    def is_readonly(self) -> bool:
        return bool(self.attributes & FILE_ATTRIBUTE["readonly"])

    @property
    def flags(self) -> list[str]:
        """已置位的属性名列表，便于展示。"""
        return [name for name, bit in FILE_ATTRIBUTE.items() if self.attributes & bit]

    # -- 时间
    @property
    def mtime_local(self) -> datetime | None:
        """最后修改时间（本地时区）；时间戳离谱时退回 ``None`` 而不是抛错。"""
        if self.mtime is None:
            return None
        try:
            return self.mtime.astimezone()
        except (OSError, OverflowError, ValueError):  # Windows 对 1601/9999 会直接报错
            return None

    @property
    def mtime_epoch(self) -> float | None:
        """最后修改时间的 UNIX 时间戳。"""
        if self.mtime is None:
            return None
        try:
            return self.mtime.timestamp()
        except (OSError, OverflowError, ValueError):
            return None

    def age(self, *, now: datetime | None = None) -> timedelta | None:
        """距今多久（``now`` 默认取当前 UTC 时间）。"""
        if self.mtime is None:
            return None
        try:
            return (as_utc(now) or datetime.now(UTC)) - self.mtime
        except (OverflowError, ValueError):
            return None

    # -- 序列化
    def to_dict(self, *, times: bool = True) -> dict[str, Any]:
        """转为字典；``times=False`` 时省略时间字段，列表场景更紧凑。"""
        payload: dict[str, Any] = {
            "name": self.name,
            "path": self.path,
            "type": "dir" if self.is_dir else "file",
            "size": self.size,
            "size_human": human_size(self.size) if self.is_file else "-",
        }
        if times:
            age = self.age()
            local = self.mtime_local
            payload.update(
                mtime=local.isoformat(timespec="seconds") if local else None,
                mtime_utc=self.mtime.isoformat(timespec="seconds")
                if self.mtime
                else None,
                mtime_epoch=self.mtime_epoch,
                age_seconds=round(age.total_seconds(), 3) if age else None,
            )
        if self.attributes:
            payload["attributes"] = self.flags
        if self.is_hidden:
            payload["hidden"] = True
        return payload

    def __str__(self) -> str:
        kind = "dir " if self.is_dir else "file"
        local = self.mtime_local
        stamp = local.strftime("%Y-%m-%d %H:%M") if local else "   ?   "
        size = "-" if self.is_dir else human_size(self.size)
        return f"{stamp}  {kind}  {size:>10}  {self.path or '.'}"


# ---------------------------------------------------------------- 服务器信息
@dataclass(frozen=True, slots=True)
class ServerInfo:
    """一次成功连接的快照。"""

    server: str
    port: int
    share: str
    dialect: str
    server_guid: str
    username: str
    signing: bool
    encrypted: bool
    connected_at: datetime

    @property
    def unc(self) -> str:
        return (
            f"\\\\{self.server}\\{self.share}" if self.share else f"\\\\{self.server}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "server": self.server,
            "port": self.port,
            "share": self.share,
            "unc": self.unc,
            "dialect": self.dialect,
            "server_guid": self.server_guid,
            "username": self.username or "anonymous",
            "signing": self.signing,
            "encrypted": self.encrypted,
            "connected_at": self.connected_at.astimezone().isoformat(
                timespec="seconds"
            ),
        }

    def __str__(self) -> str:
        user = self.username or "anonymous"
        secure = "签名+加密" if self.encrypted else ("签名" if self.signing else "明文")
        return f"{self.unc} [{self.dialect}] {user} ({secure})"


# ---------------------------------------------------------------- 传输结果
@dataclass(frozen=True, slots=True)
class TransferResult:
    """一次上传/下载的统计结果。"""

    action: Literal["download", "upload", "copy"]
    source: str
    target: str
    size: int
    seconds: float
    resumed_bytes: int = 0

    @property
    def speed(self) -> float:
        """平均速度（字节/秒）。"""
        return self.size / self.seconds if self.seconds > 0 else float(self.size)

    @property
    def speed_human(self) -> str:
        return f"{human_size(int(self.speed))}/s"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "source": self.source,
            "target": self.target,
            "size": self.size,
            "size_human": human_size(self.size),
            "seconds": round(self.seconds, 3),
            "speed": self.speed_human,
            "resumed_bytes": self.resumed_bytes,
        }

    def __str__(self) -> str:
        verb = {"download": "下载", "upload": "上传", "copy": "复制"}[self.action]
        resume = (
            f"（续传 {human_size(self.resumed_bytes)}）" if self.resumed_bytes else ""
        )
        return (
            f"{verb} {human_size(self.size)} {self.source} -> {self.target} "
            f"耗时 {self.seconds:.2f}s（{self.speed_human}）{resume}"
        )


def ensure_remote_path(value: StrPath) -> str:
    """把 ``str`` / ``Path`` 统一成远程路径字符串（保留原始分隔符，交由 RemotePath 处理）。"""
    if isinstance(value, str):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise SambaPathError(f"需要远程路径（str 或 Path），收到 {type(value).__name__}")
