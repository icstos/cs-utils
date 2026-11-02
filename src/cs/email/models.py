"""邮件领域模型：账号配置、待发邮件、收信结果。

模型全部是 ``dataclass``，且尽量保持不可变（``frozen=True``），
密码字段 ``repr=False`` 不进日志，也不进 ``to_dict()`` 的输出。

>>> SmtpConfig("smtp.example.com", username="me@example.com", password="x").port
465
"""

from __future__ import annotations

import mimetypes
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from .errors import MailConfigError, MailError

__all__ = [
    "Account",
    "AddressInput",
    "Attachment",
    "AttachmentInfo",
    "ImapConfig",
    "Mail",
    "Received",
    "Security",
    "SendResult",
    "SmtpConfig",
    "normalize_addresses",
    "safe_filename",
]

#: 地址入参：单个地址、逗号分隔串，或地址序列
type AddressInput = str | Sequence[str] | None


class Security(StrEnum):
    """传输层加密方式。"""

    SSL = "ssl"
    STARTTLS = "starttls"
    NONE = "none"

    @classmethod
    def parse(cls, value: Security | str | None) -> Security:
        """宽松解析：``None``/空串按 ``SSL``，大小写不敏感。"""
        if isinstance(value, cls):
            return value
        if value is None or not str(value).strip():
            return cls.SSL
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            options = " / ".join(member.value for member in cls)
            raise MailConfigError(
                f"security 只能是 {options}，收到 {value!r}"
            ) from None

    def default_port(self, protocol: str) -> int:
        """按协议与加密方式取标准端口。"""
        table = {
            ("smtp", "ssl"): 465,
            ("smtp", "starttls"): 587,
            ("smtp", "none"): 25,
            ("imap", "ssl"): 993,
            ("imap", "starttls"): 143,
            ("imap", "none"): 143,
        }
        return table[(protocol, self.value)]


# ============================================================
# 地址与文件名
# ============================================================

_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def normalize_addresses(value: AddressInput) -> tuple[str, ...]:
    """把 ``"a@x"``、``"a@x, b@y"``、``["a@x", "b@y"]`` 统一成去重保序的元组。"""
    if value is None:
        return ()
    items: Sequence[str] = (value,) if isinstance(value, str) else value
    return tuple(
        dict.fromkeys(
            address
            for item in items
            for part in str(item).replace(";", ",").split(",")
            if (address := part.strip())
        )
    )


def safe_filename(name: str, fallback: str = "attachment") -> str:
    """去掉路径分隔符与控制字符，避免收信方文件名造成目录穿越。"""
    cleaned = _UNSAFE_CHARS.sub("_", (name or "").strip()).strip(". ")
    return cleaned or fallback


# ============================================================
# 账号配置
# ============================================================


@dataclass(slots=True, frozen=True)
class SmtpConfig:
    """SMTP 发信配置；``port=0`` 时按 ``security`` 自动取标准端口。"""

    host: str = ""
    port: int = 0
    username: str = ""
    password: str = field(default="", repr=False)
    sender: str = ""
    security: Security = Security.SSL
    timeout: float = 30.0

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise MailConfigError("SMTP 主机（smtp_host）不能为空")
        security = Security.parse(self.security)
        object.__setattr__(self, "security", security)
        port = self.port or security.default_port("smtp")
        if not 1 <= port <= 65535:
            raise MailConfigError(f"SMTP 端口越界：{port}")
        object.__setattr__(self, "port", port)
        if self.timeout <= 0:
            raise MailConfigError("timeout 必须大于 0")
        if self.password and not self.username:
            raise MailConfigError("配置了 SMTP 密码但缺少 username")

    @property
    def from_addr(self) -> str:
        """发件地址：优先 ``sender``，否则退回 ``username``。"""
        return (self.sender or self.username).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "security": self.security.value,
            "username": self.username,
            "sender": self.from_addr,
            "timeout": self.timeout,
            "password": "***" if self.password else "",
        }


@dataclass(slots=True, frozen=True)
class ImapConfig:
    """IMAP 收信配置；``mailbox`` 为默认打开的文件夹。"""

    host: str = ""
    port: int = 0
    username: str = ""
    password: str = field(default="", repr=False)
    security: Security = Security.SSL
    timeout: float = 30.0
    mailbox: str = "INBOX"

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise MailConfigError("IMAP 主机（imap_host）不能为空")
        security = Security.parse(self.security)
        object.__setattr__(self, "security", security)
        port = self.port or security.default_port("imap")
        if not 1 <= port <= 65535:
            raise MailConfigError(f"IMAP 端口越界：{port}")
        object.__setattr__(self, "port", port)
        if self.timeout <= 0:
            raise MailConfigError("timeout 必须大于 0")
        if not self.mailbox.strip():
            object.__setattr__(self, "mailbox", "INBOX")
        if self.password and not self.username:
            raise MailConfigError("配置了 IMAP 密码但缺少 username")

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "security": self.security.value,
            "username": self.username,
            "mailbox": self.mailbox,
            "timeout": self.timeout,
            "password": "***" if self.password else "",
        }


@dataclass(slots=True, frozen=True)
class Account:
    """一个邮箱账号：SMTP 与 IMAP 可只配其一。"""

    name: str = "default"
    smtp: SmtpConfig | None = None
    imap: ImapConfig | None = None

    def require_smtp(self) -> SmtpConfig:
        if self.smtp is None:
            raise MailConfigError(f"账号 {self.name!r} 未配置 SMTP（smtp_host）")
        return self.smtp

    def require_imap(self) -> ImapConfig:
        if self.imap is None:
            raise MailConfigError(f"账号 {self.name!r} 未配置 IMAP（imap_host）")
        return self.imap

    @property
    def capabilities(self) -> tuple[str, ...]:
        return tuple(
            protocol
            for protocol, config in (("smtp", self.smtp), ("imap", self.imap))
            if config is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "capabilities": list(self.capabilities),
            "smtp": self.smtp.to_dict() if self.smtp else None,
            "imap": self.imap.to_dict() if self.imap else None,
        }


# ============================================================
# 附件
# ============================================================


@dataclass(slots=True, frozen=True)
class AttachmentInfo:
    """附件元信息（不含内容），用于读信时展示。"""

    filename: str = ""
    size: int = 0
    mime_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "size": self.size,
            "mime_type": self.mime_type,
        }


@dataclass(slots=True, frozen=True)
class Attachment:
    """带内容的附件：由本地文件构造，或从远端邮件中解析。"""

    filename: str = ""
    data: bytes = field(default=b"", repr=False)
    mime_type: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "filename", safe_filename(self.filename))
        if not self.mime_type:
            guessed, _ = mimetypes.guess_type(self.filename)
            object.__setattr__(self, "mime_type", guessed or "application/octet-stream")

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def subtype(self) -> tuple[str, str]:
        """拆成 ``(maintype, subtype)``，缺省按二进制流。"""
        maintype, _, subtype = self.mime_type.partition("/")
        return maintype or "application", subtype or "octet-stream"

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        file = Path(path).expanduser()
        if not file.is_file():
            raise MailError(f"附件不存在：{file}")
        return cls(filename=file.name, data=file.read_bytes())

    def info(self) -> AttachmentInfo:
        return AttachmentInfo(self.filename, self.size, self.mime_type)

    def save(self, directory: str | Path = ".", *, overwrite: bool = False) -> Path:
        """保存到目录（同名文件默认不覆盖），返回实际路径。"""
        target = Path(directory).expanduser() / self.filename
        if target.exists() and not overwrite:
            raise MailError(f"目标文件已存在：{target}（需要覆盖请开启 overwrite）")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.data)
        return target

    def to_dict(self) -> dict[str, Any]:
        return self.info().to_dict()


# ============================================================
# 待发送邮件
# ============================================================


@dataclass(slots=True)
class Mail:
    """待发送邮件。

    ``to`` / ``cc`` / ``bcc`` 接受单个地址、逗号分隔串或地址序列；
    ``attachments`` 接受 :class:`Attachment` 或本地路径。
    """

    subject: str = ""
    text: str = ""
    html: str | None = None
    to: AddressInput = None
    cc: AddressInput = None
    bcc: AddressInput = None
    attachments: Sequence[Attachment | str | Path] = ()
    reply_to: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def recipients(self) -> tuple[str, ...]:
        """全部收件地址（to + cc + bcc，去重保序）——SMTP 投递用。"""
        merged = (
            normalize_addresses(self.to)
            + normalize_addresses(self.cc)
            + normalize_addresses(self.bcc)
        )
        return tuple(dict.fromkeys(merged))

    def resolved_attachments(self) -> tuple[Attachment, ...]:
        return tuple(
            item if isinstance(item, Attachment) else Attachment.from_path(item)
            for item in self.attachments
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "to": list(normalize_addresses(self.to)),
            "cc": list(normalize_addresses(self.cc)),
            "bcc": list(normalize_addresses(self.bcc)),
            "attachments": len(self.attachments),
            "has_html": bool(self.html),
        }


# ============================================================
# 收信结果
# ============================================================


@dataclass(slots=True, frozen=True)
class Received:
    """一封收到的邮件；列表场景只填头部，正文与附件为空。"""

    uid: int = 0
    mailbox: str = "INBOX"
    subject: str = ""
    sender: str = ""
    to: tuple[str, ...] = ()
    cc: tuple[str, ...] = ()
    date: datetime | None = None
    size: int = 0
    flags: tuple[str, ...] = ()
    text: str = ""
    html: str = ""
    attachments: tuple[AttachmentInfo, ...] = ()

    @property
    def is_seen(self) -> bool:
        return any("seen" in flag.casefold() for flag in self.flags)

    @property
    def has_attachments(self) -> bool:
        return bool(self.attachments)

    def to_dict(self, *, max_chars: int = 0) -> dict[str, Any]:
        """``max_chars > 0`` 时截断正文，避免整封邮件灌进模型上下文。"""
        text = self.text if max_chars <= 0 else self.text[:max_chars]
        html = self.html if max_chars <= 0 else self.html[:max_chars]
        return {
            "uid": self.uid,
            "mailbox": self.mailbox,
            "subject": self.subject,
            "from": self.sender,
            "to": list(self.to),
            "cc": list(self.cc),
            "date": self.date.isoformat() if self.date else None,
            "size": self.size,
            "flags": list(self.flags),
            "seen": self.is_seen,
            "attachments": [item.to_dict() for item in self.attachments],
            "text": text,
            "html": html,
            "truncated": (text != self.text) or (html != self.html),
        }


@dataclass(slots=True, frozen=True)
class SendResult:
    """一次投递的结果摘要。"""

    sender: str = ""
    recipients: tuple[str, ...] = ()
    subject: str = ""
    attachments: int = 0
    message_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "recipients": list(self.recipients),
            "subject": self.subject,
            "attachments": self.attachments,
            "message_id": self.message_id,
        }
