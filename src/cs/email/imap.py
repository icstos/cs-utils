"""IMAP 收信客户端：列目录、检索、读信、下载附件、标记 / 移动 / 删除。

只依赖标准库 ``imaplib``，默认一操作一系列连接；批量操作建议放进 ``with``::

    with ImapClient.from_account("work") as client:
        for mail in client.messages(limit=10, unseen=True):
            print(mail.uid, mail.subject)

注意：文件夹名按服务器原样返回，非 ASCII 名称可能是 IMAP modified UTF-7，
原样传回 ``mailbox`` 参数即可正确选中。
"""

from __future__ import annotations

import imaplib
import re
import ssl
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Self

from .config import load_account
from .errors import (
    MailAttachmentError,
    MailAuthError,
    MailReceiveError,
)
from .mime import extract_attachments, parse_message
from .models import Attachment, ImapConfig, Received, Security

__all__ = ["ImapClient", "search_criteria"]

_FLAGS_RE = re.compile(r"FLAGS \(([^)]*)\)")
_SIZE_RE = re.compile(r"RFC822\.SIZE (\d+)")
_NAME_RE = re.compile(r'"([^"]*)"\s*$')

#: ``BODY.PEEK[]`` 不改已读状态；``BODY[]`` 会顺带置 \Seen
_SPEC_FULL = "BODY.PEEK[]"
_SPEC_FULL_SEEN = "BODY[]"
_SPEC_HEADER = "BODY.PEEK[HEADER]"


def search_criteria(query: str | None) -> str:
    """把用户输入翻译成 IMAP 检索条件。

    空 → ``ALL``；含空白/括号/引号视为原生语法（如 ``SINCE 01-Jan-2026``）；
    其余按关键词，在主题与发件人里做 OR 模糊匹配。
    """
    text = (query or "").strip()
    if not text:
        return "ALL"
    if any(char in text for char in ' ()"'):
        return text
    escaped = text.replace('"', '\\"')
    return f'OR SUBJECT "{escaped}" FROM "{escaped}"'


class ImapClient:
    """同步 IMAP 收信器。"""

    def __init__(self, config: ImapConfig) -> None:
        self.config = config
        self._connection: imaplib.IMAP4 | None = None

    @classmethod
    def from_account(
        cls, name: str | None = None, *, config: str | Path | None = None
    ) -> Self:
        """按名字从配置文件 / 环境变量加载账号。"""
        return cls(load_account(name, config=config).require_imap())

    # ------------------------------------------------------------ 生命周期

    def __enter__(self) -> Self:
        self._connection = self._connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            _shutdown(connection)

    def verify(self) -> str:
        """登录并列出目录数，用于校验配置。"""
        return (
            f"{self.config.host}:{self.config.port}（{len(self.folders())} 个文件夹）"
        )

    # ------------------------------------------------------------ 读取

    def folders(self, *, pattern: str = "*") -> list[str]:
        """列出邮箱文件夹。"""
        with self._session() as connection:
            try:
                status, data = connection.list(pattern)
            except imaplib.IMAP4.error as exc:
                raise MailReceiveError(f"读取文件夹列表失败：{exc}") from exc
        if status != "OK":
            raise MailReceiveError(f"读取文件夹列表失败：{_as_text(data)}")
        names: list[str] = []
        for raw in data or []:
            if isinstance(raw, bytes) and (name := _folder_name(raw)):
                names.append(name)
        return names

    def count(
        self,
        *,
        mailbox: str | None = None,
        criteria: str = "ALL",
        unseen: bool = False,
    ) -> int:
        """统计满足条件的邮件数量。"""
        with self._session() as connection:
            self._select(connection, mailbox)
            return len(self._search(connection, criteria, unseen))

    def search(
        self,
        criteria: str = "ALL",
        *,
        mailbox: str | None = None,
        unseen: bool = False,
        limit: int = 0,
    ) -> list[int]:
        """按 IMAP 检索条件返回 UID（默认新的在前）。"""
        with self._session() as connection:
            self._select(connection, mailbox)
            uids = self._search(connection, criteria, unseen)
        uids.reverse()
        return uids[:limit] if limit > 0 else uids

    def messages(
        self,
        *,
        mailbox: str | None = None,
        limit: int = 10,
        unseen: bool = False,
        criteria: str = "ALL",
    ) -> list[Received]:
        """列出邮件（只取头部，正文与附件为空），新的在前。"""
        with self._session() as connection:
            box = self._select(connection, mailbox)
            uids = self._search(connection, criteria, unseen)
            uids.reverse()
            if limit > 0:
                uids = uids[:limit]
            return [self._head(connection, uid, box) for uid in uids]

    def fetch(
        self,
        uid: int,
        *,
        mailbox: str | None = None,
        mark_seen: bool = False,
    ) -> Received:
        """读取一封完整邮件（正文 + 附件元信息）；默认不改变已读状态。"""
        spec = _SPEC_FULL_SEEN if mark_seen else _SPEC_FULL
        with self._session() as connection:
            box = self._select(connection, mailbox)
            raw, flags, size = self._fetch(connection, uid, spec)
        if mark_seen and not any("seen" in flag.casefold() for flag in flags):
            flags = (*flags, "\\Seen")
        return parse_message(raw, uid=uid, mailbox=box, flags=flags, size=size)

    def raw(self, uid: int, *, mailbox: str | None = None) -> bytes:
        """取原始邮件字节（供上层自行解析）。"""
        with self._session() as connection:
            self._select(connection, mailbox)
            raw, _, _ = self._fetch(connection, uid, _SPEC_FULL)
        return raw

    # ------------------------------------------------------------ 附件

    def attachment(
        self,
        uid: int,
        *,
        name: str | None = None,
        index: int = 0,
        mailbox: str | None = None,
    ) -> Attachment:
        """按文件名或下标取附件（含内容）。"""
        items = extract_attachments(self.raw(uid, mailbox=mailbox))
        if not items:
            raise MailAttachmentError(f"邮件 {uid} 没有附件")
        if name:
            for item in items:
                if item.filename == name:
                    return item
            available = ", ".join(item.filename for item in items)
            raise MailAttachmentError(
                f"邮件 {uid} 没有名为 {name!r} 的附件，可选：{available}"
            )
        if not 0 <= index < len(items):
            raise MailAttachmentError(f"附件下标越界：{index}（共 {len(items)} 个）")
        return items[index]

    def download(
        self,
        uid: int,
        directory: str | Path = ".",
        *,
        name: str | None = None,
        index: int = 0,
        mailbox: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        """把某个附件保存到本地目录，返回实际路径。"""
        attachment = self.attachment(uid, name=name, index=index, mailbox=mailbox)
        return attachment.save(directory, overwrite=overwrite)

    # ------------------------------------------------------------ 写入

    def mark(
        self,
        uid: int,
        flag: str = "\\Seen",
        *,
        value: bool = True,
        mailbox: str | None = None,
    ) -> None:
        """加 / 去一个标记；``flag`` 可写 ``seen`` 或 ``\\Seen``。"""
        with self._session() as connection:
            self._select(connection, mailbox)
            self._store(connection, uid, "+FLAGS" if value else "-FLAGS", _flag(flag))

    def move(self, uid: int, folder: str, *, mailbox: str | None = None) -> None:
        """移动到另一个文件夹；服务器不支持 ``MOVE`` 时退回 copy + delete。"""
        with self._session() as connection:
            self._select(connection, mailbox)
            target = _quote(folder)
            try:
                status, _ = connection.uid("MOVE", str(uid), target)
            except imaplib.IMAP4.error:
                status = "NO"
            if status == "OK":
                return
            self._copy_then_delete(connection, uid, target, folder)

    def delete(
        self, uid: int, *, mailbox: str | None = None, expunge: bool = True
    ) -> None:
        """删除邮件：打 ``\\Deleted`` 标记（可选立即 expunge）。"""
        with self._session() as connection:
            self._select(connection, mailbox)
            self._store(connection, uid, "+FLAGS", "(\\Deleted)")
            if expunge:
                with suppress(imaplib.IMAP4.error):
                    connection.expunge()

    # ------------------------------------------------------------ 内部

    def _connect(self) -> imaplib.IMAP4:
        cfg = self.config
        try:
            if cfg.security is Security.SSL:
                connection: imaplib.IMAP4 = imaplib.IMAP4_SSL(
                    cfg.host,
                    cfg.port,
                    ssl_context=ssl.create_default_context(),
                    timeout=cfg.timeout,
                )
            else:
                connection = imaplib.IMAP4(cfg.host, cfg.port, timeout=cfg.timeout)
                if cfg.security is Security.STARTTLS:
                    connection.starttls(ssl_context=ssl.create_default_context())
        except (imaplib.IMAP4.error, OSError) as exc:
            raise MailReceiveError(
                f"连接 IMAP 服务器 {cfg.host}:{cfg.port} 失败：{exc}"
            ) from exc

        if cfg.username:
            try:
                connection.login(cfg.username, cfg.password)
            except imaplib.IMAP4.error as exc:
                with suppress(Exception):
                    connection.shutdown()
                raise MailAuthError(f"IMAP 登录失败（{cfg.username}）：{exc}") from exc
        return connection

    @contextmanager
    def _session(self) -> Iterator[imaplib.IMAP4]:
        if self._connection is not None:
            yield self._connection
            return
        connection = self._connect()
        try:
            yield connection
        finally:
            _shutdown(connection)

    def _select(self, connection: imaplib.IMAP4, mailbox: str | None) -> str:
        name = mailbox or self.config.mailbox
        try:
            status, data = connection.select(name, readonly=False)
        except imaplib.IMAP4.error as exc:
            raise MailReceiveError(f"打开文件夹 {name!r} 失败：{exc}") from exc
        if status != "OK":
            raise MailReceiveError(f"打开文件夹 {name!r} 失败：{_as_text(data)}")
        return name

    @staticmethod
    def _search(connection: imaplib.IMAP4, criteria: str, unseen: bool) -> list[int]:
        query = (criteria or "ALL").strip() or "ALL"
        if unseen:
            query = "UNSEEN" if query.upper() == "ALL" else f"UNSEEN {query}"
        try:
            status, data = connection.uid("search", None, query)
        except imaplib.IMAP4.error as exc:
            raise MailReceiveError(f"检索失败（{query}）：{exc}") from exc
        if status != "OK":
            raise MailReceiveError(f"检索失败（{query}）：{_as_text(data)}")
        payload = data[0] if data else b""
        if not isinstance(payload, bytes):
            return []
        return [int(token) for token in payload.split() if token.isdigit()]

    @staticmethod
    def _fetch(
        connection: imaplib.IMAP4, uid: int, spec: str
    ) -> tuple[bytes, tuple[str, ...], int]:
        """取 ``spec`` 指定的内容，并顺带解析 FLAGS / RFC822.SIZE。"""
        try:
            status, data = connection.uid(
                "fetch", str(uid), f"({spec} FLAGS RFC822.SIZE)"
            )
        except imaplib.IMAP4.error as exc:
            raise MailReceiveError(f"读取邮件 {uid} 失败：{exc}") from exc
        if status != "OK":
            raise MailReceiveError(f"读取邮件 {uid} 失败：{_as_text(data)}")

        raw = b""
        meta_parts: list[bytes] = []
        for item in data or []:
            if isinstance(item, tuple):
                meta_parts.append(item[0])
                raw = item[1] or raw
            elif isinstance(item, bytes):
                meta_parts.append(item)
        meta = b" ".join(meta_parts).decode("ascii", "replace")
        flags = _FLAGS_RE.search(meta)
        size = _SIZE_RE.search(meta)
        return (
            raw,
            tuple(flags.group(1).split()) if flags else (),
            int(size.group(1)) if size else len(raw),
        )

    def _head(self, connection: imaplib.IMAP4, uid: int, mailbox: str) -> Received:
        raw, flags, size = self._fetch(connection, uid, _SPEC_HEADER)
        return parse_message(raw, uid=uid, mailbox=mailbox, flags=flags, size=size)

    @staticmethod
    def _store(connection: imaplib.IMAP4, uid: int, operation: str, flags: str) -> None:
        try:
            status, data = connection.uid("store", str(uid), operation, flags)
        except imaplib.IMAP4.error as exc:
            raise MailReceiveError(f"更新邮件 {uid} 标记失败：{exc}") from exc
        if status != "OK":
            raise MailReceiveError(f"更新邮件 {uid} 标记失败：{_as_text(data)}")

    def _copy_then_delete(
        self, connection: imaplib.IMAP4, uid: int, target: str, folder: str
    ) -> None:
        try:
            status, data = connection.uid("copy", str(uid), target)
        except imaplib.IMAP4.error as exc:
            raise MailReceiveError(f"移动邮件 {uid} 到 {folder!r} 失败：{exc}") from exc
        if status != "OK":
            raise MailReceiveError(
                f"移动邮件 {uid} 到 {folder!r} 失败：{_as_text(data)}"
            )
        self._store(connection, uid, "+FLAGS", "(\\Deleted)")
        with suppress(imaplib.IMAP4.error):
            connection.expunge()


# ------------------------------------------------------------ 内部工具


def _shutdown(connection: imaplib.IMAP4) -> None:
    """先 CLOSE 退出选中态，再 LOGOUT；任何一步失败都不影响调用方。"""
    with suppress(Exception):
        if connection.state == "SELECTED":
            connection.close()
    with suppress(Exception):
        connection.logout()


def _as_text(data: object) -> str:
    if not isinstance(data, (list, tuple)):
        return str(data or "")
    parts = [_decode(item) for item in data if isinstance(item, bytes)]
    return " ".join(part for part in parts if part)


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", "replace").strip()


def _folder_name(raw: bytes) -> str:
    """从 ``(\\HasNoChildren) "/" "INBOX"`` 这类行里取出文件夹名。"""
    match = _NAME_RE.search(_decode(raw))
    if match:
        return match.group(1)
    return _decode(raw).rsplit(" ", 1)[-1].strip('"')


def _flag(flag: str) -> str:
    """把 ``seen`` 规范成 ``\\Seen``；已带反斜杠的原样返回。"""
    text = (flag or "\\Seen").strip()
    return text if text.startswith("\\") else f"\\{text.capitalize()}"


def _quote(folder: str) -> str:
    escaped = folder.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
