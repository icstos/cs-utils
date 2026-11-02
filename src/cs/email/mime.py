"""MIME 组装与解析：在 :class:`~cs.email.models.Mail` 与 RFC 822 字节之间转换。

只依赖标准库 ``email``，不引第三方解析器。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime
from email import policy
from email.message import EmailMessage, Message
from email.parser import BytesParser
from email.utils import formatdate, getaddresses, make_msgid, parsedate_to_datetime

from .errors import MailSendError
from .models import (
    Attachment,
    AttachmentInfo,
    Mail,
    Received,
    normalize_addresses,
    safe_filename,
)

__all__ = ["build_message", "extract_attachments", "parse_message"]

#: 由调用方决定、禁止被 ``Mail.headers`` 覆盖的头
_RESERVED_HEADERS = frozenset(
    {"from", "to", "cc", "bcc", "subject", "date", "message-id"}
)


def build_message(mail: Mail, *, sender: str) -> EmailMessage:
    """组装邮件：纯文本正文 + 可选 HTML 备选 + 附件，并补齐 Date / Message-ID。"""
    to = normalize_addresses(mail.to)
    if not to:
        raise MailSendError("至少需要一个收件人（to）")
    cc = normalize_addresses(mail.cc)

    message = EmailMessage()
    message["Subject"] = mail.subject
    message["From"] = sender
    message["To"] = ", ".join(to)
    if cc:
        message["Cc"] = ", ".join(cc)
    if mail.reply_to:
        message["Reply-To"] = mail.reply_to
    message["Date"] = formatdate(localtime=True)
    domain = sender.rpartition("@")[2] or None
    message["Message-ID"] = make_msgid(domain=domain)
    for name, value in mail.headers.items():
        if name.strip().casefold() not in _RESERVED_HEADERS:
            message[name] = value

    message.set_content(mail.text, cte=_cte(mail.text))
    if mail.html:
        message.add_alternative(mail.html, subtype="html", cte=_cte(mail.html))
    for attachment in mail.resolved_attachments():
        maintype, subtype = attachment.subtype
        text = _decoded_text(attachment.data) if maintype == "text" else None
        if text is not None:
            # 文本附件显式声明 utf-8，否则中文会被当成 ascii 编坏
            message.add_attachment(
                text,
                subtype=subtype,
                charset="utf-8",
                cte=_cte(text),
                filename=attachment.filename,
            )
        else:
            message.add_attachment(
                attachment.data,
                maintype=maintype,
                subtype=subtype,
                filename=attachment.filename,
            )
    return message


def parse_message(
    raw: bytes,
    *,
    uid: int = 0,
    mailbox: str = "INBOX",
    flags: Sequence[str] = (),
    size: int = 0,
) -> Received:
    """把原始邮件字节解析成 :class:`Received`（附件只留元信息）。"""
    message = BytesParser(policy=policy.default).parsebytes(raw)
    return Received(
        uid=uid,
        mailbox=mailbox,
        subject=_header(message.get("Subject")),
        sender=_header(message.get("From")),
        to=_addresses(message.get_all("To", [])),
        cc=_addresses(message.get_all("Cc", [])),
        date=_date(message.get("Date")),
        size=size or len(raw),
        flags=tuple(flags),
        text=_body(message, "plain"),
        html=_body(message, "html"),
        attachments=tuple(_info(part) for part in _attachment_parts(message)),
    )


def extract_attachments(raw: bytes) -> tuple[Attachment, ...]:
    """从原始邮件中取出带内容的附件（按邮件里的顺序）。"""
    message = BytesParser(policy=policy.default).parsebytes(raw)
    return tuple(_attachment(part) for part in _attachment_parts(message))


# ------------------------------------------------------------ 内部工具


def _attachment_parts(message: Message) -> Iterator[Message]:
    yield from message.iter_attachments()


def _decoded_text(data: bytes) -> str | None:
    """能按 UTF-8 解码且不含 NUL 才当文本处理。"""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return None if "\x00" in text else text


def _cte(text: str) -> str | None:
    """含非 ASCII 时强制 base64，保证报文 7-bit 干净，不依赖服务器 8BITMIME。"""
    return None if text.isascii() else "base64"


def _attachment(part: Message) -> Attachment:
    payload = part.get_payload(decode=True)
    return Attachment(
        filename=part.get_filename() or "attachment",
        data=payload if isinstance(payload, bytes) else b"",
        mime_type=part.get_content_type(),
    )


def _info(part: Message) -> AttachmentInfo:
    payload = part.get_payload(decode=True)
    return AttachmentInfo(
        filename=safe_filename(part.get_filename() or "attachment"),
        size=len(payload) if isinstance(payload, bytes) else 0,
        mime_type=part.get_content_type(),
    )


def _header(value: str | None) -> str:
    """折叠头里的换行与多余空白。"""
    return " ".join(str(value).split()) if value else ""


def _addresses(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(address for _, address in getaddresses(values) if address)
    )


def _date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None


def _body(message: Message, subtype: str) -> str:
    """取指定子类型的正文；编码异常时按 charset 兜底解码。"""
    try:
        part = message.get_body(preferencelist=(subtype,))
    except (KeyError, ValueError):
        return ""
    if part is None:
        return ""
    try:
        content = part.get_content()
    except (LookupError, UnicodeDecodeError):
        payload = part.get_payload(decode=True)
        charset = part.get_content_charset() or "utf-8"
        return payload.decode(charset, "replace") if isinstance(payload, bytes) else ""
    return content if isinstance(content, str) else ""
