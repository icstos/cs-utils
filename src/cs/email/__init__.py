"""邮件收发工具：账号配置、SMTP 发信、IMAP 收信，附 MCP / CLI 入口。

三种调用方式：

- 类方法：:class:`SmtpSender` / :class:`ImapClient`（支持 ``with`` 复用连接）；
- 便捷函数：:func:`send` / :func:`inbox`；
- MCP：``python -m cs.email serve``，工具见 :mod:`cs.email.mcp_server`。

>>> from cs.email import Mail, send
>>> send(Mail(to="a@b.com", subject="hi", text="hello"))

>>> from cs.email import ImapClient
>>> with ImapClient.from_account("work") as client:
...     mails = client.messages(limit=5, unseen=True)

账号来自 TOML 配置或 ``CS_MAIL_*`` 环境变量，见 :mod:`cs.email.config`。
"""

from __future__ import annotations

from pathlib import Path

from .config import (
    SAMPLE_CONFIG,
    account_from_env,
    load_account,
    load_profiles,
    resolve_config_path,
    resolve_secret,
)
from .errors import (
    MailAttachmentError,
    MailAuthError,
    MailConfigError,
    MailError,
    MailReceiveError,
    MailSendError,
)
from .imap import ImapClient, search_criteria
from .mime import build_message, extract_attachments, parse_message
from .models import (
    Account,
    AddressInput,
    Attachment,
    AttachmentInfo,
    ImapConfig,
    Mail,
    Received,
    Security,
    SendResult,
    SmtpConfig,
    normalize_addresses,
)
from .smtp import SmtpSender

__all__ = [
    "SAMPLE_CONFIG",
    "Account",
    "AddressInput",
    "Attachment",
    "AttachmentInfo",
    "ImapClient",
    "ImapConfig",
    "Mail",
    "MailAttachmentError",
    "MailAuthError",
    "MailConfigError",
    "MailError",
    "MailReceiveError",
    "MailSendError",
    "Received",
    "Security",
    "SendResult",
    "SmtpConfig",
    "SmtpSender",
    "account_from_env",
    "build_message",
    "extract_attachments",
    "inbox",
    "load_account",
    "load_profiles",
    "normalize_addresses",
    "parse_message",
    "resolve_config_path",
    "resolve_secret",
    "search_criteria",
    "send",
]


def send(
    mail: Mail,
    *,
    account: str | None = None,
    config: str | Path | None = None,
    sender: str | None = None,
) -> SendResult:
    """一次性发信：按账号加载配置 → 建连 → 投递。"""
    return SmtpSender.from_account(account, config=config).send(mail, sender=sender)


def inbox(
    *,
    account: str | None = None,
    config: str | Path | None = None,
    mailbox: str | None = None,
    limit: int = 10,
    unseen: bool = False,
    criteria: str = "ALL",
) -> list[Received]:
    """一次性读取最近邮件（只取头部）。"""
    with ImapClient.from_account(account, config=config) as client:
        return client.messages(
            mailbox=mailbox, limit=limit, unseen=unseen, criteria=criteria
        )
