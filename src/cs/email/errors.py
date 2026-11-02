"""邮件模块的异常层级。

与 :mod:`cs.config`、:mod:`cs.cv_draw` 一致：领域异常统一继承
:class:`MailError`，MCP 层再把它们转换成 ``ToolError``，避免 traceback 泄漏给模型。
需要兼容旧代码时可以直接捕获内建类型：

* :class:`MailConfigError` 也是 ``ValueError``；
* :class:`MailSendError` / :class:`MailReceiveError` 也是 ``OSError``。
"""

from __future__ import annotations

__all__ = [
    "MailAttachmentError",
    "MailAuthError",
    "MailConfigError",
    "MailError",
    "MailReceiveError",
    "MailSendError",
]


class MailError(Exception):
    """邮件相关错误的基类。"""


class MailConfigError(MailError, ValueError):
    """账号配置缺失、字段非法或配置文件无法解析。"""


class MailAuthError(MailError):
    """SMTP / IMAP 认证失败。"""


class MailSendError(MailError, OSError):
    """连接、投递或附件读取失败。"""


class MailReceiveError(MailError, OSError):
    """IMAP 连接、检索或读取失败。"""


class MailAttachmentError(MailError, OSError):
    """附件不存在、下标越界或落盘失败。"""
