"""SMTP 发信客户端。

默认一次投递建一次连接；需要连续发多封时用 ``with`` 复用连接::

    with SmtpSender.from_account("work") as sender:
        sender.send(Mail(to="a@b.com", subject="hi", text="..."))
        sender.send(Mail(to="c@d.com", subject="hi", text="..."))
"""

from __future__ import annotations

import smtplib
import ssl
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Self

from .config import load_account
from .errors import MailAuthError, MailConfigError, MailSendError
from .mime import build_message
from .models import Mail, Security, SendResult, SmtpConfig

__all__ = ["SmtpSender"]


class SmtpSender:
    """同步 SMTP 发信器（SSL / STARTTLS / 明文）。"""

    def __init__(self, config: SmtpConfig) -> None:
        self.config = config
        self._connection: smtplib.SMTP | None = None

    @classmethod
    def from_account(
        cls, name: str | None = None, *, config: str | Path | None = None
    ) -> Self:
        """按名字从配置文件 / 环境变量加载账号。"""
        return cls(load_account(name, config=config).require_smtp())

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

    # ------------------------------------------------------------ 发送

    def send(self, mail: Mail, *, sender: str | None = None) -> SendResult:
        """发送一封邮件；``sender`` 可临时覆盖配置里的发件地址。"""
        from_addr = (sender or self.config.from_addr).strip()
        if not from_addr:
            raise MailConfigError("缺少发件人地址：请配置 sender 或 username")
        message = build_message(mail, sender=from_addr)
        recipients = mail.recipients
        with self._session() as connection:
            try:
                connection.send_message(
                    message, from_addr=from_addr, to_addrs=list(recipients)
                )
            except (smtplib.SMTPException, OSError) as exc:
                raise MailSendError(f"发送失败：{exc}") from exc
        return SendResult(
            sender=from_addr,
            recipients=recipients,
            subject=mail.subject,
            attachments=len(mail.attachments),
            message_id=str(message.get("Message-ID", "")),
        )

    def verify(self) -> str:
        """建连并发一次 NOOP，只验配置与认证，不投递任何邮件。"""
        with self._session() as connection:
            try:
                code, _ = connection.noop()
            except (smtplib.SMTPException, OSError) as exc:
                raise MailSendError(f"SMTP 校验失败：{exc}") from exc
        if code != 250:
            raise MailSendError(f"SMTP 校验失败：NOOP 返回 {code}")
        return f"{self.config.host}:{self.config.port}"

    # ------------------------------------------------------------ 内部

    def _connect(self) -> smtplib.SMTP:
        cfg = self.config
        context = ssl.create_default_context()
        try:
            if cfg.security is Security.SSL:
                connection: smtplib.SMTP = smtplib.SMTP_SSL(
                    cfg.host, cfg.port, timeout=cfg.timeout, context=context
                )
            else:
                connection = smtplib.SMTP(cfg.host, cfg.port, timeout=cfg.timeout)
                connection.ehlo()
                if cfg.security is Security.STARTTLS:
                    connection.starttls(context=context)
                    connection.ehlo()
        except (smtplib.SMTPException, OSError) as exc:
            raise MailSendError(
                f"连接 SMTP 服务器 {cfg.host}:{cfg.port} 失败：{exc}"
            ) from exc

        if cfg.username and cfg.password:
            try:
                connection.login(cfg.username, cfg.password)
            except smtplib.SMTPAuthenticationError as exc:
                with suppress(Exception):
                    connection.close()
                raise MailAuthError(f"SMTP 登录失败（{cfg.username}）：{exc}") from exc
            except (smtplib.SMTPException, OSError) as exc:
                with suppress(Exception):
                    connection.close()
                raise MailSendError(f"SMTP 登录失败：{exc}") from exc
        return connection

    @contextmanager
    def _session(self) -> Iterator[smtplib.SMTP]:
        """复用已建立的连接，否则临时建连并及时关闭。"""
        if self._connection is not None:
            yield self._connection
            return
        connection = self._connect()
        try:
            yield connection
        finally:
            _shutdown(connection)


def _shutdown(connection: smtplib.SMTP) -> None:
    """QUIT 优雅收尾，失败时直接关 socket。"""
    with suppress(Exception):
        connection.quit()
    with suppress(Exception):
        connection.close()
