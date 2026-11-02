"""把 :mod:`cs.email` 暴露为 MCP 工具（FastMCP ≥ 4）。

启动方式::

    python -m cs.email serve                    # stdio，供 MCP 客户端拉起
    python -m cs.email serve --transport http --port 8000
    fastmcp run src/cs/email/mcp_server.py:mcp

约定
----
* 工具名统一 ``mail_*``；成功返回 ``{"ok": true, ...}``；
* 密码永不返回（``mail_account_show`` 里显示为 ``***``）；
* 业务异常统一转成 ``ToolError``，不把 traceback 泄露给模型；
* 账号从 TOML 配置或 ``CS_MAIL_*`` 环境变量解析，见 :mod:`cs.email.config`。
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from .config import account_from_env, load_account, load_profiles
from .errors import MailError
from .imap import ImapClient, search_criteria
from .models import Mail
from .smtp import SmtpSender

__all__ = ["create_server", "mcp"]

_SERVER_NAME = "cs-email"
_INSTRUCTIONS = """\
本服务器收发邮件：查看账号配置、发送带附件/HTML 的邮件、列出文件夹与邮件、
读信、下载附件、标记 / 移动 / 删除邮件。密码只从本地配置读取，任何工具都不会返回密码。

使用建议：
1. 先 mail_accounts 看有哪些账号，mail_account_show 看某账号的 SMTP/IMAP 配置；
2. 首次使用先 mail_verify 验证连通性，避免盲目重试发送；
3. 发信前确认真实收件人，mail_send 会立刻投递，不可撤回；
4. 列邮件用 mail_list（只取头部，默认 10 封，支持 unseen 与 query），
   需要正文时再对具体 uid 调 mail_read；
5. 附件先 mail_attachments 看清单，再用 mail_download 存到本地目录；
6. mail_delete 默认立即 expunge，不可恢复，除非用户明确要求否则不要调用。
"""


def _guard[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """把领域异常转换成 MCP 客户端能看懂的错误消息。"""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except (MailError, OSError, ValueError) as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


def _done(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def _client(account: str | None, config: str | None) -> ImapClient:
    return ImapClient.from_account(account, config=config)


def create_server(
    name: str = _SERVER_NAME, *, instructions: str = _INSTRUCTIONS
) -> FastMCP:
    """构造一个独立的服务器实例（便于测试或多实例部署）。"""
    server: FastMCP = FastMCP(name, instructions=instructions)

    # ---------------------------------------------------------- 配置
    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def mail_accounts(config: str | None = None) -> dict[str, Any]:
        """列出配置里的账号名、默认账号与环境变量是否可用（不含密码）。"""
        file = load_profiles(config)
        return _done(
            config=str(file.path) if file.path else None,
            default=file.default or None,
            accounts=sorted(file.profiles),
            env_available=account_from_env() is not None,
        )

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def mail_account_show(
        account: str | None = None, config: str | None = None
    ) -> dict[str, Any]:
        """查看某个账号的配置详情（密码显示为 ***）。"""
        return _done(**load_account(account, config=config).to_dict())

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def mail_verify(
        account: str | None = None, config: str | None = None
    ) -> dict[str, Any]:
        """验证账号连通性：SMTP NOOP + IMAP 登录并统计文件夹数（不投递邮件）。"""
        resolved = load_account(account, config=config)
        result: dict[str, Any] = {"account": resolved.name}
        if resolved.smtp is not None:
            result["smtp"] = SmtpSender(resolved.require_smtp()).verify()
        if resolved.imap is not None:
            with ImapClient(resolved.require_imap()) as client:
                result["imap"] = client.verify()
        return _done(**result)

    # ---------------------------------------------------------- 发送
    @server.tool()
    @_guard
    def mail_send(
        to: str | list[str],
        subject: str,
        body: str = "",
        html: str | None = None,
        cc: str | list[str] | None = None,
        bcc: str | list[str] | None = None,
        attachments: list[str] | None = None,
        account: str | None = None,
        config: str | None = None,
        sender: str | None = None,
        reply_to: str = "",
    ) -> dict[str, Any]:
        """发送一封邮件（立即可撤回不了），返回收件人清单与 Message-ID。

        - body 为纯文本正文；给 html 时同时发送纯文本 + HTML 两个备选版本；
        - attachments 为本地文件绝对路径列表，会自动推断 MIME 类型；
        - sender 可临时覆盖配置里的发件地址。
        """
        resolved = load_account(account, config=config)
        mail = Mail(
            subject=subject,
            text=body,
            html=html,
            to=to,
            cc=cc,
            bcc=bcc,
            attachments=tuple(attachments or ()),
            reply_to=reply_to,
        )
        result = SmtpSender(resolved.require_smtp()).send(mail, sender=sender)
        return _done(**result.to_dict())

    # ---------------------------------------------------------- 收信
    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def mail_folders(
        account: str | None = None,
        config: str | None = None,
        pattern: str = "*",
    ) -> dict[str, Any]:
        """列出邮箱文件夹（IMAP 目录）。"""
        with _client(account, config) as client:
            folders = client.folders(pattern=pattern)
        return _done(folders=folders, count=len(folders))

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def mail_list(
        account: str | None = None,
        config: str | None = None,
        mailbox: str | None = None,
        limit: int = 10,
        unseen: bool = False,
        query: str = "",
    ) -> dict[str, Any]:
        """列出最近邮件（只取头部：主题/发件人/日期/大小/标记，不含正文）。

        query 为简单关键词时按主题或发件人模糊匹配；也可以直接写 IMAP 语法，
        例如 ``SINCE 01-Jan-2026``、``SUBJECT "账单"``。
        """
        with _client(account, config) as client:
            mails = client.messages(
                mailbox=mailbox,
                limit=limit,
                unseen=unseen,
                criteria=search_criteria(query),
            )
            box = mailbox or client.config.mailbox
        return _done(
            mailbox=box,
            count=len(mails),
            messages=[item.to_dict() for item in mails],
        )

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def mail_read(
        uid: int,
        account: str | None = None,
        config: str | None = None,
        mailbox: str | None = None,
        mark_seen: bool = False,
        max_chars: int = 20000,
    ) -> dict[str, Any]:
        """读取一封邮件的完整内容（纯文本 + HTML + 附件清单），默认不改已读状态。"""
        with _client(account, config) as client:
            mail = client.fetch(uid, mailbox=mailbox, mark_seen=mark_seen)
        return _done(**mail.to_dict(max_chars=max_chars))

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def mail_attachments(
        uid: int,
        account: str | None = None,
        config: str | None = None,
        mailbox: str | None = None,
    ) -> dict[str, Any]:
        """列出某封邮件的附件（文件名 / 大小 / 类型），不下载内容。"""
        with _client(account, config) as client:
            mail = client.fetch(uid, mailbox=mailbox)
        return _done(
            uid=uid,
            count=len(mail.attachments),
            attachments=[item.to_dict() for item in mail.attachments],
        )

    @server.tool()
    @_guard
    def mail_download(
        uid: int,
        output_dir: str,
        account: str | None = None,
        config: str | None = None,
        mailbox: str | None = None,
        name: str | None = None,
        index: int = 0,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """把某个附件保存到本地目录（同名文件默认不覆盖），返回路径与大小。"""
        with _client(account, config) as client:
            path = client.download(
                uid,
                output_dir,
                name=name,
                index=index,
                mailbox=mailbox,
                overwrite=overwrite,
            )
        return _done(path=str(path), size=path.stat().st_size)

    # ---------------------------------------------------------- 整理
    @server.tool()
    @_guard
    def mail_mark(
        uid: int,
        flag: str = "seen",
        value: bool = True,
        account: str | None = None,
        config: str | None = None,
        mailbox: str | None = None,
    ) -> dict[str, Any]:
        """加 / 去一个 IMAP 标记：seen、flagged、answered、deleted（value=false 表示去掉）。"""
        with _client(account, config) as client:
            client.mark(uid, flag, value=value, mailbox=mailbox)
        return _done(uid=uid, flag=flag, value=value)

    @server.tool()
    @_guard
    def mail_move(
        uid: int,
        folder: str,
        account: str | None = None,
        config: str | None = None,
        mailbox: str | None = None,
    ) -> dict[str, Any]:
        """把邮件移动到另一个文件夹（服务器不支持 MOVE 时自动退回 copy+delete）。"""
        with _client(account, config) as client:
            client.move(uid, folder, mailbox=mailbox)
        return _done(uid=uid, folder=folder)

    @server.tool()
    @_guard
    def mail_delete(
        uid: int,
        account: str | None = None,
        config: str | None = None,
        mailbox: str | None = None,
        expunge: bool = True,
    ) -> dict[str, Any]:
        """删除邮件（打 ``\\Deleted`` 标记，默认立即 expunge，不可恢复）。"""
        with _client(account, config) as client:
            client.delete(uid, mailbox=mailbox, expunge=expunge)
        return _done(uid=uid, expunged=expunge)

    return server


mcp = create_server()
