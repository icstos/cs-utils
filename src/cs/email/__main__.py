"""命令行入口：``python -m cs.email ...``（也可用 ``cs-email``）。

示例::

    python -m cs.email init mail.toml                     # 生成配置骨架
    python -m cs.email accounts --config mail.toml
    python -m cs.email verify --account work
    python -m cs.email send --to a@b.com --subject hi --body "hello"
    python -m cs.email send --to a@b.com --subject hi --attach a.pdf --html "<b>hi</b>"
    python -m cs.email list -n 5 --unseen --query 账单
    python -m cs.email read 12 --mark-seen
    python -m cs.email download 12 -o ./att --name report.pdf --overwrite
    python -m cs.email mark 12 --flag flagged
    python -m cs.email move 12 Archive
    python -m cs.email delete 12
    python -m cs.email serve --transport http --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import SAMPLE_CONFIG, load_account, load_profiles
from .errors import MailError
from .imap import ImapClient, search_criteria
from .models import Mail
from .smtp import SmtpSender

__all__ = ["main"]

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def _common() -> argparse.ArgumentParser:
    """``--config`` / ``--account`` 每个子命令都要有。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", help="配置 TOML 路径")
    parser.add_argument("--account", help="账号名（配置里的 profile）")
    return parser


def _build_parser() -> argparse.ArgumentParser:
    common = _common()
    parser = argparse.ArgumentParser(
        prog="python -m cs.email",
        description="邮件收发：账号配置、发信、读收件箱、下载附件、整理邮件。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="生成配置骨架")
    init.add_argument("path", nargs="?", default="mail.toml", help="输出路径")
    init.add_argument("--force", action="store_true", help="允许覆盖已存在文件")

    sub.add_parser("accounts", parents=[common], help="列出配置里的账号")

    sub.add_parser("show", parents=[common], help="查看账号配置（密码打码）")

    sub.add_parser("verify", parents=[common], help="验证 SMTP/IMAP 连通性")

    send = sub.add_parser("send", parents=[common], help="发送邮件")
    send.add_argument("--to", action="append", required=True, help="收件人，可重复")
    send.add_argument("--subject", required=True, help="主题")
    send.add_argument("--body", default="", help="纯文本正文")
    send.add_argument("--body-file", help="从文件读取纯文本正文")
    send.add_argument("--html", help="HTML 正文")
    send.add_argument("--html-file", help="从文件读取 HTML 正文")
    send.add_argument("--cc", action="append", default=[], help="抄送，可重复")
    send.add_argument("--bcc", action="append", default=[], help="密送，可重复")
    send.add_argument("--attach", action="append", default=[], help="附件路径，可重复")
    send.add_argument("--sender", help="临时覆盖发件地址")

    folders = sub.add_parser("folders", parents=[common], help="列出文件夹")
    folders.add_argument("--pattern", default="*", help="IMAP LIST 模式")

    list_ = sub.add_parser("list", parents=[common], help="列出最近邮件")
    list_.add_argument("-n", "--limit", type=int, default=10, help="数量，0 表示不限制")
    list_.add_argument("--mailbox", help="文件夹，默认账号配置里的 mailbox")
    list_.add_argument("--unseen", action="store_true", help="只看未读")
    list_.add_argument("--query", default="", help="关键词或原生 IMAP 检索条件")

    read = sub.add_parser("read", parents=[common], help="读取一封邮件")
    read.add_argument("uid", type=int)
    read.add_argument("--mailbox")
    read.add_argument("--mark-seen", action="store_true", help="读后置为已读")
    read.add_argument("--html", action="store_true", help="输出 HTML 正文")
    read.add_argument("--json", action="store_true", help="输出完整 JSON")

    attachments = sub.add_parser("attachments", parents=[common], help="列出附件")
    attachments.add_argument("uid", type=int)
    attachments.add_argument("--mailbox")

    download = sub.add_parser("download", parents=[common], help="下载附件")
    download.add_argument("uid", type=int)
    download.add_argument("-o", "--output", default=".", help="保存目录")
    download.add_argument("--name", help="按文件名选择附件")
    download.add_argument("--index", type=int, default=0, help="按顺序选择附件")
    download.add_argument("--mailbox")
    download.add_argument("--overwrite", action="store_true")

    mark = sub.add_parser("mark", parents=[common], help="加/去标记")
    mark.add_argument("uid", type=int)
    mark.add_argument(
        "--flag", default="seen", help="seen / flagged / answered / deleted"
    )
    mark.add_argument("--off", action="store_true", help="去掉该标记")
    mark.add_argument("--mailbox")

    move = sub.add_parser("move", parents=[common], help="移动邮件")
    move.add_argument("uid", type=int)
    move.add_argument("folder", help="目标文件夹")
    move.add_argument("--mailbox")

    delete = sub.add_parser("delete", parents=[common], help="删除邮件")
    delete.add_argument("uid", type=int)
    delete.add_argument("--mailbox")
    delete.add_argument("--no-expunge", action="store_true", help="只打标记不立即清理")

    serve = sub.add_parser("serve", help="以 MCP 服务方式启动")
    serve.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--name", default="cs-email")

    return parser


def _dump(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _read_text(inline: str | None, file: str | None) -> str:
    if file:
        return Path(file).expanduser().read_text(encoding="utf-8")
    return inline or ""


def _dispatch(args: argparse.Namespace) -> int:
    match args.command:
        case "init":
            target = Path(args.path).expanduser()
            if target.exists() and not args.force:
                print(f"已存在：{target}（需要覆盖请加 --force）", file=sys.stderr)
                return 1
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(SAMPLE_CONFIG, encoding="utf-8")
            print(target)
            return 0

        case "accounts":
            file = load_profiles(args.config)
            _dump(
                {
                    "config": str(file.path) if file.path else None,
                    "default": file.default or None,
                    "accounts": sorted(file.profiles),
                }
            )
            return 0

        case "show":
            _dump(load_account(args.account, config=args.config).to_dict())
            return 0

        case "verify":
            account = load_account(args.account, config=args.config)
            result: dict[str, Any] = {"account": account.name}
            if account.smtp is not None:
                result["smtp"] = SmtpSender(account.require_smtp()).verify()
            if account.imap is not None:
                with ImapClient(account.require_imap()) as client:
                    result["imap"] = client.verify()
            _dump(result)
            return 0

        case "send":
            mail = Mail(
                subject=args.subject,
                text=_read_text(args.body, args.body_file),
                html=_read_text(args.html, args.html_file) or None,
                to=args.to,
                cc=args.cc or None,
                bcc=args.bcc or None,
                attachments=tuple(args.attach),
            )
            sender = SmtpSender.from_account(args.account, config=args.config)
            _dump(sender.send(mail, sender=args.sender).to_dict())
            return 0

        case "folders":
            with ImapClient.from_account(args.account, config=args.config) as client:
                _dump({"folders": client.folders(pattern=args.pattern)})
            return 0

        case "list":
            with ImapClient.from_account(args.account, config=args.config) as client:
                mails = client.messages(
                    mailbox=args.mailbox,
                    limit=args.limit,
                    unseen=args.unseen,
                    criteria=search_criteria(args.query),
                )
            _dump({"count": len(mails), "messages": [m.to_dict() for m in mails]})
            return 0

        case "read":
            with ImapClient.from_account(args.account, config=args.config) as client:
                mail = client.fetch(
                    args.uid, mailbox=args.mailbox, mark_seen=args.mark_seen
                )
            if args.json:
                _dump(mail.to_dict())
            elif args.html:
                print(mail.html or mail.text)
            else:
                print(f"主题: {mail.subject}")
                print(f"发件人: {mail.sender}")
                print(f"日期: {mail.date}")
                print(f"附件: {len(mail.attachments)} 个")
                print("-" * 60)
                print(mail.text or mail.html)
            return 0

        case "attachments":
            with ImapClient.from_account(args.account, config=args.config) as client:
                mail = client.fetch(args.uid, mailbox=args.mailbox)
            _dump([item.to_dict() for item in mail.attachments])
            return 0

        case "download":
            with ImapClient.from_account(args.account, config=args.config) as client:
                path = client.download(
                    args.uid,
                    args.output,
                    name=args.name,
                    index=args.index,
                    mailbox=args.mailbox,
                    overwrite=args.overwrite,
                )
            _dump({"path": str(path), "size": path.stat().st_size})
            return 0

        case "mark":
            with ImapClient.from_account(args.account, config=args.config) as client:
                client.mark(
                    args.uid, args.flag, value=not args.off, mailbox=args.mailbox
                )
            _dump({"uid": args.uid, "flag": args.flag, "value": not args.off})
            return 0

        case "move":
            with ImapClient.from_account(args.account, config=args.config) as client:
                client.move(args.uid, args.folder, mailbox=args.mailbox)
            _dump({"uid": args.uid, "folder": args.folder})
            return 0

        case "delete":
            with ImapClient.from_account(args.account, config=args.config) as client:
                client.delete(
                    args.uid, mailbox=args.mailbox, expunge=not args.no_expunge
                )
            _dump({"uid": args.uid, "expunged": not args.no_expunge})
            return 0

        case "serve":
            from .mcp_server import create_server

            server = create_server(name=args.name)
            if args.transport == "http":
                server.run(transport="http", host=args.host, port=args.port)
            else:
                server.run()
            return 0

    return 2  # pragma: no cover - argparse 已限制取值范围


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except MailError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
