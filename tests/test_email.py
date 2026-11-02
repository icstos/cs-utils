"""``cs.email`` 的单元测试：MIME 组装/解析、SMTP、IMAP、配置与 MCP 工具。"""

from __future__ import annotations

import asyncio
import imaplib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs.email import (
    Account,
    Attachment,
    ImapClient,
    ImapConfig,
    Mail,
    MailAttachmentError,
    MailAuthError,
    MailConfigError,
    MailError,
    MailReceiveError,
    MailSendError,
    SmtpConfig,
    SmtpSender,
    load_account,
    normalize_addresses,
    parse_message,
    resolve_secret,
)
from cs.email.config import account_from_env
from cs.email.mime import build_message, extract_attachments

RAW_MAIL = (
    "From: Alice <alice@example.com>\r\n"
    "To: Bob <bob@example.com>, carol@example.com\r\n"
    "Cc: Dave <dave@example.com>\r\n"
    "Subject: =?utf-8?b?5rWL6K+V6YKu5Lu2?=\r\n"
    "Date: Tue, 14 Apr 2026 09:30:00 +0800\r\n"
    "Message-ID: <abc@example.com>\r\n"
    "MIME-Version: 1.0\r\n"
    'Content-Type: multipart/mixed; boundary="BOUND"\r\n'
    "\r\n"
    "--BOUND\r\n"
    'Content-Type: multipart/alternative; boundary="ALT"\r\n'
    "\r\n"
    "--ALT\r\n"
    "Content-Type: text/plain; charset=utf-8\r\n"
    "\r\n"
    "纯文本正文\r\n"
    "--ALT\r\n"
    "Content-Type: text/html; charset=utf-8\r\n"
    "\r\n"
    "<p>HTML 正文</p>\r\n"
    "--ALT--\r\n"
    "--BOUND\r\n"
    "Content-Type: application/pdf\r\n"
    'Content-Disposition: attachment; filename="report.pdf"\r\n'
    "Content-Transfer-Encoding: base64\r\n"
    "\r\n"
    "aGVsbG8="
    "\r\n--BOUND--\r\n"
).encode()


# ============================================================
# 模型
# ============================================================


def test_normalize_addresses_merges_forms():
    assert normalize_addresses("a@x.com") == ("a@x.com",)
    assert normalize_addresses("a@x.com, b@x.com;c@x.com") == (
        "a@x.com",
        "b@x.com",
        "c@x.com",
    )
    assert normalize_addresses(["a@x.com", " a@x.com ", "b@x.com"]) == (
        "a@x.com",
        "b@x.com",
    )
    assert normalize_addresses(None) == ()


def test_mail_recipients_dedupes_to_cc_bcc():
    mail = Mail(to="a@x.com", cc=["a@x.com", "b@x.com"], bcc="c@x.com")
    assert mail.recipients == ("a@x.com", "b@x.com", "c@x.com")


def test_config_defaults_ports_and_repr_hides_password():
    smtp = SmtpConfig("smtp.example.com", username="me@example.com", password="secret")
    assert smtp.port == 465
    assert smtp.from_addr == "me@example.com"
    assert "secret" not in repr(smtp)

    starttls = SmtpConfig("smtp.example.com", security="starttls")
    assert starttls.port == 587
    assert ImapConfig("imap.example.com").port == 993


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"host": ""}, "主机"),
        ({"host": "h", "port": 70000}, "端口"),
        ({"host": "h", "security": "tls"}, "security"),
        ({"host": "h", "password": "x"}, "username"),
        ({"host": "h", "timeout": 0}, "timeout"),
    ],
)
def test_smtp_config_rejects_bad_values(kwargs, match):
    with pytest.raises(MailConfigError, match=match):
        SmtpConfig(**kwargs)


def test_account_requires_protocol():
    with pytest.raises(MailConfigError, match="SMTP"):
        Account().require_smtp()


def test_attachment_from_path_guesses_type_and_saves(tmp_path):
    source = tmp_path / "note.txt"
    source.write_text("content", encoding="utf-8")
    attachment = Attachment.from_path(source)
    assert attachment.mime_type == "text/plain"
    assert attachment.subtype == ("text", "plain")
    assert attachment.size == 7

    target = attachment.save(tmp_path / "out")
    assert target.read_text(encoding="utf-8") == "content"
    with pytest.raises(MailError, match="已存在"):
        attachment.save(tmp_path / "out")


def test_attachment_filename_is_sanitized():
    assert Attachment(filename="../etc/passwd").filename == "_etc_passwd"
    assert Attachment(filename="").filename == "attachment"


# ============================================================
# MIME
# ============================================================


def test_build_message_headers_body_and_attachment(tmp_path):
    file = tmp_path / "a.txt"
    file.write_text("附件内容", encoding="utf-8")
    mail = Mail(
        subject="主题",
        text="纯文本",
        html="<p>HTML</p>",
        to=["to@example.com", "second@example.com"],
        cc="cc@example.com",
        bcc="hidden@example.com",
        attachments=[file],
    )
    message = build_message(mail, sender="me@example.com")

    assert message["To"] == "to@example.com, second@example.com"
    assert message["Cc"] == "cc@example.com"
    assert "Bcc" not in message
    assert message["Message-ID"] and message["Date"]
    assert message.get_body(preferencelist=("plain",)).get_content().strip() == "纯文本"
    assert (
        message.get_body(preferencelist=("html",)).get_content().strip()
        == "<p>HTML</p>"
    )
    part = next(iter(message.iter_attachments()))
    assert part.get_filename() == "a.txt"
    assert part.get_content().strip() == "附件内容"


def test_build_message_requires_recipient():
    with pytest.raises(MailSendError, match="收件人"):
        build_message(Mail(subject="s", text="t"), sender="me@example.com")


def test_parse_message_extracts_bodies_and_attachments():
    received = parse_message(RAW_MAIL, uid=42, flags=["\\Seen"])
    assert received.uid == 42
    assert received.subject == "测试邮件"
    assert received.sender == "Alice <alice@example.com>"
    assert received.to == ("bob@example.com", "carol@example.com")
    assert received.cc == ("dave@example.com",)
    assert received.text.strip() == "纯文本正文"
    assert received.html.strip() == "<p>HTML 正文</p>"
    assert received.date is not None and received.date.year == 2026
    assert received.is_seen
    assert [item.filename for item in received.attachments] == ["report.pdf"]
    assert received.attachments[0].mime_type == "application/pdf"

    attachments = extract_attachments(RAW_MAIL)
    assert attachments[0].data == b"hello"


def test_received_dict_truncates_body():
    received = parse_message(RAW_MAIL)
    payload = received.to_dict(max_chars=2)
    assert payload["text"] == "纯文"
    assert payload["truncated"] is True
    assert payload["attachments"][0]["filename"] == "report.pdf"


# ============================================================
# SMTP
# ============================================================


class FakeSMTP:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.message = None
        self.recipients = None
        self.started_tls = False
        self.logged_in = None
        self.noop_code = 250
        self.quit_called = False

    def ehlo(self):
        return 250, b"ok"

    def starttls(self, **kwargs):
        self.started_tls = True

    def login(self, username, password):
        self.logged_in = (username, password)

    def send_message(self, message, *, from_addr, to_addrs):
        self.message = message
        self.recipients = (from_addr, to_addrs)

    def noop(self):
        return self.noop_code, b"ok"

    def quit(self):
        self.quit_called = True

    def close(self):
        self.closed = True


def test_smtp_send_over_ssl(monkeypatch):
    smtp = FakeSMTP()

    def factory(*args, **kwargs):
        smtp.args = args
        smtp.kwargs = kwargs
        return smtp

    monkeypatch.setattr("cs.email.smtp.smtplib.SMTP_SSL", factory)

    mail = Mail(
        subject="Subject",
        text="Plain text",
        to=["to@example.com", "second@example.com"],
        cc="cc@example.com",
        bcc=["hidden@example.com"],
    )
    result = SmtpSender(
        SmtpConfig("smtp.example.com", username="u", password="p")
    ).send(mail)

    assert smtp.args == ("smtp.example.com", 465)
    assert smtp.logged_in == ("u", "p")
    assert smtp.recipients == (
        "u",
        [
            "to@example.com",
            "second@example.com",
            "cc@example.com",
            "hidden@example.com",
        ],
    )
    assert smtp.message["To"] == "to@example.com, second@example.com"
    assert result.recipients[0] == "to@example.com"
    assert result.message_id.startswith("<")
    assert smtp.quit_called


def test_smtp_starttls_and_sender_override(monkeypatch):
    smtp = FakeSMTP()
    monkeypatch.setattr("cs.email.smtp.smtplib.SMTP", lambda *a, **kw: smtp)

    sender = SmtpSender(
        SmtpConfig(
            "smtp.example.com",
            port=587,
            username="smtp-user",
            password="secret",
            security="starttls",
        )
    )
    result = sender.send(
        Mail(to="to@example.com", subject="s", text="t"), sender="boss@example.com"
    )

    assert smtp.started_tls
    assert smtp.logged_in == ("smtp-user", "secret")
    assert smtp.message["From"] == "boss@example.com"
    assert result.sender == "boss@example.com"


def test_smtp_reuses_connection_in_context(monkeypatch):
    created: list[FakeSMTP] = []

    def factory(*args, **kwargs):
        created.append(FakeSMTP())
        return created[-1]

    monkeypatch.setattr("cs.email.smtp.smtplib.SMTP_SSL", factory)

    with SmtpSender(SmtpConfig("h", username="u", password="p")) as sender:
        sender.send(Mail(to="a@x.com", subject="1", text="a"))
        sender.send(Mail(to="b@x.com", subject="2", text="b"))

    assert len(created) == 1
    assert created[0].quit_called


def test_smtp_missing_sender_raises(monkeypatch):
    monkeypatch.setattr("cs.email.smtp.smtplib.SMTP_SSL", lambda *a, **kw: FakeSMTP())
    with pytest.raises(MailConfigError, match="发件人"):
        SmtpSender(SmtpConfig("h")).send(Mail(to="a@x.com"))


def test_smtp_login_failure_is_auth_error(monkeypatch):
    import smtplib

    smtp = FakeSMTP()

    def auth_error(username, password):
        raise smtplib.SMTPAuthenticationError(535, b"bad credentials")

    smtp.login = auth_error
    monkeypatch.setattr("cs.email.smtp.smtplib.SMTP_SSL", lambda *a, **kw: smtp)

    with pytest.raises(MailAuthError, match="登录失败"):
        SmtpSender(SmtpConfig("h", username="u", password="bad")).send(
            Mail(to="a@x.com")
        )


def test_smtp_verify_noop(monkeypatch):
    smtp = FakeSMTP()
    monkeypatch.setattr("cs.email.smtp.smtplib.SMTP_SSL", lambda *a, **kw: smtp)
    assert SmtpSender(SmtpConfig("h", username="u", password="p")).verify() == "h:465"

    smtp.noop_code = 421
    with pytest.raises(MailSendError, match="NOOP"):
        SmtpSender(SmtpConfig("h", username="u", password="p")).verify()


# ============================================================
# IMAP
# ============================================================


class FakeIMAP:
    """只实现被测代码用到的 IMAP4 子集。"""

    def __init__(self, messages: dict[int, bytes] | None = None):
        self.messages = messages or {}
        self.state = "NONAUTH"
        self.logged_in = None
        self.selected = None
        self.stored: list[tuple] = []
        self.copied: list[tuple] = []
        self.expunged = 0
        self.moved: list[tuple] = []
        self.support_move = False
        self.search_error = False

    def login(self, username, password):
        self.logged_in = (username, password)
        self.state = "AUTH"
        return "OK", [b"LOGIN completed"]

    def select(self, mailbox, readonly=False):
        self.selected = mailbox
        self.state = "SELECTED"
        return "OK", [b"1"]

    def list(self, pattern="*"):
        return "OK", [
            b'(\\HasNoChildren) "/" "INBOX"',
            b'(\\HasNoChildren) "/" "Sent"',
        ]

    def uid(self, command, *args):
        match command.upper():
            case "SEARCH":
                if self.search_error:
                    return "NO", [b"bad criteria"]
                return "OK", [
                    b" ".join(str(uid).encode() for uid in sorted(self.messages))
                ]
            case "FETCH":
                uid, spec = int(args[0]), args[1]
                raw = self.messages[uid]
                if "HEADER" in spec:
                    raw = raw.split(b"\r\n\r\n", 1)[0] + b"\r\n\r\n"
                literal = b"1 (BODY[] {%d}" % len(raw)
                flags = b"FLAGS (\\Seen)" if spec.startswith("BODY[") else b"FLAGS ()"
                return "OK", [
                    (literal, raw),
                    flags,
                    b"RFC822.SIZE %d)" % len(self.messages[uid]),
                ]
            case "STORE":
                self.stored.append(args)
                return "OK", [b"STORE completed"]
            case "COPY":
                self.copied.append(args)
                return "OK", [b"COPY completed"]
            case "MOVE":
                if not self.support_move:
                    raise imaplib.IMAP4.error("MOVE unsupported")
                self.moved.append(args)
                return "OK", [b"MOVE completed"]
        return "NO", [b"unsupported"]

    def expunge(self):
        self.expunged += 1
        return "OK", [b"EXPUNGE completed"]

    def close(self):
        self.state = "AUTH"

    def shutdown(self):
        self.state = "LOGOUT"

    def logout(self):
        self.state = "LOGOUT"


@pytest.fixture
def fake_imap(monkeypatch):
    """把 imaplib.IMAP4_SSL 换成可检查的假对象。"""
    holder: dict[str, object] = {"client": FakeIMAP({7: RAW_MAIL})}

    def factory(*args, **kwargs):
        holder["args"] = (args, kwargs)
        return holder["client"]

    monkeypatch.setattr("cs.email.imap.imaplib.IMAP4_SSL", factory)
    return holder


def _client(**kwargs) -> ImapClient:
    return ImapClient(
        ImapConfig("imap.example.com", username="u", password="p", **kwargs)
    )


def test_imap_folders_and_login(fake_imap):
    with _client() as client:
        assert client.folders() == ["INBOX", "Sent"]
    assert fake_imap["client"].logged_in == ("u", "p")
    assert fake_imap["client"].state == "LOGOUT"


def test_imap_messages_reads_headers(fake_imap):
    mails = _client().messages(limit=5)
    assert len(mails) == 1
    assert mails[0].uid == 7
    assert mails[0].subject == "测试邮件"
    assert mails[0].text == ""  # 列表只取头部


def test_imap_fetch_returns_full_body(fake_imap):
    mail = _client().fetch(7)
    assert mail.text.strip() == "纯文本正文"
    assert mail.attachments[0].filename == "report.pdf"
    assert mail.size > 0


def test_imap_fetch_mark_seen_adds_flag(fake_imap):
    mail = _client().fetch(7, mark_seen=True)
    assert mail.is_seen


def test_imap_attachment_and_download(fake_imap, tmp_path):
    client = _client()
    attachment = client.attachment(7)
    assert attachment.data == b"hello"

    path = client.download(7, tmp_path)
    assert path.read_bytes() == b"hello"
    with pytest.raises(MailError, match="已存在"):
        client.download(7, tmp_path)
    assert client.download(7, tmp_path, overwrite=True) == path

    with pytest.raises(MailAttachmentError, match="没有名为"):
        client.attachment(7, name="missing.txt")
    with pytest.raises(MailAttachmentError, match="越界"):
        client.attachment(7, index=3)


def test_imap_mark_move_delete(fake_imap):
    client = _client()
    client.mark(7, "flagged")
    assert fake_imap["client"].stored[-1][1:] == ("+FLAGS", "\\Flagged")
    client.mark(7, "seen", value=False)
    assert fake_imap["client"].stored[-1][1:] == ("-FLAGS", "\\Seen")

    client.move(7, "Archive")  # 服务器不支持 MOVE → copy + delete
    assert fake_imap["client"].copied
    assert fake_imap["client"].stored[-1][1] == "+FLAGS"
    assert fake_imap["client"].expunged == 1

    fake_imap["client"].support_move = True
    client.move(7, "Archive")
    assert fake_imap["client"].moved

    client.delete(7)
    assert fake_imap["client"].stored[-1][1] == "+FLAGS"
    assert fake_imap["client"].expunged == 2


def test_imap_search_error_is_wrapped(fake_imap):
    fake_imap["client"].search_error = True
    with pytest.raises(MailReceiveError, match="检索失败"):
        _client().messages()


def test_imap_count_and_unseen_query(fake_imap):
    assert _client().count(unseen=True) == 1


def test_search_criteria_translation():
    from cs.email import search_criteria

    assert search_criteria("") == "ALL"
    assert search_criteria("  ") == "ALL"
    assert search_criteria("账单") == 'OR SUBJECT "账单" FROM "账单"'
    assert search_criteria('SUBJECT "账单"') == 'SUBJECT "账单"'
    assert search_criteria("SINCE 01-Jan-2026") == "SINCE 01-Jan-2026"


# ============================================================
# 配置
# ============================================================


def test_load_account_from_toml(tmp_path):
    (tmp_path / "secret.txt").write_text("p@ss\n", encoding="utf-8")
    path = tmp_path / "mail.toml"
    path.write_text(
        """
default = "work"

[profiles.work]
username = "me@example.com"
password = "file:secret.txt"
smtp_host = "smtp.example.com"
imap_host = "imap.example.com"
mailbox = "Archive"

[profiles.other]
smtp_host = "smtp2.example.com"
""",
        encoding="utf-8",
    )

    account = load_account(config=path)
    assert account.name == "work"
    assert account.capabilities == ("smtp", "imap")
    assert account.require_smtp().password == "p@ss"
    assert account.require_imap().mailbox == "Archive"
    assert "p@ss" not in repr(account)

    assert load_account("other", config=path).imap is None
    with pytest.raises(MailConfigError, match="没有账号"):
        load_account("missing", config=path)


def test_load_account_from_flat_tables(tmp_path):
    path = tmp_path / "mail.toml"
    path.write_text(
        """
[solo]
smtp_host = "smtp.example.com"
username = "me@example.com"
""",
        encoding="utf-8",
    )
    assert load_account(config=path).name == "solo"


def test_load_account_uses_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CS_MAIL_SMTP_HOST", "smtp.env.com")
    monkeypatch.setenv("CS_MAIL_IMAP_HOST", "imap.env.com")
    monkeypatch.setenv("CS_MAIL_USERNAME", "env@example.com")
    monkeypatch.setenv("CS_MAIL_PASSWORD", "envpass")
    account = account_from_env()
    assert account is not None
    assert account.name == "env"
    assert account.require_imap().host == "imap.env.com"
    assert account.require_smtp().password == "envpass"


def test_load_account_without_config(monkeypatch):
    for name in ("CS_MAIL_SMTP_HOST", "CS_MAIL_IMAP_HOST", "CS_MAIL_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("cs.email.config.DEFAULT_CONFIG_PATHS", ())
    with pytest.raises(MailConfigError, match="没有找到邮箱配置"):
        load_account()


def test_resolve_secret_literal_env_and_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_VAR", "from-env")
    assert resolve_secret("plain") == "plain"
    assert resolve_secret("env:SECRET_VAR") == "from-env"
    with pytest.raises(MailConfigError, match="未设置"):
        resolve_secret("env:MISSING_VAR")

    secret = tmp_path / "s.txt"
    secret.write_text("from-file\n", encoding="utf-8")
    assert resolve_secret("file:s.txt", base_dir=tmp_path) == "from-file"
    with pytest.raises(MailConfigError, match="不存在"):
        resolve_secret("file:none.txt", base_dir=tmp_path)


def test_load_account_rejects_missing_protocol(tmp_path):
    path = tmp_path / "mail.toml"
    path.write_text("[profiles.empty]\nusername = 'a@b.c'\n", encoding="utf-8")
    with pytest.raises(MailConfigError, match="至少需要"):
        load_account("empty", config=path)


# ============================================================
# MCP
# ============================================================


def test_mcp_server_exposes_tools():
    pytest.importorskip("fastmcp")
    from fastmcp import Client

    from cs.email.mcp_server import create_server

    async def names() -> list[str]:
        async with Client(create_server()) as client:
            return [tool.name for tool in await client.list_tools()]

    assert asyncio.run(names()) == [
        "mail_accounts",
        "mail_account_show",
        "mail_verify",
        "mail_send",
        "mail_folders",
        "mail_list",
        "mail_read",
        "mail_attachments",
        "mail_download",
        "mail_mark",
        "mail_move",
        "mail_delete",
    ]


def test_mcp_send_tool(monkeypatch, tmp_path):
    pytest.importorskip("fastmcp")
    from fastmcp import Client

    from cs.email.mcp_server import create_server

    smtp = FakeSMTP()
    monkeypatch.setattr("cs.email.smtp.smtplib.SMTP_SSL", lambda *a, **kw: smtp)

    config = tmp_path / "mail.toml"
    config.write_text(
        """
[profiles.work]
username = "me@example.com"
password = "secret"
smtp_host = "smtp.example.com"
""",
        encoding="utf-8",
    )

    async def call() -> dict:
        async with Client(create_server()) as client:
            result = await client.call_tool(
                "mail_send",
                {
                    "to": "a@example.com",
                    "subject": "hi",
                    "body": "hello",
                    "config": str(config),
                },
            )
            return json.loads(result.content[0].text)

    payload = asyncio.run(call())
    assert payload["ok"] is True
    assert payload["recipients"] == ["a@example.com"]
    assert smtp.message["Subject"] == "hi"


def test_mcp_account_show_hides_password(tmp_path):
    pytest.importorskip("fastmcp")
    from fastmcp import Client

    from cs.email.mcp_server import create_server

    config = tmp_path / "mail.toml"
    config.write_text(
        """
[profiles.work]
username = "me@example.com"
password = "topsecret"
smtp_host = "smtp.example.com"
""",
        encoding="utf-8",
    )

    async def call() -> dict:
        async with Client(create_server()) as client:
            result = await client.call_tool(
                "mail_account_show", {"config": str(config)}
            )
            return json.loads(result.content[0].text)

    payload = asyncio.run(call())
    assert payload["smtp"]["password"] == "***"
    assert "topsecret" not in str(payload)
