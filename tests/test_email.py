from email.message import EmailMessage
import sys

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs.email import Email


class FakeSMTP:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.message = None
        self.recipients = None
        self.started_tls = False
        self.logged_in = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def starttls(self, **kwargs):
        self.started_tls = True

    def login(self, username, password):
        self.logged_in = (username, password)

    def send_message(self, message, *, from_addr, to_addrs):
        self.message = message
        self.recipients = (from_addr, to_addrs)


def test_send_email_with_html_recipients_and_attachment(tmp_path, monkeypatch):
    attachment = tmp_path / "note.txt"
    attachment.write_text("attachment content", encoding="utf-8")
    smtp = FakeSMTP()

    def create_smtp(*args, **kwargs):
        smtp.args = args
        smtp.kwargs = kwargs
        return smtp

    monkeypatch.setattr("cs.email.smtplib.SMTP_SSL", create_smtp)

    email = Email("smtp.example.com", "sender@example.com", "secret")
    email.send_email(
        ["to@example.com", "second@example.com"],
        "Subject",
        "Plain text",
        cc="cc@example.com",
        bcc=["hidden@example.com"],
        html_body="<p>Hello</p>",
        attachments=[attachment],
    )

    assert smtp.args == ("smtp.example.com", 465)
    assert smtp.logged_in == ("sender@example.com", "secret")
    assert smtp.recipients == (
        "sender@example.com",
        [
            "to@example.com",
            "second@example.com",
            "cc@example.com",
            "hidden@example.com",
        ],
    )
    assert isinstance(smtp.message, EmailMessage)
    assert smtp.message["To"] == "to@example.com, second@example.com"
    assert smtp.message["Cc"] == "cc@example.com"
    assert "Bcc" not in smtp.message
    assert smtp.message.get_body(preferencelist=("plain",)).get_content() == (
        "Plain text\n"
    )
    assert smtp.message.get_body(preferencelist=("html",)).get_content() == (
        "<p>Hello</p>\n"
    )
    attachment_part = list(smtp.message.iter_attachments())[0]
    assert attachment_part.get_filename() == "note.txt"
    assert attachment_part.get_content() == "attachment content"


def test_send_email_can_use_starttls(monkeypatch):
    smtp = FakeSMTP()
    monkeypatch.setattr("cs.email.smtplib.SMTP", lambda *args, **kwargs: smtp)

    email = Email(
        "smtp.example.com",
        "sender@example.com",
        "secret",
        smtp_port=587,
        use_ssl=False,
        use_starttls=True,
        username="smtp-user",
    )
    email.send_email("to@example.com", "Subject", "Body")

    assert smtp.started_tls
    assert smtp.logged_in == ("smtp-user", "secret")
    assert smtp.recipients[1] == ["to@example.com"]
    assert "secret" not in repr(email)


def test_email_rejects_conflicting_tls_modes():
    with pytest.raises(ValueError, match="cannot both be enabled"):
        Email(
            "smtp.example.com",
            "sender@example.com",
            "secret",
            use_ssl=True,
            use_starttls=True,
        )


def test_send_email_requires_recipient():
    email = Email("smtp.example.com", "sender@example.com", "secret")

    with pytest.raises(ValueError, match="At least one"):
        email.send_email([], "Subject", "Body")
