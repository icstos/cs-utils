"""账号配置：从 TOML 文件或环境变量组装 :class:`~cs.email.models.Account`。

TOML 结构（``python -m cs.email init`` 可生成骨架）::

    default = "work"

    [profiles.work]
    username = "me@example.com"
    password = "env:CS_MAIL_PASSWORD"   # 也支持 file:/path/secret.txt
    smtp_host = "smtp.example.com"
    imap_host = "imap.example.com"

密码不进配置文件时，可用环境变量整体配置一个账号::

    CS_MAIL_SMTP_HOST / CS_MAIL_SMTP_PORT / CS_MAIL_SMTP_SECURITY
    CS_MAIL_IMAP_HOST / CS_MAIL_IMAP_PORT / CS_MAIL_IMAP_SECURITY
    CS_MAIL_USERNAME / CS_MAIL_PASSWORD / CS_MAIL_SENDER / CS_MAIL_MAILBOX
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import MailConfigError
from .models import Account, ImapConfig, Security, SmtpConfig

__all__ = [
    "DEFAULT_CONFIG_PATHS",
    "ENV_CONFIG",
    "ENV_PREFIX",
    "SAMPLE_CONFIG",
    "account_from_env",
    "load_account",
    "load_profiles",
    "resolve_config_path",
    "resolve_secret",
]

#: 指定配置文件路径的环境变量
ENV_CONFIG = "CS_MAIL_CONFIG"
#: 账号字段的环境变量前缀
ENV_PREFIX = "CS_MAIL_"

#: 未显式给 ``--config`` 时依次查找
DEFAULT_CONFIG_PATHS = (
    Path("mail.toml"),
    Path("config/mail.toml"),
    Path("~/.config/cs/mail.toml").expanduser(),
    Path("~/.cs/mail.toml").expanduser(),
)

SAMPLE_CONFIG = """\
# cs-email 账号配置；用 --config 指定，或放在 ~/.config/cs/mail.toml
default = "work"

[profiles.work]
username = "me@example.com"
# 密码建议不要写明文：env:变量名 读环境变量，file:相对路径 读文件
password = "env:CS_MAIL_PASSWORD"
sender = "me@example.com"        # 可省略，缺省用 username

smtp_host = "smtp.example.com"
smtp_port = 465                  # 0 = 按 smtp_security 自动选择
smtp_security = "ssl"            # ssl / starttls / none

imap_host = "imap.example.com"
imap_port = 993                  # 0 = 按 imap_security 自动选择
imap_security = "ssl"
mailbox = "INBOX"
timeout = 30
"""


@dataclass(slots=True, frozen=True)
class ConfigFile:
    """解析后的配置文件快照。"""

    path: Path | None = None
    default: str = ""
    profiles: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def resolve_config_path(path: str | Path | None = None) -> Path | None:
    """定位配置文件：显式路径 → ``CS_MAIL_CONFIG`` → 默认候选；找不到返回 ``None``。"""
    if path is not None:
        candidate = Path(path).expanduser()
        if not candidate.is_file():
            raise MailConfigError(f"配置文件不存在：{candidate}")
        return candidate
    from_env = os.environ.get(ENV_CONFIG, "").strip()
    if from_env:
        candidate = Path(from_env).expanduser()
        if not candidate.is_file():
            raise MailConfigError(f"{ENV_CONFIG} 指向的配置文件不存在：{candidate}")
        return candidate
    for candidate in DEFAULT_CONFIG_PATHS:
        if candidate.is_file():
            return candidate
    return None


def resolve_secret(value: str | None, *, base_dir: Path | None = None) -> str:
    """解析密码：``env:VAR`` 取环境变量，``file:path`` 取文件内容，其他按字面。"""
    text = (value or "").strip()
    if text.startswith("env:"):
        variable = text[4:].strip()
        resolved = os.environ.get(variable, "")
        if not resolved:
            raise MailConfigError(f"环境变量 {variable} 未设置或为空")
        return resolved
    if text.startswith("file:"):
        file = Path(text[5:].strip()).expanduser()
        if not file.is_absolute() and base_dir is not None:
            file = base_dir / file
        if not file.is_file():
            raise MailConfigError(f"密码文件不存在：{file}")
        return file.read_text(encoding="utf-8").strip()
    return text


def load_profiles(path: str | Path | None = None) -> ConfigFile:
    """读取配置文件；文件不存在时返回空快照（后续会退回环境变量）。"""
    resolved = resolve_config_path(path)
    if resolved is None:
        return ConfigFile()
    try:
        data = tomllib.loads(resolved.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise MailConfigError(f"配置文件 {resolved} 解析失败：{exc}") from exc

    raw = data.get("profiles")
    if not isinstance(raw, Mapping):
        # 兼容 ``[work]`` 这种省掉 profiles 层的写法
        raw = {key: value for key, value in data.items() if isinstance(value, Mapping)}
    profiles = {
        str(name): dict(section)
        for name, section in raw.items()
        if isinstance(section, Mapping)
    }
    return ConfigFile(
        path=resolved, default=str(data.get("default", "")), profiles=profiles
    )


def account_from_env() -> Account | None:
    """用 ``CS_MAIL_*`` 环境变量拼一个账号；关键字段缺失时返回 ``None``。"""
    values = {
        field_name: value
        for field_name, env_name in _ENV_FIELDS.items()
        if (value := os.environ.get(f"{ENV_PREFIX}{env_name}", "").strip())
    }
    if not values.get("smtp_host") and not values.get("imap_host"):
        return None
    return _account_from_mapping("env", values)


def load_account(
    name: str | None = None, *, config: str | Path | None = None
) -> Account:
    """按名字取账号：配置文件优先，其次环境变量。

    ``name=None`` 时依次尝试配置里的 ``default``、唯一 profile、环境变量。
    """
    file = load_profiles(config)
    chosen = name or file.default

    if chosen and chosen in file.profiles:
        return _account_from_mapping(
            chosen, file.profiles[chosen], base_dir=_base_dir(file)
        )
    if chosen and file.profiles:
        available = ", ".join(sorted(file.profiles))
        raise MailConfigError(f"配置里没有账号 {chosen!r}，可选：{available}")
    if not chosen and len(file.profiles) == 1:
        only, section = next(iter(file.profiles.items()))
        return _account_from_mapping(only, section, base_dir=_base_dir(file))
    if name is None:
        from_env = account_from_env()
        if from_env is not None:
            return from_env
    if not file.profiles:
        raise MailConfigError(
            "没有找到邮箱配置：请用 --config 指定 TOML、设置 "
            f"{ENV_CONFIG}，或通过 {ENV_PREFIX}SMTP_HOST / {ENV_PREFIX}IMAP_HOST "
            "等环境变量提供"
        )
    available = ", ".join(sorted(file.profiles))
    raise MailConfigError(f"请指定账号名，可选：{available}")


# ------------------------------------------------------------ 内部工具

_ENV_FIELDS = {
    "username": "USERNAME",
    "password": "PASSWORD",
    "sender": "SENDER",
    "smtp_host": "SMTP_HOST",
    "smtp_port": "SMTP_PORT",
    "smtp_security": "SMTP_SECURITY",
    "imap_host": "IMAP_HOST",
    "imap_port": "IMAP_PORT",
    "imap_security": "IMAP_SECURITY",
    "mailbox": "MAILBOX",
    "timeout": "TIMEOUT",
}


def _base_dir(file: ConfigFile) -> Path | None:
    return file.path.parent if file.path is not None else None


def _account_from_mapping(
    name: str,
    data: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> Account:
    username = _text(data.get("username"))
    password = resolve_secret(_text(data.get("password")), base_dir=base_dir)
    sender = _text(data.get("sender"))
    timeout = _number(data.get("timeout"), 30.0)

    smtp = None
    if smtp_host := _text(data.get("smtp_host")):
        smtp = SmtpConfig(
            host=smtp_host,
            port=_integer(data.get("smtp_port"), 0),
            username=_text(data.get("smtp_username")) or username,
            password=resolve_secret(_text(data.get("smtp_password")), base_dir=base_dir)
            or password,
            sender=_text(data.get("smtp_sender")) or sender,
            security=Security.parse(_text(data.get("smtp_security"))),
            timeout=timeout,
        )

    imap = None
    if imap_host := _text(data.get("imap_host")):
        imap = ImapConfig(
            host=imap_host,
            port=_integer(data.get("imap_port"), 0),
            username=_text(data.get("imap_username")) or username,
            password=resolve_secret(_text(data.get("imap_password")), base_dir=base_dir)
            or password,
            security=Security.parse(_text(data.get("imap_security"))),
            timeout=timeout,
            mailbox=_text(data.get("mailbox")) or "INBOX",
        )

    if smtp is None and imap is None:
        raise MailConfigError(f"账号 {name!r} 至少需要 smtp_host 或 imap_host")
    return Account(name=name, smtp=smtp, imap=imap)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _integer(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise MailConfigError(f"端口必须是整数，收到 {value!r}") from None


def _number(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        raise MailConfigError(f"timeout 必须是数字，收到 {value!r}") from None
