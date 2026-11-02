"""格式校验：常用类型判断、规则注册表与链式校验器。

三种用法：

- 单个判断：``is_email(x)`` / ``Validate.is_email(x)``
- 批量校验：``validate(x, "email", "length", min_length=6)`` → :class:`CheckResult`
- 链式校验：``Validate(x).require("non_empty").require("email").run()``

规则只做「形态判断」，不连接网络、不查号段库；需要业务约束时组合使用即可。
"""

from __future__ import annotations

import base64
import binascii
import inspect
import ipaddress
import json
import re
import uuid as uuid_lib
from collections.abc import Callable, Sequence, Sized
from datetime import date as date_cls
from datetime import datetime
from datetime import time as time_cls
from typing import Any, Self

from .errors import ValidationError
from .models import Check, CheckResult

__all__ = [
    "CHECKS",
    "Validate",
    "has_chinese",
    "is_alnum",
    "is_ascii",
    "is_base64",
    "is_boolean",
    "is_chinese",
    "is_cidr",
    "is_credit_card",
    "is_date",
    "is_datetime",
    "is_domain",
    "is_email",
    "is_float",
    "is_hex",
    "is_id_card",
    "is_in",
    "is_integer",
    "is_ip",
    "is_ipv4",
    "is_ipv6",
    "is_json",
    "is_mac",
    "is_not_none",
    "is_number",
    "is_password",
    "is_phone",
    "is_regexp",
    "is_scientific",
    "is_semver",
    "is_slug",
    "is_string",
    "is_time",
    "is_url",
    "is_uuid",
    "length",
    "non_empty",
    "parse_checks",
    "validate",
]

_EMAIL = re.compile(
    r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)
_DOMAIN = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}$"
)
_MAC = re.compile(
    r"^(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}$|^(?:[0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4}$"
)
_HEX = re.compile(r"^[0-9A-Fa-f]+$")
_BASE64 = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
_BASE64URL = re.compile(r"^[A-Za-z0-9_-]+={0,2}$")
_INTEGER = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_SCIENTIFIC = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PHONE_CN = re.compile(r"^1[3-9]\d{9}$")
_CHINESE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+$")
_HAS_CHINESE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SEMVER = re.compile(
    r"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECKSUM = "10X98765432"

type Predicate = Callable[..., bool]


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


# ---------------------------------------------------------------- 基础


def non_empty(value: Any) -> bool:
    """非 ``None``、非空串（空白串视为空）、非空容器；``0`` 与 ``False`` 算有值。"""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Sized):
        return len(value) > 0
    return True


def length(value: Any, *, min_length: int = 0, max_length: int | None = None) -> bool:
    """按 ``len()`` 检查长度；``None`` 仅在 ``min_length <= 0`` 时通过。"""
    if min_length < 0:
        raise ValidationError("min_length 不能为负")
    if max_length is not None and max_length < min_length:
        raise ValidationError("max_length 不能小于 min_length")
    if value is None:
        return min_length <= 0
    size = len(value)
    return size >= min_length and (max_length is None or size <= max_length)


def is_string(value: Any) -> bool:
    return isinstance(value, str)


def is_integer(value: Any) -> bool:
    """整数或纯整数字符串；``bool`` 不算整数。"""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    text = _as_str(value)
    return bool(text and _INTEGER.fullmatch(text))


def is_float(value: Any) -> bool:
    """浮点或可转浮点的字符串；整数也算（``3`` → ``3.0``）。"""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    text = _as_str(value)
    return bool(text and _FLOAT.fullmatch(text))


def is_number(value: Any) -> bool:
    """整数、小数或科学计数法。"""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    text = _as_str(value)
    return bool(
        text
        and (
            _INTEGER.fullmatch(text)
            or _FLOAT.fullmatch(text)
            or _SCIENTIFIC.fullmatch(text)
        )
    )


def is_scientific(value: Any) -> bool:
    """科学计数法（必须带指数部分）。"""
    text = _as_str(value)
    return bool(text and _SCIENTIFIC.fullmatch(text))


def is_boolean(value: Any) -> bool:
    """``bool`` 或常见布尔字面量（true/false/1/0/yes/no/on/off）。"""
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return value in {0, 1}
    return _as_str(value).lower() in {
        "true",
        "false",
        "1",
        "0",
        "yes",
        "no",
        "on",
        "off",
    }


# ---------------------------------------------------------------- 网络


def is_email(value: Any) -> bool:
    text = _as_str(value)
    return bool(text and len(text) <= 254 and _EMAIL.fullmatch(text))


def is_domain(value: Any) -> bool:
    text = _as_str(value)
    return bool(text and _DOMAIN.fullmatch(text))


def is_url(value: Any, *, schemes: Sequence[str] = ("http", "https", "ftp")) -> bool:
    """要求带 scheme 与主机；主机可以是域名、IP 或 ``localhost``。"""
    from urllib.parse import urlsplit

    text = _as_str(value)
    if not text or any(char.isspace() for char in text):
        return False
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        return False
    if parts.scheme.lower() not in schemes or not parts.netloc:
        return False
    if port is not None and not 0 < port < 65536:
        return False
    host = parts.hostname
    if not host:
        return False
    return host == "localhost" or is_ip(host) or is_domain(host)


def is_ipv4(value: Any) -> bool:
    try:
        ipaddress.IPv4Address(_as_str(value))
    except (ipaddress.AddressValueError, ValueError):
        return False
    return True


def is_ipv6(value: Any) -> bool:
    try:
        ipaddress.IPv6Address(_as_str(value))
    except (ipaddress.AddressValueError, ValueError):
        return False
    return True


def is_ip(value: Any) -> bool:
    return is_ipv4(value) or is_ipv6(value)


def is_cidr(value: Any) -> bool:
    """``192.168.0.0/24`` / ``2001:db8::/32``；``strict=False`` 允许非网络地址。"""
    text = _as_str(value)
    if "/" not in text:
        return False
    try:
        ipaddress.ip_network(text, strict=False)
    except ValueError:
        return False
    return True


def is_mac(value: Any) -> bool:
    return bool(_MAC.fullmatch(_as_str(value)))


def is_uuid(value: Any, *, version: int | None = None) -> bool:
    """16 进制 UUID（可带连字符/花括号）；``version`` 可限定 1/4/7 等。"""
    try:
        parsed = uuid_lib.UUID(_as_str(value))
    except (ValueError, AttributeError, TypeError):
        return False
    return version is None or parsed.version == version


# ---------------------------------------------------------------- 编码


def is_hex(value: Any, *, prefix: bool = True, even: bool = False) -> bool:
    """十六进制串；``prefix`` 控制是否允许 ``0x``，``even`` 要求偶数位。"""
    text = _as_str(value)
    if not text:
        return False
    body = text[2:] if text[:2].lower() == "0x" else text
    if text[:2].lower() == "0x" and not prefix:
        return False
    if not _HEX.fullmatch(body):
        return False
    return len(body) % 2 == 0 if even else True


def is_base64(value: Any, *, urlsafe: bool = False) -> bool:
    """标准或 URL-safe base64（要求长度是 4 的倍数，可含尾部 ``=``）。"""
    text = _as_str(value).replace("\n", "").replace("\r", "")
    if not text or len(text) % 4:
        return False
    pattern = _BASE64URL if urlsafe else _BASE64
    if not pattern.fullmatch(text):
        return False
    try:
        base64.b64decode(text.replace("-", "+").replace("_", "/"))
    except (binascii.Error, ValueError):
        return False
    return True


def is_json(value: Any) -> bool:
    """合法 JSON 文本（对象、数组、标量都算）。"""
    text = _as_str(value) if not isinstance(value, str) else value
    if not text:
        return False
    try:
        json.loads(text)
    except (ValueError, TypeError):
        return False
    return True


# ---------------------------------------------------------------- 时间


def is_date(value: Any) -> bool:
    """``YYYY-MM-DD`` 且日期真实存在。"""
    text = _as_str(value)
    if not _DATE.fullmatch(text):
        return False
    try:
        date_cls.fromisoformat(text)
    except ValueError:
        return False
    return True


def is_time(value: Any) -> bool:
    """ISO 时间：``12:30`` / ``12:30:45`` / ``12:30:45.123+08:00``。"""
    try:
        time_cls.fromisoformat(_as_str(value))
    except ValueError:
        return False
    return True


def is_datetime(value: Any) -> bool:
    """ISO 日期时间，``Z`` 结尾也接受。"""
    text = _as_str(value).replace("Z", "+00:00")
    if "T" not in text and " " not in text:
        return False
    try:
        datetime.fromisoformat(text)
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------- 业务


def is_phone(value: Any) -> bool:
    """中国大陆手机号（1 开头 11 位）。"""
    return bool(_PHONE_CN.fullmatch(_as_str(value)))


def is_id_card(value: Any) -> bool:
    """中国大陆 18 位身份证号，含模 11 校验位验证。"""
    text = _as_str(value).upper()
    if not re.fullmatch(r"\d{17}[\dX]", text):
        return False
    total = sum(
        int(char) * weight for char, weight in zip(text[:17], _ID_WEIGHTS, strict=True)
    )
    return text[-1] == _ID_CHECKSUM[total % 11]


def is_credit_card(value: Any) -> bool:
    """12~19 位数字并通过 Luhn 校验（允许空格/连字符分组）。"""
    digits = re.sub(r"[\s-]", "", _as_str(value))
    if not digits.isdigit() or not 12 <= len(digits) <= 19:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        number = int(char)
        if index % 2 == 1:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return total % 10 == 0


def is_semver(value: Any) -> bool:
    """语义化版本 ``MAJOR.MINOR.PATCH[-pre][+build]``。"""
    return bool(_SEMVER.fullmatch(_as_str(value)))


def is_slug(value: Any) -> bool:
    """小写、连字符分隔的 URL slug。"""
    return bool(_SLUG.fullmatch(_as_str(value)))


def is_password(
    value: Any,
    *,
    min_length: int = 8,
    max_length: int = 64,
    require: Sequence[str] = ("digit", "letter"),
) -> bool:
    """口令强度：长度范围内且满足 ``require`` 中的字符类别。

    可选类别：``digit`` / ``lower`` / ``upper`` / ``letter`` / ``symbol`` / ``space``。
    """
    text = _as_str(value)
    if not text or not min_length <= len(text) <= max_length:
        return False
    categories = {
        "digit": any(char.isdigit() for char in text),
        "lower": any(char.islower() for char in text),
        "upper": any(char.isupper() for char in text),
        "letter": any(char.isalpha() for char in text),
        "symbol": any(not char.isalnum() for char in text),
        "space": any(char.isspace() for char in text),
    }
    unknown = set(require) - set(categories)
    if unknown:
        raise ValidationError(
            f"未知口令类别 {sorted(unknown)}，可选：{sorted(categories)}"
        )
    return all(categories[name] for name in require)


# ---------------------------------------------------------------- 通用


def is_ascii(value: Any) -> bool:
    text = _as_str(value)
    return bool(text) and text.isascii()


def is_alnum(value: Any) -> bool:
    text = _as_str(value)
    return bool(text) and text.isalnum()


def has_chinese(value: Any) -> bool:
    return bool(_HAS_CHINESE.search(_as_str(value)))


def is_chinese(value: Any) -> bool:
    return bool(_CHINESE.fullmatch(_as_str(value)))


def is_regexp(value: Any, pattern: str) -> bool:
    """整串匹配正则；``pattern`` 非法时抛 :class:`ValidationError`。"""
    text = _as_str(value)
    if not text:
        return False
    try:
        return re.fullmatch(pattern, text) is not None
    except re.error as exc:
        raise ValidationError(f"正则表达式非法：{exc}") from exc


def is_in(value: Any, choices: Sequence[Any]) -> bool:
    """取值是否在候选集合内。"""
    return value in choices


def is_not_none(value: Any) -> bool:
    """只要不是 ``None`` 即通过（空串、0、False 都算有值）。"""
    return value is not None


# ---------------------------------------------------------------- 注册表


#: 可批量调用的规则名 → 判断函数
CHECKS: dict[str, Predicate] = {
    "non_empty": non_empty,
    "length": length,
    "string": is_string,
    "integer": is_integer,
    "float": is_float,
    "number": is_number,
    "scientific": is_scientific,
    "boolean": is_boolean,
    "email": is_email,
    "domain": is_domain,
    "url": is_url,
    "ipv4": is_ipv4,
    "ipv6": is_ipv6,
    "ip": is_ip,
    "cidr": is_cidr,
    "mac": is_mac,
    "uuid": is_uuid,
    "hex": is_hex,
    "base64": is_base64,
    "json": is_json,
    "date": is_date,
    "time": is_time,
    "datetime": is_datetime,
    "phone": is_phone,
    "id_card": is_id_card,
    "credit_card": is_credit_card,
    "semver": is_semver,
    "slug": is_slug,
    "password": is_password,
    "ascii": is_ascii,
    "alnum": is_alnum,
    "chinese": is_chinese,
    "has_chinese": has_chinese,
    "regexp": is_regexp,
    "in": is_in,
}


def parse_checks(spec: str | Sequence[str]) -> tuple[str, ...]:
    """把 ``"email,length"`` 或列表解析成规则名元组。"""
    if isinstance(spec, str):
        items = re.split(r"[,;\s]+", spec)
    else:
        items = list(spec)
    return tuple(item.strip() for item in items if item and item.strip())


def validate(value: Any, *names: str, **options: Any) -> CheckResult:
    """按规则名逐个校验，返回全部结果（不短路）。

    ``options`` 会按签名自动过滤，所以既能一次传 ``min_length``，
    也能同时给 ``regexp`` 用的 ``pattern``。
    """
    selected = names or ("non_empty",)
    checks: list[Check] = []
    for name in selected:
        func = CHECKS.get(name)
        if func is None:
            raise ValidationError(
                f"未知校验规则 {name!r}，可用：{', '.join(sorted(CHECKS))}"
            )
        applied = _applicable(func, options)
        checks.append(
            Check(
                name=name,
                passed=bool(func(value, **applied)),
                detail=", ".join(f"{key}={val}" for key, val in applied.items()),
            )
        )
    return CheckResult(value=_as_str(value), checks=tuple(checks))


def _applicable(func: Predicate, options: dict[str, Any]) -> dict[str, Any]:
    """按函数签名挑出可用参数；必填参数缺失时报错而不是静默通过。"""
    parameters = inspect.signature(func).parameters
    applied: dict[str, Any] = {}
    for key, value in options.items():
        if key in parameters and key != "value":
            applied[key] = value
    missing = [
        key
        for key, parameter in parameters.items()
        if key != "value"
        and parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and key not in applied
    ]
    if missing:
        raise ValidationError(
            f"{func.__name__} 缺少参数：{', '.join(missing)}（校验此类规则时需一并提供）"
        )
    return applied


class Validate:
    """链式校验器，同时以静态方法暴露全部规则。

    >>> Validate("a@b.com").require("email").run()
    True
    """

    # ---- 静态规则（与模块级函数同源）
    non_empty = staticmethod(non_empty)
    length = staticmethod(length)
    is_string = staticmethod(is_string)
    is_str = staticmethod(is_string)
    is_integer = staticmethod(is_integer)
    is_float = staticmethod(is_float)
    is_number = staticmethod(is_number)
    is_scientific = staticmethod(is_scientific)
    is_boolean = staticmethod(is_boolean)
    is_email = staticmethod(is_email)
    is_domain = staticmethod(is_domain)
    is_url = staticmethod(is_url)
    is_ipv4 = staticmethod(is_ipv4)
    is_ipv6 = staticmethod(is_ipv6)
    is_ip = staticmethod(is_ip)
    is_cidr = staticmethod(is_cidr)
    is_mac = staticmethod(is_mac)
    is_uuid = staticmethod(is_uuid)
    is_hex = staticmethod(is_hex)
    is_base64 = staticmethod(is_base64)
    is_json = staticmethod(is_json)
    is_date = staticmethod(is_date)
    is_time = staticmethod(is_time)
    is_datetime = staticmethod(is_datetime)
    is_phone = staticmethod(is_phone)
    is_phone_number = staticmethod(is_phone)
    is_id_card = staticmethod(is_id_card)
    is_credit_card = staticmethod(is_credit_card)
    is_semver = staticmethod(is_semver)
    is_slug = staticmethod(is_slug)
    is_password = staticmethod(is_password)
    is_ascii = staticmethod(is_ascii)
    is_alnum = staticmethod(is_alnum)
    has_chinese = staticmethod(has_chinese)
    is_chinese = staticmethod(is_chinese)
    is_regexp = staticmethod(is_regexp)
    is_in = staticmethod(is_in)
    is_enum = staticmethod(is_in)
    required = staticmethod(non_empty)
    is_any = staticmethod(is_not_none)

    def __init__(self, value: Any = None, *, stop_on_failure: bool = False) -> None:
        self.value = value
        self.stop_on_failure = stop_on_failure
        self.rules: list[tuple[str, Predicate, dict[str, Any]]] = []
        self.result: CheckResult = CheckResult(value=_as_str(value))

    def add(self, rule: Predicate, *, name: str = "", **options: Any) -> Self:
        """加入自定义判断函数。"""
        if not callable(rule):
            raise ValidationError("rule 必须是可调用对象")
        self.rules.append((name or getattr(rule, "__name__", "rule"), rule, options))
        return self

    def require(self, name: str, **options: Any) -> Self:
        """加入注册表中的规则。"""
        func = CHECKS.get(name)
        if func is None:
            raise ValidationError(
                f"未知校验规则 {name!r}，可用：{', '.join(sorted(CHECKS))}"
            )
        self.rules.append((name, func, _applicable(func, options)))
        return self

    def run(self, value: Any = None) -> bool:
        """执行全部规则；``stop_on_failure=True`` 时首个失败即停止。"""
        target = self.value if value is None else value
        checks: list[Check] = []
        for name, func, options in self.rules:
            passed = bool(func(target, **options))
            checks.append(
                Check(
                    name=name,
                    passed=passed,
                    detail=", ".join(f"{key}={val}" for key, val in options.items()),
                )
            )
            if not passed and self.stop_on_failure:
                break
        self.result = CheckResult(value=_as_str(target), checks=tuple(checks))
        return self.result.ok

    @property
    def valid(self) -> bool:
        """上一次 :meth:`run` 的结果（未执行时视为通过）。"""
        return self.result.ok

    @property
    def errors(self) -> tuple[str, ...]:
        return self.result.failed

    def __bool__(self) -> bool:
        return self.run()

    def __str__(self) -> str:
        return str(self.result)
