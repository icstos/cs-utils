"""``Text`` 门面：把比较、变换、校验、编码、摘要、加解密收在一处。

各子模块可以单独导入（``from cs.text import compare``），门面只做转发，
不引入额外状态。

>>> from cs.text import Text
>>> Text.snake("HTTPResponse")
'http_response'
>>> Text.is_email("cs@cstos.com")
True
"""

from __future__ import annotations

from typing import Any

from . import codec, compare, crypto, digest, transform
from . import validators as rules
from .models import Encoding, HashAlgorithm

__all__ = ["Text"]


class Text:
    """文本工具门面（全部静态方法，无实例状态）。"""

    #: 可批量调用的校验规则表
    CHECKS = rules.CHECKS

    #: 枚举直接挂在门面上，便于 ``Text.Encoding.HEX``
    Encoding = Encoding
    HashAlgorithm = HashAlgorithm

    # ------------------------------------------------------------ 比较
    diff = staticmethod(compare.diff)
    ratio = staticmethod(compare.ratio)
    quick_ratio = staticmethod(compare.quick_ratio)
    similar = staticmethod(compare.similar)
    unified = staticmethod(compare.unified)
    distance = staticmethod(compare.distance)
    best_matches = staticmethod(compare.best_matches)
    tokenize = staticmethod(compare.tokenize)

    # ------------------------------------------------------------ 变换
    to_str = staticmethod(transform.to_str)
    normalize = staticmethod(transform.normalize)
    collapse_whitespace = staticmethod(transform.collapse_whitespace)
    one_line = staticmethod(transform.one_line)
    strip_control = staticmethod(transform.strip_control)
    strip_lines = staticmethod(transform.strip_lines)
    to_halfwidth = staticmethod(transform.to_halfwidth)
    width = staticmethod(transform.width)
    char_width = staticmethod(transform.char_width)
    truncate = staticmethod(transform.truncate)
    wrap = staticmethod(transform.wrap)
    pad = staticmethod(transform.pad)
    mask = staticmethod(transform.mask)
    indent = staticmethod(transform.indent)
    split_words = staticmethod(transform.split_words)
    snake = staticmethod(transform.snake)
    kebab = staticmethod(transform.kebab)
    constant = staticmethod(transform.constant)
    camel = staticmethod(transform.camel)
    pascal = staticmethod(transform.pascal)
    title_case = staticmethod(transform.title_case)
    slugify = staticmethod(transform.slugify)

    # ------------------------------------------------------------ 校验
    validate = staticmethod(rules.validate)
    check = staticmethod(rules.validate)
    parse_checks = staticmethod(rules.parse_checks)
    Validate = rules.Validate

    is_email = staticmethod(rules.is_email)
    is_url = staticmethod(rules.is_url)
    is_domain = staticmethod(rules.is_domain)
    is_ip = staticmethod(rules.is_ip)
    is_ipv4 = staticmethod(rules.is_ipv4)
    is_ipv6 = staticmethod(rules.is_ipv6)
    is_cidr = staticmethod(rules.is_cidr)
    is_uuid = staticmethod(rules.is_uuid)
    is_mac = staticmethod(rules.is_mac)
    is_hex = staticmethod(rules.is_hex)
    is_base64 = staticmethod(rules.is_base64)
    is_json = staticmethod(rules.is_json)
    is_number = staticmethod(rules.is_number)
    is_integer = staticmethod(rules.is_integer)
    is_float = staticmethod(rules.is_float)
    is_boolean = staticmethod(rules.is_boolean)
    is_date = staticmethod(rules.is_date)
    is_time = staticmethod(rules.is_time)
    is_datetime = staticmethod(rules.is_datetime)
    is_phone = staticmethod(rules.is_phone)
    is_id_card = staticmethod(rules.is_id_card)
    is_credit_card = staticmethod(rules.is_credit_card)
    is_semver = staticmethod(rules.is_semver)
    is_slug = staticmethod(rules.is_slug)
    is_password = staticmethod(rules.is_password)
    is_regexp = staticmethod(rules.is_regexp)
    is_chinese = staticmethod(rules.is_chinese)
    has_chinese = staticmethod(rules.has_chinese)

    # ------------------------------------------------------------ 编解码
    encode = staticmethod(codec.encode)
    decode = staticmethod(codec.decode)
    compress = staticmethod(codec.compress)
    decompress = staticmethod(codec.decompress)

    # ------------------------------------------------------------ 摘要
    hash = staticmethod(digest.hash_text)
    hash_bytes = staticmethod(digest.hash_bytes)
    hmac = staticmethod(digest.hmac_text)
    pbkdf2 = staticmethod(digest.derive_key)
    token = staticmethod(digest.token)
    file_hash = staticmethod(digest.file_hash)
    fingerprint = staticmethod(digest.fingerprint)
    secure_equals = staticmethod(digest.verify)

    # ------------------------------------------------------------ 加解密
    encrypt = staticmethod(crypto.encrypt)
    decrypt = staticmethod(crypto.decrypt)
    encrypt_file = staticmethod(crypto.encrypt_file)
    decrypt_file = staticmethod(crypto.decrypt_file)
    new_key = staticmethod(crypto.new_key)
    token_info = staticmethod(crypto.token_info)

    @staticmethod
    def summary(value: Any) -> dict[str, Any]:
        """快速画像：长度、显示宽度、是否含中文、摘要值。"""
        text = transform.to_str(value, default="") or ""
        return {
            "chars": len(text),
            "width": transform.width(text),
            "lines": len(text.splitlines()) or (1 if text else 0),
            "has_chinese": rules.has_chinese(text),
            "sha256": digest.hash_text(text),
        }
