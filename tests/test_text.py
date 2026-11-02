"""``cs.text`` 的单元测试。

覆盖：模型与 JSON 化、文本变换（宽度/截断/折行/打码/命名风格）、比较（相似度/
差异/编辑距离/候选）、编解码与压缩、摘要与 PBKDF2、口令与密钥加解密、
格式校验、MCP 工具清单与调用、CLI 退出码。
"""

from __future__ import annotations

import asyncio
import base64

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from cs.text import (
    CHECKS,
    CheckResult,
    CodecError,
    CryptoError,
    DecryptError,
    DerivedKey,
    DiffResult,
    DigestError,
    Encoding,
    Granularity,
    HashAlgorithm,
    Text,
    TextError,
    TransformError,
    Validate,
    ValidationError,
    best_matches,
    codec,
    compare,
    crypto,
    digest,
    transform,
    validators,
)
from cs.text.__main__ import main
from cs.text.mcp_server import create_server

# ---------------------------------------------------------------- 模型


def test_hash_algorithm_coerce_and_props():
    assert HashAlgorithm.coerce("SHA-256") is HashAlgorithm.SHA256
    assert HashAlgorithm.coerce("md5") is HashAlgorithm.MD5
    assert HashAlgorithm.SHA256.hex_length == 64
    assert HashAlgorithm.SHA256.weak is False
    assert HashAlgorithm.SHA1.weak is True
    with pytest.raises(ValueError, match="未知摘要算法"):
        HashAlgorithm.coerce("sha999")


def test_diff_result_reports_and_json():
    result = compare.diff("abc", "abd")
    assert isinstance(result, DiffResult)
    assert not result.equal
    assert result.distance == 2
    assert "相似度" in str(result)
    payload = result.to_dict()
    assert payload["chunks"][0]["tag"] == "replace"
    assert "a" not in payload
    assert result.to_dict(include_text=True)["a"] == "abc"
    assert compare.diff("same", "same").equal is True


def test_check_result_flags_and_bool():
    result = validators.validate("cs@cstos.com", "email", "length", min_length=5)
    assert result.ok and bool(result) and result.valid
    assert result.passed == ("email", "length")
    assert result.failed == ()
    failed = validators.validate("bad", "email")
    assert not failed and failed.errors == ("email",)
    assert "未通过" in str(failed)
    assert isinstance(CheckResult(value="x"), CheckResult)


# ---------------------------------------------------------------- 变换


def test_to_str_handles_none_bytes_and_scalars():
    assert transform.to_str("abc") == "abc"
    assert transform.to_str(b"abc") == "abc"
    assert transform.to_str(None) is None
    assert transform.to_str(None, default="") == ""
    assert transform.to_str(12) == "12"


def test_width_counts_east_asian_as_two_columns():
    assert transform.width("abc") == 3
    assert transform.width("中文") == 4
    assert transform.char_width("中") == 2
    assert transform.char_width("\u200b") == 0
    assert transform.char_width("\n") == 0


def test_truncate_reserves_suffix_width():
    assert transform.truncate("中文标题很长", 6) == "中文…"
    assert transform.truncate("abc", 10) == "abc"
    assert transform.truncate("abc", 0) == ""
    assert (
        transform.truncate("hello world", 8, suffix="~", word_boundary=True) == "hello~"
    )


def test_wrap_breaks_by_display_width():
    lines = transform.wrap("中文中文中文", 4)
    assert lines == ["中文", "中文", "中文"]
    assert transform.wrap("hello world", 5) == ["hello", "world"]
    with pytest.raises(TransformError):
        transform.wrap("x", 0)


def test_pad_alignments_and_validation():
    assert transform.pad("ab", 5) == "ab   "
    assert transform.pad("ab", 5, align="right") == "   ab"
    assert transform.pad("ab", 5, align="center") == " ab  "
    with pytest.raises(TransformError, match="align"):
        transform.pad("ab", 5, align="middle")


def test_mask_keeps_head_and_tail():
    assert transform.mask("13812345678", keep=3, tail=4) == "138****5678"
    assert transform.mask("abc", keep=4) == "***"
    assert transform.mask("") == ""


def test_case_conversions_from_mixed_input():
    assert transform.split_words("HTTPServer_error-v2") == [
        "HTTP",
        "Server",
        "error",
        "v2",
    ]
    assert transform.snake("HTTPServerError") == "http_server_error"
    assert transform.kebab("user_id") == "user-id"
    assert transform.constant("user id") == "USER_ID"
    assert transform.camel("user_id") == "userId"
    assert transform.pascal("user id") == "UserId"
    assert transform.title_case("hello world") == "Hello World"
    assert transform.snake("") == ""


def test_slugify_ascii_and_unicode_modes():
    assert transform.slugify("Hello, World!") == "hello-world"
    assert transform.slugify("Café au lait") == "cafe-au-lait"
    assert transform.slugify("中文 标题") == ""
    assert transform.slugify("中文 标题", allow_unicode=True) == "中文-标题"


def test_normalize_collapse_and_one_line():
    assert transform.normalize("ＡＢＣ") == "ABC"
    assert transform.normalize("  a  b  ", collapse=True, strip=True) == "a b"
    assert transform.one_line(" a\n b\t c ") == "a b c"
    assert transform.collapse_whitespace("a\n\n\n\nb") == "a\n\nb"
    with pytest.raises(TransformError, match="归一化"):
        transform.normalize("x", form="NFX")


def test_strip_control_and_halfwidth():
    assert transform.strip_control("a\x00b\u200bc", keep="\n") == "abc"
    assert transform.strip_control("a\nb") == "a\nb"
    assert transform.to_halfwidth("ＡＢＣ　１２３") == "ABC 123"


def test_indent_and_strip_lines():
    assert transform.indent("a\nb", prefix="> ") == "> a\n> b"
    assert transform.indent("a\nb", prefix="> ", skip_first=True) == "a\n> b"
    assert transform.strip_lines("  a  \n  b  ") == "a\nb"


# ---------------------------------------------------------------- 比较


def test_ratio_ignore_and_case_options():
    assert compare.ratio("hello world", "hello  world") < 1.0
    assert compare.ratio("hello world", "hello  world", ignore=" ") == 1.0
    assert compare.ratio("ABC", "abc") < 1.0
    assert compare.ratio("ABC", "abc", case_sensitive=False) == 1.0
    assert compare.quick_ratio("ABCDE", "ABCDF") >= compare.ratio("ABCDE", "ABCDF")
    assert compare.similar("python", "pyhton", threshold=0.8) is True
    assert compare.similar("python", "java", threshold=0.8) is False


def test_diff_counts_and_tags():
    result = compare.diff("kitten", "sitting")
    assert result.added == 3 and result.removed == 2
    assert {chunk.tag.value for chunk in result.changes} <= {
        "replace",
        "insert",
        "delete",
    }
    assert all(chunk.changed for chunk in result.changes)

    inserted = compare.diff("abc", "abcd")
    assert inserted.added == 1 and inserted.removed == 0
    assert inserted.changes[0].tag.value == "insert"


def test_diff_granularity_line_and_word():
    line_diff = compare.diff("a\nb\nc", "a\nB\nc", granularity="line")
    assert line_diff.granularity is Granularity.LINE
    assert line_diff.added == 1 and line_diff.removed == 1
    assert "B" in line_diff.changes[0].b_text

    word_diff = compare.diff(
        "hello world", "hello python", granularity=Granularity.WORD
    )
    assert word_diff.granularity is Granularity.WORD
    assert word_diff.changes[0].b_text.strip() == "python"

    with pytest.raises(TextError, match="未知比较粒度"):
        compare.diff("a", "b", granularity="nope")


def test_unified_diff_output():
    text = compare.unified("a\nb\nc", "a\nB\nc")
    assert "--- a" in text and "+++ b" in text
    assert "-b" in text and "+B" in text
    assert compare.unified("same", "same") == ""


def test_levenshtein_distance_and_limit():
    assert compare.distance("kitten", "sitting") == 3
    assert compare.distance("abc", "abc") == 0
    assert compare.distance("", "abc") == 3
    assert compare.distance("abc", "xyz", limit=1) == 2  # limit + 1 表示已超限


def test_best_matches_orders_and_filters():
    matches = best_matches("pythn", ["python", "path", "java"], limit=2)
    assert [item.text for item in matches] == ["python", "path"]
    assert matches[0].score > matches[1].score
    assert best_matches("pythn", ["java"], cutoff=0.9) == ()
    assert best_matches("x", ["y"], limit=0) == ()


# ---------------------------------------------------------------- 编解码


@pytest.mark.parametrize("encoding", ["base64", "base64url", "base32", "hex", "url"])
def test_codec_roundtrip(encoding):
    source = "中文 with space/plus+"
    encoded = codec.encode(source, encoding)
    assert codec.decode(encoded, encoding) == source


def test_codec_padding_and_auto_detect():
    assert codec.encode("hi", Encoding.BASE64) == "aGk="
    assert codec.encode("hi", Encoding.BASE64, padding=False) == "aGk"
    assert codec.decode("aGk", Encoding.BASE64) == "hi"
    assert codec.decode(codec.encode("hi", Encoding.HEX), Encoding.AUTO) == "hi"
    assert codec.decode("aGk=", "auto") == "hi"
    assert Encoding.HEX in codec.guess("e4b8ad")
    assert codec.guess("") == ()


def test_codec_rejects_bad_input():
    with pytest.raises(CodecError, match="未知编码"):
        codec.encode("x", "rot13")
    with pytest.raises(CodecError, match="base64"):
        codec.decode("!!!not-base64!!!")
    with pytest.raises(CodecError, match="不支持"):
        codec.encode("x", Encoding.AUTO)


@pytest.mark.parametrize("algorithm", ["gzip", "zlib"])
def test_compress_roundtrip(algorithm):
    payload = "重复内容" * 50
    packed = codec.compress(payload, algorithm=algorithm, level=9)
    assert len(packed) < len(payload.encode("utf-8"))
    assert codec.decompress(packed, algorithm=algorithm) == payload
    with pytest.raises(CodecError, match="未知压缩算法"):
        codec.compress("x", algorithm="bzip2")


# ---------------------------------------------------------------- 摘要


def test_hash_known_vectors():
    assert digest.hash_text("abc", "md5") == "900150983cd24fb0d6963f7d28e17f72"
    assert digest.hash_text("abc", "sha1") == "a9993e364706816aba3e25717850c26c9cd0d89d"
    assert (
        digest.hash_text("abc", HashAlgorithm.SHA256)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert digest.hash_text("abc", uppercase=True)[:2] == "BA"


def test_hmac_known_vector():
    # RFC 4231 测试用例 2
    assert (
        digest.hmac_text("what do ya want for nothing?", "Jefe", "sha256")
        == "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
    )
    with pytest.raises(DigestError, match="key"):
        digest.hmac_text("x", "")


def test_derive_key_is_deterministic_and_salted():
    first = digest.derive_key("pass", salt="c2FsdA==", iterations=1000)
    second = digest.derive_key("pass", salt="c2FsdA==", iterations=1000)
    other = digest.derive_key("pass", salt="c2FsdA==", iterations=2000)
    random_salted = digest.derive_key("pass", iterations=1000)
    assert isinstance(first, DerivedKey)
    assert first.key == second.key
    assert first.key != other.key
    assert random_salted.salt != first.salt
    assert len(base64.b64decode(first.key)) == 32
    with pytest.raises(DigestError, match="iterations"):
        digest.derive_key("pass", iterations=0)


def test_verify_and_token():
    value = digest.hash_text("abc")
    assert digest.verify(value, value) is True
    assert digest.verify(value.upper(), value) is True
    assert digest.verify(value, "deadbeef") is False
    assert digest.verify("", "x") is False
    assert digest.verify("令牌", "令牌") is True
    assert digest.verify("令牌", "令排") is False

    stamp = 1_700_000_000
    first = digest.token("user", salt="s", timestamp=stamp)
    second = digest.token("user", salt="s", timestamp=stamp)
    assert first == second
    assert len(digest.token("user", timestamp=stamp, length=8)) == 8
    assert digest.token("user", timestamp=stamp) != digest.token(
        "other", timestamp=stamp
    )


def test_file_hash_and_fingerprint(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("abc", encoding="utf-8")
    assert digest.file_hash(target) == digest.hash_text("abc")
    assert digest.file_hash(target, "md5") == digest.hash_text("abc", "md5")
    with pytest.raises(DigestError, match="文件不存在"):
        digest.file_hash(tmp_path / "missing.txt")

    expected = digest.hash_bytes(base64.b64encode(b"abc"), "md5")
    assert digest.fingerprint("abc") == expected


# ---------------------------------------------------------------- 加解密


def test_password_encrypt_roundtrip_and_envelope():
    token = crypto.encrypt("机密数据 abc", password="口令123", iterations=1000)
    assert token.startswith("cs1.")
    info = crypto.token_info(token)
    assert info["mode"] == "password" and info["iterations"] == 1000
    assert crypto.decrypt(token, password="口令123") == "机密数据 abc"
    with pytest.raises(DecryptError, match="口令错误|被篡改"):
        crypto.decrypt(token, password="wrong")


def test_key_mode_roundtrip_with_generated_and_arbitrary_key():
    key = crypto.new_key()
    token = crypto.encrypt("hello", key=key)
    assert not token.startswith("cs1.")
    assert crypto.decrypt(token, key=key) == "hello"
    assert crypto.token_info(token)["mode"] == "key"

    token2 = crypto.encrypt("hello", key="my-secret")
    assert crypto.decrypt(token2, key="my-secret") == "hello"
    with pytest.raises(DecryptError):
        crypto.decrypt(token2, key="other-secret")


def test_crypto_requires_exactly_one_secret():
    with pytest.raises(CryptoError, match="只能提供"):
        crypto.encrypt("x")
    with pytest.raises(CryptoError, match="只能提供"):
        crypto.encrypt("x", key="k", password="p")
    with pytest.raises(CryptoError, match="口令加密"):
        crypto.decrypt(crypto.encrypt("x", password="p"), key="k")
    with pytest.raises(CryptoError, match="密钥加密"):
        crypto.decrypt(crypto.encrypt("x", key="k"), password="p")
    with pytest.raises(DecryptError):
        crypto.decrypt("gAAAAA-invalid-token", key="k")
    with pytest.raises(DecryptError, match="为空"):
        crypto.decrypt("", key="k")


def test_file_encrypt_decrypt_roundtrip(tmp_path):
    source = tmp_path / "note.txt"
    source.write_text("文件内容", encoding="utf-8")
    sealed = crypto.encrypt_file(source, password="p")
    assert sealed.suffix == ".enc" and sealed.exists()
    with pytest.raises(CryptoError, match="目标已存在"):
        crypto.encrypt_file(source, password="p")

    restored = tmp_path / "restored.txt"
    assert crypto.decrypt_file(sealed, password="p", out=restored) == restored
    assert restored.read_text(encoding="utf-8") == "文件内容"
    with pytest.raises(CryptoError, match="文件不存在"):
        crypto.decrypt_file(tmp_path / "nope.enc", password="p")


# ---------------------------------------------------------------- 校验


@pytest.mark.parametrize(
    ("value", "predicate"),
    [
        ("cs@cstos.com", "is_email"),
        ("https://cstos.com/a?b=1", "is_url"),
        ("http://localhost:8000", "is_url"),
        ("sub.example.co.uk", "is_domain"),
        ("192.168.1.1", "is_ipv4"),
        ("2001:db8::1", "is_ipv6"),
        ("10.0.0.0/8", "is_cidr"),
        ("6f8d0f1e-6c1b-4a4e-9f0b-2c3d4e5f6a7b", "is_uuid"),
        ("00:1A:2B:3C:4D:5E", "is_mac"),
        ("0xFF", "is_hex"),
        ("5Lit5paH", "is_base64"),
        ('{"a": 1}', "is_json"),
        ("2024-02-29", "is_date"),
        ("20:00:00", "is_time"),
        ("2026-11-02T20:00:00Z", "is_datetime"),
        ("13812345678", "is_phone"),
        ("11010519491231002X", "is_id_card"),
        ("4111111111111111", "is_credit_card"),
        ("1.2.3-rc.1+build.5", "is_semver"),
        ("hello-world-2", "is_slug"),
        ("中文", "is_chinese"),
        ("abc中文", "has_chinese"),
        ("abc123", "is_alnum"),
        ("abc", "is_ascii"),
        ("-12", "is_integer"),
        ("1.5", "is_float"),
        ("1e-3", "is_scientific"),
        ("true", "is_boolean"),
    ],
)
def test_predicates_accept_valid_values(value, predicate):
    assert getattr(validators, predicate)(value) is True


@pytest.mark.parametrize(
    ("value", "predicate"),
    [
        ("not-an-email", "is_email"),
        ("cstos.com", "is_url"),
        ("-bad.com", "is_domain"),
        ("256.1.1.1", "is_ipv4"),
        ("not-a-uuid", "is_uuid"),
        ("00:1A:2B:3C:4D", "is_mac"),
        ("0xZZ", "is_hex"),
        ("abc", "is_base64"),
        ("{a: 1}", "is_json"),
        ("2026-02-29", "is_date"),
        ("25:00", "is_time"),
        ("2026-11-02", "is_datetime"),
        ("12812345678", "is_phone"),
        ("110105194912310021", "is_id_card"),
        ("4111111111111112", "is_credit_card"),
        ("1.2", "is_semver"),
        ("Hello", "is_slug"),
        ("abc", "is_chinese"),
        ("", "is_ascii"),
    ],
)
def test_predicates_reject_invalid_values(value, predicate):
    assert getattr(validators, predicate)(value) is False


def test_validate_registry_filters_options_by_signature():
    result = validators.validate(
        "abcdef", "length", "regexp", min_length=3, pattern=r"^a.*f$"
    )
    assert result.ok
    assert result.checks[0].detail == "min_length=3"
    assert "pattern" in result.checks[1].detail
    with pytest.raises(ValidationError, match="未知校验规则"):
        validators.validate("x", "nope")
    with pytest.raises(ValidationError, match="缺少参数"):
        validators.validate("x", "regexp")


def test_length_and_password_rules():
    assert validators.length("abc", min_length=3, max_length=3)
    assert validators.length(None, min_length=0)
    assert not validators.length(None, min_length=1)
    with pytest.raises(ValidationError, match="max_length"):
        validators.length("x", min_length=5, max_length=2)

    assert validators.is_password("abc12345")
    assert not validators.is_password("abcdefgh")
    assert validators.is_password("Abc12345", require=("digit", "upper"))
    with pytest.raises(ValidationError, match="未知口令类别"):
        validators.is_password("abc12345", require=("emoji",))


def test_is_regexp_reports_invalid_pattern():
    assert validators.is_regexp("abc123", r"[a-z]+\d+")
    with pytest.raises(ValidationError, match="正则"):
        validators.is_regexp("abc", "[")


def test_validate_chain_and_static_aliases():
    checker = Validate("cs@cstos.com").require("non_empty").require("email")
    assert checker.run() is True
    assert checker.result.ok and checker.valid

    failing = Validate(stop_on_failure=True).add(
        lambda value: False, name="always_false"
    )
    assert failing.run("x") is False
    assert failing.errors == ("always_false",)
    assert len(failing.result.checks) == 1

    assert Validate.is_email("cs@cstos.com")
    assert Validate.is_str("x")
    assert Validate.is_uuid("6f8d0f1e-6c1b-4a4e-9f0b-2c3d4e5f6a7b")
    assert validators.is_not_none(0) and not validators.is_not_none(None)
    assert set(CHECKS) >= {"email", "length", "regexp", "password"}


# ---------------------------------------------------------------- 门面


def test_text_facade_delegates():
    assert Text.snake("HTTPResponse") == "http_response"
    assert Text.width("中文") == 4
    assert Text.ratio("abc", "abc") == 1.0
    assert Text.validate("cs@cstos.com", "email").ok
    assert Text.encode("hi") == "aGk="
    assert Text.hash("abc").startswith("ba7816bf")
    token = Text.encrypt("x", password="p", iterations=1000)
    assert Text.decrypt(token, password="p") == "x"
    summary = Text.summary("中文abc")
    assert summary["width"] == 7 and summary["has_chinese"] is True


# ---------------------------------------------------------------- MCP


def _call(tool: str, payload: dict) -> dict:
    async def go() -> dict:
        async with Client(create_server()) as client:
            result = await client.call_tool(tool, payload)
            return result.data

    return asyncio.run(go())


def test_mcp_exposes_expected_tools():
    async def go() -> set[str]:
        async with Client(create_server()) as client:
            return {tool.name for tool in await client.list_tools()}

    names = asyncio.run(go())
    assert names == {
        "text_diff",
        "text_similarity",
        "text_find",
        "text_validate",
        "text_hash",
        "text_encode",
        "text_decode",
        "text_encrypt",
        "text_decrypt",
        "text_transform",
        "text_truncate",
        "text_mask",
        "text_width",
        "text_summary",
    }


def test_mcp_diff_and_validate():
    payload = _call("text_diff", {"a": "kitten", "b": "sitting", "unified": True})
    assert payload["added"] == 3 and payload["removed"] == 2
    assert "sitting" in payload["unified"]
    assert payload["truncated"] is False

    checked = _call(
        "text_validate", {"value": "abc", "checks": "non_empty,length", "max_length": 5}
    )
    assert checked["ok"] is True
    assert checked["passed"] == ["non_empty", "length"]


def test_mcp_encrypt_decrypt_via_env(monkeypatch):
    monkeypatch.setenv("CS_TEXT_TEST_PASS", "口令123")
    sealed = _call(
        "text_encrypt", {"text": "机密", "password_env": "CS_TEXT_TEST_PASS"}
    )
    assert sealed["mode"] == "password"
    opened = _call(
        "text_decrypt", {"token": sealed["token"], "password_env": "CS_TEXT_TEST_PASS"}
    )
    assert opened["text"] == "机密"

    key = crypto.new_key()
    sealed_key = _call("text_encrypt", {"text": "hi", "key": key})
    assert sealed_key["mode"] == "key"
    assert (
        _call("text_decrypt", {"token": sealed_key["token"], "key": key})["text"]
        == "hi"
    )


def test_mcp_reports_tool_errors():
    for tool, payload in [
        ("text_encrypt", {"text": "x"}),
        ("text_hash", {"path": "no-such-file.txt"}),
        ("text_validate", {"value": "x", "checks": "unknown"}),
        ("text_transform", {"text": "x", "style": "nope"}),
    ]:
        with pytest.raises(ToolError):
            _call(tool, payload)


# ---------------------------------------------------------------- CLI


def test_cli_hash_and_diff(capsys):
    assert main(["hash", "abc"]) == 0
    assert capsys.readouterr().out.strip() == digest.hash_text("abc")

    assert main(["diff", "abc", "abd", "-j"]) == 0
    assert '"ratio"' in capsys.readouterr().out

    assert main(["similar", "abc", "xyz"]) == 1
    assert "不相近" in capsys.readouterr().out


def test_cli_validate_exit_codes(capsys):
    assert main(["validate", "cs@cstos.com", "--check", "email"]) == 0
    assert "全部通过" in capsys.readouterr().out
    assert main(["validate", "bad", "--check", "email"]) == 1
    assert "不通过" in capsys.readouterr().out


def test_cli_encrypt_decrypt(monkeypatch, capsys):
    monkeypatch.setenv("CS_TEXT_CLI_PASS", "p")
    assert main(["encrypt", "hi", "--password-env", "CS_TEXT_CLI_PASS"]) == 0
    token = capsys.readouterr().out.strip()
    assert token.startswith("cs1.")
    assert main(["decrypt", token, "--password-env", "CS_TEXT_CLI_PASS"]) == 0
    assert capsys.readouterr().out.strip() == "hi"

    assert main(["encrypt", "hi"]) == 2
    assert "口令" in capsys.readouterr().err


def test_cli_encode_transform_width(capsys):
    assert main(["encode", "中文"]) == 0
    assert capsys.readouterr().out.strip() == "5Lit5paH"
    assert main(["transform", "user_id", "-s", "camel"]) == 0
    assert capsys.readouterr().out.strip() == "userId"
    assert main(["width", "中文abc"]) == 0
    assert "7 列" in capsys.readouterr().out
    assert main(["mask", "13812345678", "--keep", "3", "--tail", "4"]) == 0
    assert capsys.readouterr().out.strip() == "138****5678"
    assert main(["truncate", "中文标题很长", "--limit", "6"]) == 0
    assert capsys.readouterr().out.strip() == "中文…"
