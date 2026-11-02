"""把 :mod:`cs.text` 暴露为 MCP 工具（FastMCP ≥ 4）。

启动方式::

    python -m cs.text serve                  # stdio，供 MCP 客户端拉起
    python -m cs.text serve --transport http --port 8000
    fastmcp run src/cs/text/mcp_server.py:mcp

约定：工具名统一 ``text_*``；全部是纯函数（只读，无副作用）；业务异常统一转
``ToolError``；口令/密钥支持 ``password_env`` / ``key_env`` 间接引用环境变量，
避免把明文写进对话记录。
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from . import codec, compare, crypto, digest, transform, validators
from .errors import TextError
from .models import Encoding

__all__ = ["create_server", "mcp"]

_SERVER_NAME = "cs-text"
_INSTRUCTIONS = """\
本服务器提供文本处理能力：文本比较（相似度/差异片段/统一差异/最相近候选）、
格式校验、编码解码、摘要与 HMAC、对称加解密、命名风格转换与显示宽度处理。

使用建议：
1. 比较长文本先用 text_similarity 粗筛，需要定位差异再用 text_diff；
2. 校验时 checks 可传多个（逗号分隔），如 "email,length"；length 需配合
   min_length / max_length，regexp 需配合 pattern；
3. 加解密请优先用 password_env 引用环境变量名，避免明文口令进入对话；
   口令加密的密文以 "cs1." 开头，自含盐与迭代次数，换会话也能解；
4. 中文宽度与终端一致（中文按 2 列），截断/折行都按显示宽度计算。
"""


def _guard[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """把领域异常转换成 MCP 可读错误。"""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except (TextError, OSError, ValueError) as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


def _done(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def _secret(direct: str, env: str, *, what: str) -> str:
    """取口令/密钥：优先显式参数，其次环境变量名。"""
    if direct:
        return direct
    if env:
        value = os.environ.get(env)
        if not value:
            raise TextError(f"环境变量 {env} 未设置或为空（{what}）")
        return value
    raise TextError(f"需要提供 {what}：直接传参或指定对应的环境变量名")


def _options(
    *,
    min_length: int = 0,
    max_length: int = 0,
    pattern: str = "",
    require: str = "",
) -> dict[str, Any]:
    """把 0/空串视为「未提供」，避免误把 max_length=0 当成真实约束。"""
    options: dict[str, Any] = {}
    if min_length > 0:
        options["min_length"] = min_length
    if max_length > 0:
        options["max_length"] = max_length
    if pattern:
        options["pattern"] = pattern
    if require:
        options["require"] = validators.parse_checks(require)
    return options


def create_server(name: str = _SERVER_NAME) -> FastMCP:
    """构造 FastMCP 实例；每次调用都得到独立 server（便于测试）。"""
    server = FastMCP(name=name, instructions=_INSTRUCTIONS)

    # -------------------------------------------------------- 比较

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def text_diff(
        a: str,
        b: str,
        ignore: str = "",
        case_sensitive: bool = True,
        granularity: str = "char",
        unified: bool = False,
        context: int = 3,
        limit: int = 200,
    ) -> dict[str, Any]:
        """比较两段文本，返回相似度与差异片段。

        granularity 取 char / word / line；ignore 为要忽略的字符（如空格）；
        unified=true 时额外返回统一差异文本；limit 限制返回的片段数量。
        """
        result = compare.diff(
            a,
            b,
            ignore=ignore or None,
            case_sensitive=case_sensitive,
            granularity=granularity,
        )
        payload = result.to_dict()
        if limit > 0:
            payload["chunks"] = payload["chunks"][:limit]
        payload["truncated"] = len(result.changes) > len(payload["chunks"])
        if unified:
            payload["unified"] = result.unified(context=context)
        return _done(**payload)

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def text_similarity(
        a: str,
        b: str,
        ignore: str = "",
        case_sensitive: bool = True,
        threshold: float = 0.6,
        quick: bool = False,
    ) -> dict[str, Any]:
        """只算相似度（0~1），并给出是否达到 threshold。"""
        score = compare.ratio(
            a, b, ignore=ignore or None, case_sensitive=case_sensitive, quick=quick
        )
        return _done(
            ratio=round(score, 4), similar=score >= threshold, threshold=threshold
        )

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def text_find(
        query: str,
        candidates: list[str],
        limit: int = 5,
        cutoff: float = 0.0,
        ignore: str = "",
        case_sensitive: bool = True,
    ) -> dict[str, Any]:
        """在候选列表中找出与 query 最相似的若干条。"""
        matches = compare.best_matches(
            query,
            candidates,
            limit=limit,
            cutoff=cutoff,
            ignore=ignore or None,
            case_sensitive=case_sensitive,
        )
        return _done(
            count=len(matches),
            matches=[match.to_dict() for match in matches],
        )

    # -------------------------------------------------------- 校验

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def text_validate(
        value: str,
        checks: str = "non_empty",
        min_length: int = 0,
        max_length: int = 0,
        pattern: str = "",
        require: str = "",
    ) -> dict[str, Any]:
        """按规则名校验文本；checks 支持逗号分隔的多个规则。

        可用规则见返回体中的 available；length 用 min_length/max_length，
        regexp 用 pattern，password 用 require（如 "digit,upper,symbol"）。
        """
        names = validators.parse_checks(checks) or ("non_empty",)
        result = validators.validate(
            value,
            *names,
            **_options(
                min_length=min_length,
                max_length=max_length,
                pattern=pattern,
                require=require,
            ),
        )
        return _done(**result.to_dict(), available=sorted(validators.CHECKS))

    # -------------------------------------------------------- 摘要

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def text_hash(
        text: str = "",
        path: str = "",
        algorithm: str = "sha256",
        key: str = "",
        key_env: str = "",
    ) -> dict[str, Any]:
        """计算文本或文件的摘要；给 key 时计算 HMAC。

        path 非空时忽略 text，按文件内容计算（大文件分块读取）。
        """
        secret = _secret(key, key_env, what="key") if (key or key_env) else ""
        if path:
            value = digest.file_hash(path, algorithm)
            return _done(
                kind="file",
                path=path,
                algorithm=algorithm,
                hex=value,
                length=len(value),
            )
        value = (
            digest.hmac_text(text, secret, algorithm)
            if secret
            else digest.hash_text(text, algorithm)
        )
        return _done(
            kind="hmac" if secret else "hash",
            algorithm=algorithm,
            hex=value,
            length=len(value),
        )

    # -------------------------------------------------------- 编解码

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def text_encode(
        text: str,
        encoding: str = "base64",
        charset: str = "utf-8",
        padding: bool = True,
    ) -> dict[str, Any]:
        """编码文本：base64 / base64url / base32 / hex / url。"""
        encoded = codec.encode(text, encoding, charset=charset, padding=padding)
        return _done(encoding=encoding, text=encoded, length=len(encoded))

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def text_decode(
        text: str,
        encoding: str = "auto",
        charset: str = "utf-8",
        errors: str = "replace",
    ) -> dict[str, Any]:
        """解码文本；encoding=auto 时按 base64/hex/url 顺序试探。"""
        resolved = Encoding.AUTO if encoding.lower() == "auto" else encoding
        decoded = codec.decode(text, resolved, charset=charset, errors=errors)
        return _done(encoding=encoding, text=decoded, length=len(decoded))

    # -------------------------------------------------------- 加解密

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def text_encrypt(
        text: str,
        password: str = "",
        password_env: str = "",
        key: str = "",
        key_env: str = "",
        iterations: int = digest.DEFAULT_ITERATIONS,
    ) -> dict[str, Any]:
        """加密文本；提供 key 走密钥模式，提供 password 走口令模式（自含盐）。"""
        has_key = bool(key or key_env)
        has_password = bool(password or password_env)
        if has_key and has_password:
            raise TextError("key 与 password 只能提供一个")
        if has_key:
            token = crypto.encrypt(
                text,
                key=_secret(key, key_env, what="key"),
            )
            return _done(token=token, mode="key", length=len(token))
        token = crypto.encrypt(
            text,
            password=_secret(password, password_env, what="password"),
            iterations=iterations,
        )
        return _done(
            token=token,
            mode="password",
            iterations=iterations,
            length=len(token),
        )

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def text_decrypt(
        token: str,
        password: str = "",
        password_env: str = "",
        key: str = "",
        key_env: str = "",
    ) -> dict[str, Any]:
        """解密文本；cs1. 开头的是口令模式，其余为 Fernet 密钥模式。"""
        info = crypto.token_info(token)
        if info["mode"] == "password":
            text = crypto.decrypt(
                token, password=_secret(password, password_env, what="password")
            )
        else:
            text = crypto.decrypt(token, key=_secret(key, key_env, what="key"))
        return _done(text=text, mode=info["mode"], length=len(text))

    # -------------------------------------------------------- 变换

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def text_transform(
        text: str, style: str = "snake", separator: str = "-"
    ) -> dict[str, Any]:
        """文本风格转换。

        style 取 snake / camel / pascal / kebab / constant / title / slug /
        lower / upper / normalize / halfwidth / one_line / strip_control。
        """
        key = style.strip().lower()
        match key:
            case "snake":
                value = transform.snake(text)
            case "camel":
                value = transform.camel(text)
            case "pascal":
                value = transform.pascal(text)
            case "kebab":
                value = transform.kebab(text)
            case "constant":
                value = transform.constant(text)
            case "title":
                value = transform.title_case(text)
            case "slug":
                value = transform.slugify(text, separator=separator or "-")
            case "lower":
                value = text.lower()
            case "upper":
                value = text.upper()
            case "normalize":
                value = transform.normalize(text, collapse=True, strip=True)
            case "halfwidth":
                value = transform.to_halfwidth(text)
            case "one_line":
                value = transform.one_line(text)
            case "strip_control":
                value = transform.strip_control(text)
            case _:
                allowed = "snake/camel/pascal/kebab/constant/title/slug/lower/upper/normalize/halfwidth/one_line/strip_control"
                raise TextError(f"未知风格 {style!r}，可选：{allowed}")
        return _done(style=key, text=value, length=len(value))

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def text_truncate(
        text: str,
        limit: int = 40,
        suffix: str = "…",
        word_boundary: bool = False,
    ) -> dict[str, Any]:
        """按显示列宽截断（中文算 2 列），超长时补 suffix。"""
        value = transform.truncate(
            text, limit, suffix=suffix, word_boundary=word_boundary
        )
        return _done(
            text=value,
            truncated=value != text,
            width=transform.width(value),
            limit=limit,
        )

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def text_mask(
        text: str, keep: int = 4, tail: int = 0, char: str = "*"
    ) -> dict[str, Any]:
        """打码：保留首 keep 与尾 tail 个字符。"""
        value = transform.mask(text, keep=keep, tail=tail, char=char or "*")
        return _done(text=value, length=len(value))

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def text_width(text: str) -> dict[str, Any]:
        """统计字符数、显示列宽、行数与是否含中文。"""
        return _done(
            chars=len(text),
            width=transform.width(text),
            lines=len(text.splitlines()) or (1 if text else 0),
            has_chinese=validators.has_chinese(text),
            ascii=text.isascii(),
        )

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def text_summary(text: str) -> dict[str, Any]:
        """一次给出长度、宽度、摘要、编码建议等画像信息。"""
        return _done(
            chars=len(text),
            width=transform.width(text),
            lines=len(text.splitlines()) or (1 if text else 0),
            words=len(transform.split_words(text)),
            has_chinese=validators.has_chinese(text),
            ascii=text.isascii(),
            sha256=digest.hash_text(text),
            base64=codec.encode(text, Encoding.BASE64),
        )

    return server


#: 供 ``fastmcp run ...:mcp`` 使用的默认实例
mcp = create_server()
