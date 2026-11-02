"""命令行入口：``python -m cs.text ...``。

python -m cs.text diff 旧文件 新文件 -f
python -m cs.text diff "hello world" "hello python" --ignore " "
python -m cs.text similar a.txt b.txt -f
python -m cs.text find pythn python path python3
python -m cs.text validate cs@cstos.com --check email --check length --max-length 30
python -m cs.text hash 明文 --algorithm sha256
python -m cs.text hash --file report.pdf
python -m cs.text encrypt 机密 --password-env CS_PASS
python -m cs.text decrypt cs1.xxx --password-env CS_PASS
python -m cs.text encode 中文 --encoding base64
python -m cs.text transform user_id --style camel
python -m cs.text width 中文abc
python -m cs.text serve --transport http --port 8000

口令不通过命令行参数传递（会留在进程列表与 shell 历史里），
请用 ``--password-env VAR`` 或 ``--password-stdin``。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from . import codec, compare, crypto, digest, transform, validators
from .errors import TextError

__all__ = ["main"]


def _emit(
    payload: dict[str, object], *, as_json: bool, text: str | None = None
) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(text if text is not None else json.dumps(payload, ensure_ascii=False))


def _read_text(value: str | None, as_file: bool) -> str:
    if as_file:
        if not value:
            raise TextError("--file 需要提供路径")
        return Path(value).read_text(encoding="utf-8")
    if value is None or value == "-":
        if sys.stdin.isatty():
            raise TextError("请提供文本，或使用 - 从标准输入读取")
        return sys.stdin.read()
    return value


def _password(args: argparse.Namespace) -> str:
    env = getattr(args, "password_env", None)
    if env:
        secret = os.environ.get(env)
        if not secret:
            raise TextError(f"环境变量 {env} 未设置或为空")
        return secret
    if getattr(args, "password_stdin", False):
        line = sys.stdin.readline().strip()
        if not line:
            raise TextError("标准输入未读到口令")
        return line
    raise TextError("请用 --password-env VAR 或 --password-stdin 提供口令")


def _key(args: argparse.Namespace) -> str:
    env = getattr(args, "key_env", None)
    if getattr(args, "key", None):
        return args.key
    if env:
        secret = os.environ.get(env)
        if not secret:
            raise TextError(f"环境变量 {env} 未设置或为空")
        return secret
    raise TextError("请用 --key 或 --key-env VAR 提供密钥")


def _secrets(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """解析密钥/口令，返回 ``(key, password)``，二者必有一个为 ``None``。"""
    has_key = bool(getattr(args, "key", None) or getattr(args, "key_env", None))
    has_password = bool(
        getattr(args, "password_env", None) or getattr(args, "password_stdin", False)
    )
    if has_key and has_password:
        raise TextError("--key/--key-env 与 --password-env/--password-stdin 只能二选一")
    if has_key:
        return _key(args), None
    return None, _password(args)


def _add_secret_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--password-env", default=None, help="从该环境变量读取口令")
    parser.add_argument(
        "--password-stdin", action="store_true", help="从标准输入读一行口令"
    )
    parser.add_argument(
        "--key", default=None, help="Fernet 密钥（或任意口令，内部派生）"
    )
    parser.add_argument("--key-env", default=None, help="从该环境变量读取密钥")


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-j",
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="输出 JSON",
    )

    parser = argparse.ArgumentParser(
        prog="python -m cs.text",
        description="文本比较、校验、编解码、摘要、加解密与命名风格转换。",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    diff = sub.add_parser("diff", parents=[common], help="比较两段文本")
    diff.add_argument("a", help="左文本（--file 时为路径）")
    diff.add_argument("b", help="右文本（--file 时为路径）")
    diff.add_argument("-f", "--file", action="store_true", help="把参数当作文件路径")
    diff.add_argument("--ignore", default="", help="忽略的字符，如空格")
    diff.add_argument("--no-case", action="store_true", help="忽略大小写")
    diff.add_argument("--granularity", default="char", choices=["char", "word", "line"])
    diff.add_argument("--unified", action="store_true", help="输出统一差异文本")
    diff.add_argument("--context", type=int, default=3, help="统一差异的上下文行数")

    similar = sub.add_parser("similar", parents=[common], help="只看相似度")
    similar.add_argument("a")
    similar.add_argument("b")
    similar.add_argument("-f", "--file", action="store_true")
    similar.add_argument("--ignore", default="")
    similar.add_argument("--no-case", action="store_true")
    similar.add_argument("--threshold", type=float, default=0.6)

    find = sub.add_parser("find", parents=[common], help="找最相似的候选")
    find.add_argument("query")
    find.add_argument("candidates", nargs="+")
    find.add_argument("--limit", type=int, default=5)
    find.add_argument("--cutoff", type=float, default=0.0)
    find.add_argument("--ignore", default="")

    validate = sub.add_parser("validate", parents=[common], help="格式校验")
    validate.add_argument("value")
    validate.add_argument("--check", action="append", default=[], help="规则名，可重复")
    validate.add_argument("--min-length", type=int, default=0)
    validate.add_argument("--max-length", type=int, default=0)
    validate.add_argument("--pattern", default="")
    validate.add_argument("--require", default="")

    hashing = sub.add_parser("hash", parents=[common], help="摘要或 HMAC")
    hashing.add_argument("text", nargs="?", help="待摘要文本；缺省读标准输入")
    hashing.add_argument("-f", "--file", default=None, help="按文件内容计算")
    hashing.add_argument("-a", "--algorithm", default="sha256")
    hashing.add_argument("--hmac", default=None, help="HMAC 密钥")
    hashing.add_argument("--hmac-env", default=None, help="从环境变量读取 HMAC 密钥")

    encrypt = sub.add_parser("encrypt", parents=[common], help="加密文本或文件")
    encrypt.add_argument("text", nargs="?", help="待加密文本；缺省读标准输入")
    encrypt.add_argument(
        "-f", "--file", default=None, help="加密文件（读到内存后写出 .enc）"
    )
    encrypt.add_argument("-o", "--out", default=None, help="输出路径")
    encrypt.add_argument("--iterations", type=int, default=digest.DEFAULT_ITERATIONS)
    encrypt.add_argument("--overwrite", action="store_true")
    _add_secret_options(encrypt)

    decrypt = sub.add_parser("decrypt", parents=[common], help="解密文本或文件")
    decrypt.add_argument("token", nargs="?", help="密文；缺省读标准输入")
    decrypt.add_argument("-f", "--file", default=None, help="解密文件")
    decrypt.add_argument("-o", "--out", default=None, help="输出路径")
    decrypt.add_argument("--overwrite", action="store_true")
    _add_secret_options(decrypt)

    encode = sub.add_parser("encode", parents=[common], help="编码文本")
    encode.add_argument("text", nargs="?")
    encode.add_argument("-e", "--encoding", default="base64")
    encode.add_argument("--no-padding", action="store_true")

    decode = sub.add_parser("decode", parents=[common], help="解码文本")
    decode.add_argument("text", nargs="?")
    decode.add_argument("-e", "--encoding", default="auto")

    style = sub.add_parser("transform", parents=[common], help="命名风格/归一化转换")
    style.add_argument("text", nargs="?")
    style.add_argument(
        "-s",
        "--style",
        default="snake",
        choices=[
            "snake",
            "camel",
            "pascal",
            "kebab",
            "constant",
            "title",
            "slug",
            "lower",
            "upper",
            "normalize",
            "halfwidth",
            "one_line",
            "strip_control",
        ],
    )
    style.add_argument("--separator", default="-")

    width = sub.add_parser("width", parents=[common], help="显示宽度统计")
    width.add_argument("text", nargs="?")

    truncate = sub.add_parser("truncate", parents=[common], help="按宽度截断")
    truncate.add_argument("text", nargs="?")
    truncate.add_argument("-l", "--limit", type=int, default=40)
    truncate.add_argument("--suffix", default="…")
    truncate.add_argument("--word-boundary", action="store_true")

    mask = sub.add_parser("mask", parents=[common], help="打码")
    mask.add_argument("text", nargs="?")
    mask.add_argument("-k", "--keep", type=int, default=4)
    mask.add_argument("--tail", type=int, default=0)
    mask.add_argument("--char", default="*")

    serve = sub.add_parser("serve", parents=[common], help="启动 MCP 服务")
    serve.add_argument("--transport", default="stdio", choices=["stdio", "http"])
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--name", default="cs-text")

    return parser


def _serve(args: argparse.Namespace) -> int:
    from .mcp_server import create_server

    server = create_server(args.name)
    if args.transport == "http":
        server.run(transport="http", host=args.host, port=args.port)
    else:
        server.run(transport="stdio", show_banner=False)
    return 0


def _dispatch(args: argparse.Namespace) -> int:
    as_json: bool = getattr(args, "json", False)
    match args.command:
        case "diff":
            result = compare.diff(
                _read_text(args.a, args.file),
                _read_text(args.b, args.file),
                ignore=args.ignore or None,
                case_sensitive=not args.no_case,
                granularity=args.granularity,
            )
            if as_json:
                payload = result.to_dict()
                if args.unified:
                    payload["unified"] = result.unified(context=args.context)
                _emit(payload, as_json=True)
            elif args.unified:
                print(result.unified(context=args.context) or "（无差异）")
            else:
                print(str(result))
                for chunk in result.changes[:20]:
                    if chunk.tag.value == "replace":
                        print(f"  ~ {chunk.a_text!r} → {chunk.b_text!r}")
                    elif chunk.tag.value == "delete":
                        print(f"  - {chunk.a_text!r}")
                    else:
                        print(f"  + {chunk.b_text!r}")
            return 0

        case "similar":
            score = compare.ratio(
                _read_text(args.a, args.file),
                _read_text(args.b, args.file),
                ignore=args.ignore or None,
                case_sensitive=not args.no_case,
            )
            ok = score >= args.threshold
            _emit(
                {"ratio": round(score, 4), "similar": ok, "threshold": args.threshold},
                as_json=as_json,
                text=f"相似度 {score:.1%}（阈值 {args.threshold:.0%}）→ "
                + ("相近" if ok else "不相近"),
            )
            return 0 if ok else 1

        case "find":
            matches = compare.best_matches(
                args.query,
                args.candidates,
                limit=args.limit,
                cutoff=args.cutoff,
                ignore=args.ignore or None,
            )
            if as_json:
                _emit(
                    {
                        "count": len(matches),
                        "matches": [item.to_dict() for item in matches],
                    },
                    as_json=True,
                )
            elif not matches:
                print("没有达到阈值的候选")
            else:
                for item in matches:
                    print(f"{item.score:6.1%}  {item.text}")
            return 0

        case "validate":
            names = validators.parse_checks(args.check) or ("non_empty",)
            options: dict[str, object] = {}
            if args.min_length > 0:
                options["min_length"] = args.min_length
            if args.max_length > 0:
                options["max_length"] = args.max_length
            if args.pattern:
                options["pattern"] = args.pattern
            if args.require:
                options["require"] = validators.parse_checks(args.require)
            result = validators.validate(args.value, *names, **options)
            _emit(result.to_dict(), as_json=as_json, text=str(result))
            if not as_json:
                for check in result.checks:
                    print(f"  {check}")
            return 0 if result.ok else 1

        case "hash":
            if args.file:
                value = digest.file_hash(args.file, args.algorithm)
                _emit(
                    {"algorithm": args.algorithm, "path": args.file, "hex": value},
                    as_json=as_json,
                    text=value,
                )
                return 0
            text = _read_text(args.text, False)
            key = args.hmac
            if key is None and args.hmac_env:
                key = os.environ.get(args.hmac_env)
                if not key:
                    raise TextError(f"环境变量 {args.hmac_env} 未设置或为空")
            value = (
                digest.hmac_text(text, key, args.algorithm)
                if key
                else digest.hash_text(text, args.algorithm)
            )
            _emit(
                {"algorithm": args.algorithm, "hex": value, "length": len(value)},
                as_json=as_json,
                text=value,
            )
            return 0

        case "encrypt":
            key, password = _secrets(args)
            if args.file:
                target = crypto.encrypt_file(
                    args.file,
                    key=key,
                    password=password,
                    out=args.out,
                    iterations=args.iterations,
                    overwrite=args.overwrite,
                )
                _emit({"path": str(target)}, as_json=as_json, text=str(target))
                return 0
            text = _read_text(args.text, False)
            token = crypto.encrypt(
                text,
                key=key,
                password=password,
                iterations=args.iterations,
            )
            if args.out:
                Path(args.out).write_text(token, encoding="ascii")
                _emit({"path": args.out}, as_json=as_json, text=args.out)
            else:
                _emit(
                    {"token": token, "length": len(token)}, as_json=as_json, text=token
                )
            return 0

        case "decrypt":
            key, password = _secrets(args)
            if args.file:
                target = crypto.decrypt_file(
                    args.file,
                    key=key,
                    password=password,
                    out=args.out,
                    overwrite=args.overwrite,
                )
                _emit({"path": str(target)}, as_json=as_json, text=str(target))
                return 0
            token = _read_text(args.token, False).strip()
            text = crypto.decrypt(token, key=key, password=password)
            _emit({"text": text, "length": len(text)}, as_json=as_json, text=text)
            return 0

        case "encode":
            encoded = codec.encode(
                _read_text(args.text, False),
                args.encoding,
                padding=not args.no_padding,
            )
            _emit(
                {"encoding": args.encoding, "text": encoded, "length": len(encoded)},
                as_json=as_json,
                text=encoded,
            )
            return 0

        case "decode":
            decoded = codec.decode(_read_text(args.text, False), args.encoding)
            _emit(
                {"encoding": args.encoding, "text": decoded, "length": len(decoded)},
                as_json=as_json,
                text=decoded,
            )
            return 0

        case "transform":
            text = _read_text(args.text, False)
            mapping = {
                "snake": transform.snake,
                "camel": transform.camel,
                "pascal": transform.pascal,
                "kebab": transform.kebab,
                "constant": transform.constant,
                "title": transform.title_case,
                "slug": lambda value: transform.slugify(
                    value, separator=args.separator
                ),
                "lower": str.lower,
                "upper": str.upper,
                "normalize": lambda value: transform.normalize(
                    value, collapse=True, strip=True
                ),
                "halfwidth": transform.to_halfwidth,
                "one_line": transform.one_line,
                "strip_control": transform.strip_control,
            }
            value = mapping[args.style](text)
            _emit({"style": args.style, "text": value}, as_json=as_json, text=value)
            return 0

        case "width":
            text = _read_text(args.text, False)
            payload = {
                "chars": len(text),
                "width": transform.width(text),
                "lines": len(text.splitlines()) or (1 if text else 0),
                "has_chinese": validators.has_chinese(text),
            }
            _emit(
                payload,
                as_json=as_json,
                text=f"{payload['chars']} 字符 / {payload['width']} 列 / {payload['lines']} 行",
            )
            return 0

        case "truncate":
            text = transform.truncate(
                _read_text(args.text, False),
                args.limit,
                suffix=args.suffix,
                word_boundary=args.word_boundary,
            )
            _emit(
                {"text": text, "width": transform.width(text), "limit": args.limit},
                as_json=as_json,
                text=text,
            )
            return 0

        case "mask":
            value = transform.mask(
                _read_text(args.text, False),
                keep=args.keep,
                tail=args.tail,
                char=args.char,
            )
            _emit({"text": value}, as_json=as_json, text=value)
            return 0

        case "serve":
            return _serve(args)

    raise TextError(f"未知命令 {args.command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (TextError, OSError, ValueError) as exc:
        print(f"错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
