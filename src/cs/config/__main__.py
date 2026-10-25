"""命令行入口：``python -m cs.config ...``（也可用 ``cs-config``）。

示例::

    cs-config init config.toml --model global      # 生成带注释的骨架
    cs-config show --json
    cs-config get app.name
    cs-config get redis.port
    cs-config set app.debug true --save
    cs-config set database.port 6543 --model global --save
    cs-config unset redis.password --save
    cs-config validate --model global
    cs-config export --fmt env
    cs-config schema global
    cs-config models
    cs-config serve --transport http --port 8000

``--model`` 默认 ``raw``（不做类型校验的纯 TOML 操作）；涉及校验/默认值的命令
需要显式给出模型名（见 ``cs-config models``）。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any

from . import codec
from .errors import ConfigError, ConfigValidationError
from .manager import ConfigManager
from .samples import get_model, model_names

__all__ = ["main"]

#: 与 samba 入口保持一致：别让底层库的常规日志污染命令行输出
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cs.config",
        description="TOML 配置读写：初始化、加载、查询、更新、保存、导出、校验。",
    )
    sub = parser.add_subparsers(dest="command")

    def cmd(name: str, help_text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_text)

    def with_file_and_model(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "-f", "--file", default="config.toml", help="配置文件路径，默认 config.toml"
        )
        target.add_argument(
            "-m",
            "--model",
            default="raw",
            help=f"配置模型名（{', '.join(model_names())}）或 raw",
        )
        target.add_argument(
            "-e", "--env-prefix", default=None, help="环境变量前缀，如 CS_"
        )

    init = cmd("init", help_text="初始化配置文件")
    with_file_and_model(init)
    init.add_argument("--force", action="store_true", help="已存在时覆盖")
    init.add_argument(
        "--plain", action="store_true", help="写纯默认值而不是带注释的模板"
    )

    show = cmd("show", help_text="显示整份配置")
    with_file_and_model(show)
    show.add_argument("--json", action="store_true", help="输出 JSON")
    show.add_argument("--redact", action="store_true", help="敏感字段打码")

    get = cmd("get", help_text="读取一个配置项")
    with_file_and_model(get)
    get.add_argument("path", nargs="?", default="", help="点号路径，如 app.debug")
    get.add_argument("--json", action="store_true", help="输出 JSON")

    set_ = cmd("set", help_text="设置一个配置项")
    with_file_and_model(set_)
    set_.add_argument("path")
    set_.add_argument("value")
    set_.add_argument("--save", action="store_true", help="立即写盘")
    set_.add_argument("--json", action="store_true")

    unset = cmd("unset", help_text="删除一个配置项")
    with_file_and_model(unset)
    unset.add_argument("path")
    unset.add_argument("--save", action="store_true")

    validate = cmd("validate", help_text="按模型校验配置")
    with_file_and_model(validate)
    validate.add_argument("--json", action="store_true")

    export = cmd("export", help_text="导出配置")
    with_file_and_model(export)
    export.add_argument(
        "--fmt", choices=("toml", "json", "env"), default="toml", help="导出格式"
    )
    export.add_argument("-o", "--output", default=None, help="写入文件而不是标准输出")
    export.add_argument("--redact", action="store_true", help="敏感字段打码")
    export.add_argument(
        "--effective",
        action="store_true",
        help="导出叠加了环境变量后的值（默认只看文件）",
    )

    schema = cmd("schema", help_text="查看模型的字段与约束")
    with_file_and_model(schema)

    cmd("models", help_text="列出可用模型")

    serve = cmd("serve", help_text="以 MCP 服务方式启动")
    serve.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--name", default="cs-config")

    return parser


def _manager(args: argparse.Namespace) -> ConfigManager[Any]:
    model = None if args.model.lower() == "raw" else get_model(args.model)
    return ConfigManager(model, args.file, env_prefix=args.env_prefix)


def _emit(payload: dict[str, Any], *, as_json: bool, text: str) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(text)


def _parse_value(text: str) -> Any:
    """把命令行上的字面量当 TOML 值解析：``42`` / ``true`` / ``[1, 2]`` / ``"x"``。"""
    try:
        return codec.loads(f"value = {text}")["value"]
    except ConfigError:
        return text


def _require_model(args: argparse.Namespace) -> None:
    if args.model.lower() == "raw":
        from .errors import ConfigSchemaError

        raise ConfigSchemaError(
            f"该命令需要 --model（可选 {', '.join(model_names())}），不能用 raw"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    command = args.command or "show"

    try:
        match command:
            case "serve":
                from .mcp_server import create_server

                server = create_server(args.name)
                if args.transport == "http":
                    server.run(transport="http", host=args.host, port=args.port)
                else:
                    server.run(transport="stdio", show_banner=False)
                return 0

            case "models":
                print("\n".join(model_names()))
                print("raw  # 不做 schema 校验")
                return 0

            case "init":
                manager = _manager(args)
                path = manager.initialize(
                    overwrite=args.force,
                    template=not args.plain and args.model.lower() != "raw",
                )
                print(f"已初始化：{path}")
                if args.model.lower() != "raw":
                    print(f"模型：{args.model}")
                return 0

            case "show":
                manager = _manager(args).load()
                if args.json:
                    _emit(manager.data(), as_json=True, text="")
                else:
                    print(
                        manager.export("toml", redact=args.redact, source="effective")
                    )
                return 0

            case "get":
                manager = _manager(args).load()
                value = manager.get(args.path)
                if args.json or isinstance(value, (dict, list)):
                    _emit(
                        {"path": args.path or "<root>", "value": value},
                        as_json=True,
                        text="",
                    )
                else:
                    print(value)
                return 0

            case "set":
                manager = _manager(args).load()
                value = (
                    args.value
                    if manager.model is not None
                    else _parse_value(args.value)
                )
                manager.set(args.path, value, save=args.save)
                shown = manager.get(args.path, source="file")
                _emit(
                    {"path": args.path, "value": shown, "saved": args.save},
                    as_json=args.json,
                    text=f"{args.path} = {shown!r}{' （已保存）' if args.save else ''}",
                )
                return 0

            case "unset":
                manager = _manager(args).load()
                removed = manager.unset(args.path, save=args.save)
                print(f"{'已删除' if removed else '不存在'}：{args.path}")
                return 0 if removed else 1

            case "validate":
                _require_model(args)
                manager = _manager(args)
                try:
                    manager.load()
                    manager.validate()
                except ConfigValidationError as exc:
                    if args.json:
                        _emit(
                            {
                                "valid": False,
                                "error": str(exc),
                                "issues": [
                                    {"path": issue.path, "message": issue.message}
                                    for issue in exc.issues
                                ],
                            },
                            as_json=True,
                            text="",
                        )
                    else:
                        print(f"校验失败：{exc}", file=sys.stderr)
                    return 1
                _emit(
                    {"valid": True, "model": args.model, "file": args.file},
                    as_json=args.json,
                    text=f"校验通过：{args.file}（模型 {args.model}）",
                )
                return 0

            case "export":
                manager = _manager(args).load()
                text = manager.export(
                    args.fmt,
                    redact=args.redact,
                    source="effective" if args.effective else "file",
                )
                if args.output:
                    codec.write_file(args.output, text)
                    print(f"已导出：{args.output}")
                else:
                    print(text, end="")
                return 0

            case "schema":
                _require_model(args)
                model = get_model(args.model)
                print(
                    json.dumps(
                        ConfigManager(model).schema(), ensure_ascii=False, indent=2
                    )
                )
                return 0

            case _:  # pragma: no cover - argparse 已限定取值范围
                parser.print_help()
                return 2

    except ConfigError as exc:
        print(f"错误: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"错误: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
