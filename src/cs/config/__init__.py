"""``cs.config`` —— TOML 配置文件的读写与校验（Python 3.12+）。

两种用法：**类方法调用**和 **MCP 工具调用**。

类方法::

    from dataclasses import dataclass, field
    from cs.config import ConfigManager, option

    @dataclass(slots=True)
    class Database:
        host: str = option("localhost", description="主机")
        port: int = option(5432, ge=1, le=65535)
        password: str = option("", secret=True)

    @dataclass(slots=True)
    class App:
        name: str = option("demo")
        debug: bool = False
        database: Database = field(default_factory=Database)

    cfg = ConfigManager(App, "config.toml")
    cfg.load()                       # 读盘 + 校验（文件不存在则用默认值）
    cfg.get("database.port")         # 5432
    cfg.set("database.port", 6543)   # 立刻校验
    cfg.save()                       # 原子写盘
    cfg.export("env")                # 导出成 KEY=VALUE
    print(cfg.instance().database.host)

不需要 schema 时用 :class:`TomlDocument`::

    from cs.config import TomlDocument

    doc = TomlDocument.load("config.toml", missing_ok=True)
    doc.set("server.timeout", 30).update({"log": {"level": "INFO"}}).save()
    doc.export("json")

MCP 工具::

    python -m cs.config serve --transport stdio     # 或 http
    # 工具：config_open / config_get / config_set / config_update / config_save
    #      config_export / config_init / config_validate / config_schema …

设计要点
--------
* **分层**：``codec``（TOML 编解码）→ ``document``（文件 ↔ 字典）→ ``manager``
  （dataclass 校验）→ ``mcp_server`` / ``__main__``。上层不碰下层细节。
* **零第三方依赖**：读用标准库 ``tomllib``，写用自带的写出器，不依赖 pydantic /
  tomli_w / python-dotenv。
* **异常多继承内建类型**：``ConfigValidationError`` 也是 ``ValueError``，
  ``ConfigFileNotFoundError`` 也是 ``FileNotFoundError``，老代码可以照旧捕获。
* **一次报全**：校验失败会列出所有问题（带点号路径），而不是撞到第一个就停。
* **不写坏值**：``set`` / ``update`` 立即校验，``save`` 前再校验一次，并用
  「临时文件 + os.replace」原子替换，避免留下半截配置。
* **敏感字段脱敏**：``option(secret=True)`` 的字段在导出时替换成 ``***``。
* **无副作用导入**：不再像旧实现那样在 import 时读文件、改环境变量。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .codec import dumps, format_value, load_file, loads, read_file, write_file
from .document import TomlDocument, lookup, parse_path
from .errors import (
    ConfigError,
    ConfigExistsError,
    ConfigFileNotFoundError,
    ConfigKeyError,
    ConfigLoadError,
    ConfigParseError,
    ConfigPathError,
    ConfigSchemaError,
    ConfigSessionError,
    ConfigTypeError,
    ConfigUnsupportedError,
    ConfigValidationError,
    ConfigWriteError,
    Issue,
)
from .export import dumps_env, dumps_json, flatten, render_template
from .fields import MISSING, OPTION_KEY, Option, get_option, option
from .manager import ConfigManager
from .schema import FieldSpec, build, describe, dump, field_specs
from .sessions import DEFAULT_SESSION, ConfigRegistry, LiveConfig

if TYPE_CHECKING:  # pragma: no cover - 只为类型检查，避免运行时拉起 fastmcp
    from fastmcp import FastMCP

__all__ = [
    "DEFAULT_SESSION",
    "MISSING",
    "OPTION_KEY",
    "ConfigError",
    "ConfigExistsError",
    "ConfigFileNotFoundError",
    "ConfigKeyError",
    "ConfigLoadError",
    "ConfigManager",
    "ConfigParseError",
    "ConfigPathError",
    "ConfigRegistry",
    "ConfigSchemaError",
    "ConfigSessionError",
    "ConfigTypeError",
    "ConfigUnsupportedError",
    "ConfigValidationError",
    "ConfigWriteError",
    "FieldSpec",
    "Issue",
    "LiveConfig",
    "Option",
    "TomlDocument",
    "build",
    "create_server",
    "describe",
    "dump",
    "dumps",
    "dumps_env",
    "dumps_json",
    "field_specs",
    "flatten",
    "format_value",
    "get_option",
    "load_file",
    "loads",
    "lookup",
    "option",
    "parse_path",
    "read_file",
    "render_template",
    "write_file",
]


def create_server(name: str = "cs-config", **kwargs: Any) -> FastMCP:
    """构造 MCP 服务器（延迟导入 ``fastmcp``，不用 MCP 的人无需安装它）。"""
    from .mcp_server import create_server as _create

    return _create(name, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "mcp":
        from .mcp_server import mcp as server

        return server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
