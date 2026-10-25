"""示例配置模型：既是可直接使用的默认模板，也是 ``config_*`` 工具的可用模型清单。

字段全部用 :func:`cs.config.option` 声明，因此自带默认值、约束与说明，
``config init --template`` 能把它们渲染成一份带注释、可直接改的 TOML。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .fields import option

__all__ = [
    "API",
    "MODELS",
    "App",
    "Database",
    "ExternalServices",
    "Global",
    "LogLevel",
    "OpenAI",
    "Redis",
    "get_model",
    "model_names",
]


class LogLevel(StrEnum):
    """日志级别。"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass(slots=True)
class App:
    """应用基础信息。"""

    name: str = option("my-app", description="应用名")
    version: str = option("0.1.0", description="版本号")
    debug: bool = option(False, description="调试模式")
    log_level: LogLevel = option(  # noqa: RUF009 - option() 返回 Field，非可变默认值
        LogLevel.INFO, description="日志级别", choices=tuple(LogLevel)
    )
    allowed_origins: list[str] = option(  # noqa: RUF009 - option() 返回 Field
        default_factory=lambda: ["http://localhost:3000"],
        description="CORS 允许的来源，逗号分隔或 TOML 数组",
    )


@dataclass(slots=True)
class Database:
    """数据库连接。"""

    host: str = option("localhost", description="主机")
    port: int = option(5432, ge=1, le=65535, description="端口")
    username: str = option("postgres", description="用户名")
    password: str = option("", secret=True, description="密码")
    database: str = option("app", description="库名")
    pool_size: int = option(5, ge=1, le=100, description="连接池大小")
    max_overflow: int = option(10, ge=0, le=50, description="池外可溢出连接数")
    echo: bool = option(False, description="打印 SQL")


@dataclass(slots=True)
class Redis:
    """Redis 连接。"""

    host: str = option("localhost", description="主机")
    port: int = option(6379, ge=1, le=65535, description="端口")
    db: int = option(0, ge=0, le=15, description="库号")
    password: str = option("", secret=True, description="密码")
    decode_responses: bool = option(True, description="返回 str 而非 bytes")


@dataclass(slots=True)
class OpenAI:
    """OpenAI 兼容接口。"""

    api_key: str = option("", secret=True, description="API Key")
    base_url: str = option("https://api.openai.com/v1", description="接口地址")
    model: str = option("gpt-4o-mini", description="默认模型")
    timeout: float = option(30.0, ge=1, le=300, description="超时秒数")


@dataclass(slots=True)
class ExternalServices:
    """外部服务汇总。"""

    openai: OpenAI = field(default_factory=OpenAI)


@dataclass(slots=True)
class API:
    """对外 HTTP 接口参数。"""

    prefix: str = option("/api", description="路由前缀")
    timeout: int = option(30, ge=1, le=300, description="请求超时秒数")
    rate_limit: int = option(100, ge=1, description="每分钟允许的请求数")


@dataclass(slots=True)
class Global:
    """全局配置（app / database / redis / api / external_services）。"""

    app: App = field(default_factory=App)
    database: Database = field(default_factory=Database)
    redis: Redis = field(default_factory=Redis)
    api: API = field(default_factory=API)
    external_services: ExternalServices = field(default_factory=ExternalServices)


#: 名字 → 模型，供 CLI / MCP 按名引用
MODELS: dict[str, type] = {
    "app": App,
    "database": Database,
    "redis": Redis,
    "openai": OpenAI,
    "api": API,
    "external_services": ExternalServices,
    "global": Global,
}


def model_names() -> list[str]:
    return sorted(MODELS)


def get_model(name: str) -> type:
    """按名字取模型；``name`` 大小写不敏感，``raw`` 表示无 schema。"""
    key = name.strip().lower()
    if key not in MODELS:
        raise KeyError(f"未知配置模型：{name}（可选 {', '.join(model_names())}）")
    return MODELS[key]
