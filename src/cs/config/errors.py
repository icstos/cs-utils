"""``cs.config`` 的异常体系。

设计原则：每个领域异常都**额外多继承一个内建异常**，这样调用方既能用领域语义
精确捕获，也能用 ``except ValueError`` / ``OSError`` / ``KeyError`` 这类内建语义兜底，
不会因为引入本模块而破坏既有的错误处理习惯。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ConfigError",
    "ConfigExistsError",
    "ConfigFileNotFoundError",
    "ConfigKeyError",
    "ConfigLoadError",
    "ConfigParseError",
    "ConfigPathError",
    "ConfigSchemaError",
    "ConfigSessionError",
    "ConfigUnsupportedError",
    "ConfigValidationError",
    "ConfigWriteError",
    "Issue",
]


class ConfigError(Exception):
    """``cs.config`` 所有异常的基类。"""


class ConfigPathError(ConfigError, ValueError):
    """配置路径（点号路径 / 文件路径）不合法。"""


class ConfigKeyError(ConfigError, KeyError):
    """请求的配置键不存在。"""

    def __str__(self) -> str:  # KeyError 会给消息套一层引号，这里去掉
        return str(self.args[0]) if self.args else "配置键不存在"


class ConfigTypeError(ConfigError, TypeError):
    """值的类型与模型声明不符，且无法安全转换。"""


class ConfigFileNotFoundError(ConfigError, FileNotFoundError):
    """配置文件不存在。"""


class ConfigExistsError(ConfigError, FileExistsError):
    """配置文件已存在（初始化时未允许覆盖）。"""


class ConfigParseError(ConfigError, ValueError):
    """TOML 语法错误。"""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"TOML 解析失败（{path}）：{reason}")


class ConfigSessionError(ConfigError, LookupError):
    """会话不存在（未打开或已超时回收）。"""


class ConfigLoadError(ConfigError, OSError):
    """读取配置文件失败（编码错误、权限不足等）。"""


class ConfigWriteError(ConfigError, OSError):
    """写入配置文件失败。"""


class ConfigSchemaError(ConfigError):
    """模型本身不合法（不是 dataclass、字段无法解析等）。"""


class ConfigUnsupportedError(ConfigError, NotImplementedError):
    """遇到本模块未支持的注解或导出格式。"""


@dataclass(frozen=True, slots=True)
class Issue:
    """单条校验问题：定位到具体的点号路径。"""

    path: str
    message: str
    value: Any = field(default=None, repr=False)

    def __str__(self) -> str:
        return f"{self.path or '<root>'}: {self.message}"


class ConfigValidationError(ConfigError, ValueError):
    """聚合校验失败：一次列出**所有**问题，而不是撞到第一个就停。"""

    def __init__(self, issues: list[Issue] | tuple[Issue, ...]) -> None:
        self.issues: tuple[Issue, ...] = tuple(issues)
        detail = "\n".join(f"  - {issue}" for issue in self.issues)
        super().__init__(f"配置校验失败（{len(self.issues)} 项）：\n{detail}")

    def __iter__(self):  # 方便 for issue in exc
        return iter(self.issues)
