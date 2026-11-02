"""``cs.env`` 的异常层级。

全部继承 :class:`EnvError`；命令相关同时继承 ``OSError``、进程相关同时继承
``OSError``，方便调用方按系统错误统一兜底。
"""

from __future__ import annotations

__all__ = [
    "CommandError",
    "CommandNotFoundError",
    "CommandTimeout",
    "EnvError",
    "ProcessError",
    "ProcessNotFoundError",
    "UnsafeCommandError",
]


class EnvError(Exception):
    """env 模块的异常基类。"""


# ---------------------------------------------------------------- 命令


class CommandError(EnvError, OSError):
    """命令执行失败（找不到可执行文件、超时或显式 check 未通过）。"""


class CommandNotFoundError(CommandError):
    """可执行文件不存在或不在 ``PATH`` 中。"""


class CommandTimeout(CommandError):
    """命令超过 ``timeout`` 仍占用执行权。"""


class UnsafeCommandError(CommandError, ValueError):
    """命令字符串含 shell 元字符，或被判定为不安全。"""


# ---------------------------------------------------------------- 进程


class ProcessError(EnvError, OSError):
    """进程操作失败（权限不足、拒绝执行、无法结束等）。"""


class ProcessNotFoundError(ProcessError):
    """目标进程不存在（或已退出）。"""
