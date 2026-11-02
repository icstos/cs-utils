"""安全的子进程执行。

设计要点：

- **永不经过 shell**：``shell=False`` + argv 列表，参数由 Python 直接传给内核，
  不存在注入面；字符串形式只做分词，出现 ``| & ; < > $ ` `` 等 shell 元字符
  直接拒绝（:class:`UnsafeCommandError`），避免“以为能管道其实不能”的误解。
- **管道用数据表达**：上一条的 stdout 作为下一条的 ``stdin``，无需 shell。
- **超时不抛异常**：返回 ``timed_out=True``、``returncode=124`` 的结果对象；
  只有显式 ``check=True`` 才在非零退出时抛 :class:`CommandError`。
- **输出可读**：按 UTF-8 → 系统 locale → GBK 逐级解码，Windows 中文不再乱码。
"""

from __future__ import annotations

import locale
import os
import shlex
import shutil
import subprocess
import time
import webbrowser
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .errors import (
    CommandError,
    CommandNotFoundError,
    UnsafeCommandError,
)
from .models import CommandResult

__all__ = [
    "Command",
    "execute",
    "open_url",
    "run",
    "run_all",
    "split",
    "which",
]

#: 交给 shell 解释的字符：出现即拒绝，防止“看起来能跑”的隐式行为
SHELL_META: frozenset[str] = frozenset("|&;<>$`\n\r\0")

#: 超时时的约定退出码（与 GNU timeout 保持一致）
TIMEOUT_CODE = 124

_DEFAULT_MAX_OUTPUT = 2_000_000


def split(text: str) -> tuple[str, ...]:
    """把命令行字符串拆成参数元组。

    只处理空白分隔与成对引号，**不做**变量展开、通配符、重定向。
    含 shell 元字符时抛 :class:`UnsafeCommandError`。
    """
    banned = sorted({char for char in text if char in SHELL_META})
    if banned:
        readable = " ".join(repr(char) for char in banned)
        raise UnsafeCommandError(
            f"命令字符串含 shell 元字符（{readable}）；请改用参数列表，"
            "或把上一条命令的输出通过 stdin 传入"
        )
    # posix=False 保留反斜杠，Windows 路径才不会被吃掉；随后去掉成对引号
    try:
        tokens = shlex.split(text, posix=False)
    except ValueError as exc:
        raise UnsafeCommandError(f"无法解析命令字符串：{exc}") from exc
    args = tuple(
        token[1:-1]
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'"
        else token
        for token in tokens
    )
    if not args:
        raise UnsafeCommandError("命令为空")
    return args


def _decode(data: bytes | None, encoding: str | None = None) -> str:
    """逐级尝试 UTF-8 / 系统 locale / GBK，最后兜底替换解码。

    先试 UTF-8：现代工具大多直接输出 UTF-8，且中文 UTF-8 字节序列几乎不可能
    被 GBK 合法解码，误判概率远低于反过来。
    """
    if not data:
        return ""
    if encoding:
        return data.decode(encoding, "replace")
    for codec in dict.fromkeys(
        ("utf-8", locale.getpreferredencoding(False) or "utf-8", "gbk")
    ):
        try:
            return data.decode(codec)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", "replace")


def _encode(text: str, encoding: str | None = None) -> bytes:
    """stdin 编码：Windows 子进程通常按代码页读，优先用 locale。"""
    if encoding:
        return text.encode(encoding)
    preferred = ("utf-8", locale.getpreferredencoding(False) or "utf-8", "gbk")
    ordered = preferred[::-1] if os.name == "nt" else preferred
    for codec in dict.fromkeys(ordered):
        try:
            return text.encode(codec)
        except (UnicodeEncodeError, LookupError):
            continue
    return text.encode("utf-8", "replace")


def _cap(text: str, limit: int) -> tuple[str, bool]:
    """截断过长的输出，返回 ``(文本, 是否截断)``。"""
    if limit <= 0 or len(text) <= limit:
        return text, False
    return f"{text[:limit]}\n…[输出已截断，原始 {len(text)} 字符]", True


@dataclass(frozen=True, slots=True)
class Command:
    """一条待执行的命令。

    >>> Command(("git", "status", "--short")).argv
    ('git', 'status', '--short')
    """

    argv: tuple[str, ...]
    cwd: Path | str | None = None
    env: Mapping[str, str] | None = None
    stdin: str | None = None
    timeout: float | None = 30.0
    check: bool = False
    max_output: int = _DEFAULT_MAX_OUTPUT
    encoding: str | None = None

    def __post_init__(self) -> None:
        argv = tuple(str(item) for item in self.argv)
        if not argv:
            raise UnsafeCommandError("argv 不能为空")
        if any("\0" in item for item in argv):
            raise UnsafeCommandError("参数不能包含 NUL")
        if isinstance(self.cwd, str):
            object.__setattr__(self, "cwd", Path(self.cwd))
        object.__setattr__(self, "argv", argv)

    @classmethod
    def from_string(cls, text: str, **kwargs: Any) -> Command:
        """从命令行字符串构造（安全分词，不用 shell）。"""
        return cls(argv=split(text), **kwargs)

    def with_stdin(self, text: str) -> Command:
        """派生一条带 stdin 的命令，用于代替管道。"""
        return replace(self, stdin=text)

    def __str__(self) -> str:
        return " ".join(self.argv)


def _as_command(source: Command | str | Sequence[str], **overrides: Any) -> Command:
    match source:
        case Command():
            return replace(source, **overrides) if overrides else source
        case str():
            return Command.from_string(source, **overrides)
        case Sequence():
            return Command(argv=tuple(source), **overrides)
        case _:
            raise TypeError(f"不支持的命令类型：{type(source).__name__}")


def execute(command: Command) -> CommandResult:
    """执行一条 :class:`Command`，始终返回结果对象（超时也不抛）。"""
    started = time.perf_counter()
    kwargs: dict[str, Any] = {
        "shell": False,
        "capture_output": True,
        "cwd": command.cwd,
        "env": dict(command.env) if command.env is not None else None,
        "timeout": command.timeout,
    }
    stdin = (
        _encode(command.stdin, command.encoding) if command.stdin is not None else None
    )
    if os.name == "nt":
        # 避免在 GUI 程序里弹出黑色控制台窗口
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        completed = subprocess.run(
            list(command.argv), check=False, input=stdin, **kwargs
        )
    except FileNotFoundError as exc:
        raise CommandNotFoundError(f"找不到可执行文件：{command.argv[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started
        stdout, cut_a = _cap(_decode(exc.stdout, command.encoding), command.max_output)
        stderr, cut_b = _cap(_decode(exc.stderr, command.encoding), command.max_output)
        return CommandResult(
            argv=command.argv,
            returncode=TIMEOUT_CODE,
            stdout=stdout,
            stderr=stderr or f"命令在 {command.timeout}s 内未结束，已终止",
            duration=round(duration, 3),
            timed_out=True,
            truncated=cut_a or cut_b,
        )
    except OSError as exc:
        raise CommandError(f"无法执行 {command.argv[0]}：{exc}") from exc

    duration = round(time.perf_counter() - started, 3)
    stdout, cut_a = _cap(
        _decode(completed.stdout, command.encoding), command.max_output
    )
    stderr, cut_b = _cap(
        _decode(completed.stderr, command.encoding), command.max_output
    )
    result = CommandResult(
        argv=command.argv,
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        duration=duration,
        timed_out=False,
        truncated=cut_a or cut_b,
    )
    if command.check and not result.ok:
        detail = result.stderr.strip()[:400] or result.stdout.strip()[:400]
        raise CommandError(f"命令失败（退出码 {result.returncode}）：{detail}")
    return result


def run(source: Command | str | Sequence[str], **overrides: Any) -> CommandResult:
    """执行命令。``source`` 可以是 :class:`Command`、字符串或参数列表。

    >>> run(["git", "rev-parse", "--short", "HEAD"]).text
    'b274cb3'
    """
    return execute(_as_command(source, **overrides))


def run_all(
    *sources: Command | str | Sequence[str], stop_on_error: bool = True
) -> tuple[CommandResult, ...]:
    """顺序执行多条命令（替代 ``&&`` 串联），默认首条失败即停。"""
    results: list[CommandResult] = []
    for source in sources:
        result = run(source)
        results.append(result)
        if stop_on_error and not result.ok:
            break
    return tuple(results)


def which(name: str) -> Path | None:
    """在 ``PATH`` 中查找可执行文件。"""
    found = shutil.which(name)
    return Path(found) if found else None


def open_url(url: str) -> bool:
    """用系统默认浏览器打开 http/https/mailto 链接。"""
    if urlparse(url).scheme not in {"http", "https", "mailto"}:
        raise ValueError(f"仅支持 http/https/mailto 链接：{url}")
    return webbrowser.open(url)
