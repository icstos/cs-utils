"""常用装饰器：重试、调用日志、失败通知、输出捕获、弃用标记。

同步与异步函数共用同一套装饰器——绑定到 ``async def`` 时自动返回协程包装器，
调用方不必为异步场景换一套 API。所有包装器都用 :func:`functools.wraps` 保留
原函数的签名与元数据，参数校验在「装饰时」完成，配置错误不会拖到第一次调用。

本模块不调用 ``logging.basicConfig()``（库不应该改动宿主的日志配置），
日志一律写到 ``logging.getLogger(func.__module__)``，也可以显式传 ``logger``。
只做计时/打印的场景请用 :func:`cs.time.timer` 与
:func:`cs.logger.timed_log`，这里不重复实现。

基本用法::

    from cs.decorator import capture_output, logged, on_failure, retry

    @retry(attempts=3, delay=0.5, backoff=2.0, exceptions=(TimeoutError,))
    def fetch(url: str) -> str: ...

    @logged(log_args=True, log_result=True)
    def train(epochs: int) -> float: ...

    @on_failure(lambda f: print(f.text()), reraise=False)
    def nightly_job() -> None: ...

    @capture_output(path="build.log")
    def build() -> None: ...
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import io
import logging
import random
import threading
import time
import traceback
import warnings
from collections.abc import Callable, Mapping
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

__all__ = [
    "Failure",
    "RetryPolicy",
    "capture_output",
    "deprecated",
    "log_execution",
    "logged",
    "on_failure",
    "retry",
    "save_print_to_log",
]

#: 参数 / 返回值在日志里默认截断到多长
PRINT_LIMIT = 200

_FILE_LOCK = threading.Lock()


# ============================================================
# 内部工具
# ============================================================


def _name_of(target: Callable[..., Any]) -> str:
    """取可用于日志的名字（方法会带上类名）。"""
    return (
        getattr(target, "__qualname__", None)
        or getattr(target, "__name__", None)
        or repr(target)
    )


def _short_name(target: Callable[..., Any]) -> str:
    """取不含 ``<locals>`` 的短名，用于日志文件前缀与邮件标题。"""
    return getattr(target, "__name__", None) or _name_of(target)


def _resolve_logger(
    target: Callable[..., Any], logger: logging.Logger | str | None
) -> logging.Logger:
    if isinstance(logger, logging.Logger):
        return logger
    return logging.getLogger(logger or getattr(target, "__module__", __name__))


def _short(value: Any, limit: int = PRINT_LIMIT) -> str:
    """``repr`` 并截断，避免日志被大对象撑爆。"""
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 - 坏掉的 __repr__ 不该让日志本身失败
        return f"<{type(value).__name__} 无法 repr>"
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _format_args(
    args: tuple[Any, ...], kwargs: Mapping[str, Any], limit: int = PRINT_LIMIT
) -> str:
    parts = [_short(value, limit) for value in args]
    parts.extend(f"{key}={_short(value, limit)}" for key, value in kwargs.items())
    return ", ".join(parts)


def _elapsed(started: float) -> str:
    return f"{(time.perf_counter() - started) * 1000:.3f} ms"


def _check_supported(target: Callable[..., Any]) -> None:
    if not callable(target):
        raise TypeError(f"只能装饰可调用对象，收到 {type(target).__name__}")
    if inspect.isclass(target):
        raise TypeError(f"{_name_of(target)} 是类：本模块的装饰器只接受函数与方法")
    if inspect.isasyncgenfunction(target):
        raise TypeError(
            f"{_name_of(target)} 是异步生成器函数：yield 过程无法被重试/计时包裹"
        )


def _check_handler(handler: Any, *, label: str, allow_async: bool) -> None:
    if handler is None:
        return
    if not callable(handler):
        raise TypeError(f"{label} 必须是可调用对象，收到 {type(handler).__name__}")
    if not allow_async and inspect.iscoroutinefunction(handler):
        raise TypeError(
            f"{label} 是协程函数，但目标函数是同步的；"
            "请把目标函数写成 async def，或换用同步回调"
        )


def _notify_sync(
    handler: Callable[[Failure], Any] | None,
    failure: Failure,
    log: logging.Logger,
    label: str,
) -> None:
    """同步上下文里调用回调；回调自身的异常只记录，不干扰主流程。"""
    if handler is None:
        return
    try:
        handler(failure)
    except Exception:
        log.exception("%s 回调执行失败（已忽略）", label)


async def _notify_async(
    handler: Callable[[Failure], Any] | None,
    failure: Failure,
    log: logging.Logger,
    label: str,
) -> None:
    """异步上下文里调用回调，同步/异步回调都支持。"""
    if handler is None:
        return
    try:
        outcome = handler(failure)
        if inspect.isawaitable(outcome):
            await outcome
    except Exception:
        log.exception("%s 回调执行失败（已忽略）", label)


def _note_giveup(exc: BaseException, attempts: int) -> None:
    """在最终失败的异常上补一条注释，方便定位「已经试过几次」。"""
    if attempts > 1:
        exc.add_note(f"cs.decorator：共尝试 {attempts} 次仍失败，最后一次异常如上。")


def _append_line(path: Path, name: str, text: str) -> None:
    """按 ``[时间][函数名]:内容`` 追加到日志文件，进程内加锁串行写入。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    body = text if text.endswith("\n") else f"{text}\n"
    with _FILE_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}][{name}]:{body}")


# ============================================================
# 失败现场
# ============================================================


@dataclass(frozen=True, slots=True)
class Failure:
    """一次失败的现场快照，交给 ``on_retry`` / ``on_giveup`` / ``on_failure``。

    ``args`` / ``kwargs`` 可能是敏感数据（口令、token），写邮件前请自行评估。
    """

    func: Callable[..., Any]
    exception: BaseException
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    attempt: int = 1
    duration: float = 0.0

    @property
    def name(self) -> str:
        """带类名/外层函数的完整名字，用于日志定位。"""
        return _name_of(self.func)

    @property
    def short_name(self) -> str:
        """不含 ``<locals>`` 的短名，适合日志前缀、通知标题等展示场景。"""
        return _short_name(self.func)

    def text(self, limit: int = 800) -> str:
        """多行描述，可直接作为告警正文或日志内容。"""
        call = _format_args(self.args, self.kwargs, limit)
        lines = [
            f"函数：{self.name}",
            f"调用：{call}",
            f"尝试：第 {self.attempt} 次",
            f"耗时：{self.duration * 1000:.2f} ms",
            f"异常：{type(self.exception).__name__}: {self.exception}",
            "",
            "堆栈：",
            "".join(traceback.format_exception(self.exception)).rstrip(),
        ]
        return "\n".join(lines)


#: 失败回调：接收 :class:`Failure`，返回值可有可无（协程会被 await）
type Handler = Callable[[Failure], Any]


# ============================================================
# 重试
# ============================================================


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """重试策略。

    ``attempts`` 是总尝试次数（含首次调用）。第 ``n`` 次重试前等待
    ``delay * backoff ** (n - 1)`` 秒，上限 ``max_delay``；``jitter`` 在其上
    叠加 ±jitter 比例的随机抖动（0~1），用来错开并发重试。
    """

    attempts: int = 3
    delay: float = 0.0
    backoff: float = 2.0
    max_delay: float = 30.0
    jitter: float = 0.0
    exceptions: tuple[type[BaseException], ...] = (Exception,)

    def __post_init__(self) -> None:
        exceptions = (
            (self.exceptions,) if isinstance(self.exceptions, type) else self.exceptions
        )
        exceptions = tuple(exceptions)
        object.__setattr__(self, "exceptions", exceptions)

        if not exceptions:
            raise ValueError("exceptions 不能为空，否则永远不会重试")
        for item in exceptions:
            if not (isinstance(item, type) and issubclass(item, BaseException)):
                raise TypeError(f"exceptions 只能包含异常类，收到 {item!r}")
        if self.attempts < 1:
            raise ValueError(f"attempts 至少为 1，收到 {self.attempts}")
        if self.delay < 0:
            raise ValueError(f"delay 不能为负，收到 {self.delay}")
        if self.backoff < 1:
            raise ValueError(
                f"backoff 必须 ≥ 1，否则等待时间会越来越短，收到 {self.backoff}"
            )
        if self.max_delay < 0:
            raise ValueError(f"max_delay 不能为负，收到 {self.max_delay}")
        if not 0 <= self.jitter <= 1:
            raise ValueError(f"jitter 取值为 0~1，收到 {self.jitter}")

    def accepts(self, exc: BaseException) -> bool:
        return isinstance(exc, self.exceptions)

    def wait(self, retry: int) -> float:
        """第 ``retry`` 次重试（从 1 开始）前的等待秒数。"""
        if self.delay <= 0:
            return 0.0
        seconds = min(self.delay * self.backoff ** (retry - 1), self.max_delay)
        if self.jitter:
            seconds *= random.uniform(1 - self.jitter, 1 + self.jitter)
        return max(0.0, seconds)


def retry[**P, R](
    func: Callable[P, R] | None = None,
    /,
    *,
    attempts: int = 3,
    delay: float = 0.0,
    backoff: float = 2.0,
    max_delay: float = 30.0,
    jitter: float = 0.0,
    exceptions: tuple[type[BaseException], ...] | type[BaseException] = (Exception,),
    on_retry: Handler | None = None,
    on_giveup: Handler | None = None,
    logger: logging.Logger | str | None = None,
) -> Callable[P, R]:
    """失败后重试，支持 ``@retry`` 与 ``@retry(...)`` 两种写法。

    参数：
        attempts: 总尝试次数（含首次），最少 1。
        delay: 首次重试前的等待秒数，0 表示不等待。
        backoff: 等待时间的倍数，1 表示固定间隔。
        max_delay: 单次等待上限。
        jitter: 抖动比例，避免同一时刻大量重试。
        exceptions: 需要重试的异常，其余异常直接抛出、不浪费次数。
        on_retry: 每次「还要再试」时调用，收到 :class:`Failure`。
        on_giveup: 次数用尽时调用，随后原样抛出最后一次异常。
        logger: 日志器实例或名字，默认用函数所在模块。

    最终失败时抛出的是**最后一次的原始异常**（保留完整 traceback），并附一条
    ``add_note`` 说明尝试次数；不会包装成自定义异常，反而不利于排查。
    """
    policy = RetryPolicy(
        attempts=attempts,
        delay=delay,
        backoff=backoff,
        max_delay=max_delay,
        jitter=jitter,
        exceptions=exceptions,
    )

    def decorate(target: Callable[P, R]) -> Callable[P, R]:
        _check_supported(target)
        is_async = inspect.iscoroutinefunction(target)
        _check_handler(on_retry, label="on_retry", allow_async=is_async)
        _check_handler(on_giveup, label="on_giveup", allow_async=is_async)
        log = _resolve_logger(target, logger)
        name = _name_of(target)

        def record(
            exc: BaseException,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
            attempt: int,
            started: float,
        ) -> Failure:
            return Failure(
                func=target,
                exception=exc,
                args=args,
                kwargs=kwargs,
                attempt=attempt,
                duration=time.perf_counter() - started,
            )

        if is_async:

            @functools.wraps(target)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                attempt = 0
                while True:
                    attempt += 1
                    started = time.perf_counter()
                    try:
                        return await target(*args, **kwargs)
                    except policy.exceptions as exc:
                        failure = record(exc, args, kwargs, attempt, started)
                        if attempt >= policy.attempts:
                            await _notify_async(on_giveup, failure, log, "on_giveup")
                            _note_giveup(exc, policy.attempts)
                            raise
                        wait = policy.wait(attempt)
                        log.warning(
                            "%s 第 %d/%d 次失败（%s: %s），%.3fs 后重试",
                            name,
                            attempt,
                            policy.attempts,
                            type(exc).__name__,
                            exc,
                            wait,
                        )
                        await _notify_async(on_retry, failure, log, "on_retry")
                        await asyncio.sleep(wait)

            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(target)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            attempt = 0
            while True:
                attempt += 1
                started = time.perf_counter()
                try:
                    return target(*args, **kwargs)
                except policy.exceptions as exc:
                    failure = record(exc, args, kwargs, attempt, started)
                    if attempt >= policy.attempts:
                        _notify_sync(on_giveup, failure, log, "on_giveup")
                        _note_giveup(exc, policy.attempts)
                        raise
                    wait = policy.wait(attempt)
                    log.warning(
                        "%s 第 %d/%d 次失败（%s: %s），%.3fs 后重试",
                        name,
                        attempt,
                        policy.attempts,
                        type(exc).__name__,
                        exc,
                        wait,
                    )
                    _notify_sync(on_retry, failure, log, "on_retry")
                    if wait > 0:
                        time.sleep(wait)

        return wrapper

    return decorate(func) if func is not None else decorate


# ============================================================
# 调用日志
# ============================================================


def logged[**P, R](
    func: Callable[P, R] | None = None,
    /,
    *,
    level: int = logging.INFO,
    logger: logging.Logger | str | None = None,
    log_args: bool = False,
    log_result: bool = False,
    log_time: bool = True,
    max_length: int = PRINT_LIMIT,
) -> Callable[P, R]:
    """记录函数的进入、完成与异常，自带耗时，支持同步与异步。

    进入记 DEBUG、完成记 ``level``、异常记 ERROR（带 traceback 后原样抛出），
    因此默认配置下只会看到「完成」与「失败」两类记录。

    参数：
        level: 完成日志的级别，默认 INFO。
        log_args: 是否把入参写进日志（可能含敏感数据，默认关闭）。
        log_result: 是否把返回值写进日志，默认关闭。
        log_time: 是否附带耗时，默认开启。
        max_length: 参数/返回值写日志时的截断长度。
    """
    if not isinstance(level, int):
        raise TypeError(f"level 必须是 logging 的整数常量，收到 {level!r}")

    def decorate(target: Callable[P, R]) -> Callable[P, R]:
        _check_supported(target)
        log = _resolve_logger(target, logger)
        name = _name_of(target)

        def call_of(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> str:
            if not log_args:
                return name
            return f"{name}({_format_args(args, kwargs, max_length)})"

        def done(call: str, started: float, result: Any) -> None:
            timing = f"（{_elapsed(started)}）" if log_time else ""
            if log_result:
                log.log(
                    level, "%s 完成%s -> %s", call, timing, _short(result, max_length)
                )
            else:
                log.log(level, "%s 完成%s", call, timing)

        if inspect.iscoroutinefunction(target):

            @functools.wraps(target)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                call = call_of(args, kwargs)
                log.debug("→ %s", call)
                started = time.perf_counter()
                try:
                    result = await target(*args, **kwargs)
                except Exception:
                    log.exception("%s 失败（%s）", call, _elapsed(started))
                    raise
                done(call, started, result)
                return result

            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(target)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            call = call_of(args, kwargs)
            log.debug("→ %s", call)
            started = time.perf_counter()
            try:
                result = target(*args, **kwargs)
            except Exception:
                log.exception("%s 失败（%s）", call, _elapsed(started))
                raise
            done(call, started, result)
            return result

        return wrapper

    return decorate(func) if func is not None else decorate


#: 旧名，等价于 :func:`logged`
log_execution = logged


# ============================================================
# 失败通知
# ============================================================


def on_failure[**P, R](
    handler: Handler,
    /,
    *,
    reraise: bool = True,
    logger: logging.Logger | str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """函数抛出异常时调用 ``handler(Failure)``，再按 ``reraise`` 决定是否吞掉。

    ``reraise=False`` 时异常被吞掉、函数返回 ``None``——适合守护型任务，
    但要清楚这会让调用方失去失败信号。

    ::

        @on_failure(lambda f: alert(f.text()), reraise=False)
        def cleanup() -> None: ...
    """
    if not callable(handler):
        raise TypeError(f"handler 必须是可调用对象，收到 {type(handler).__name__}")

    def decorate(target: Callable[P, R]) -> Callable[P, R]:
        _check_supported(target)
        is_async = inspect.iscoroutinefunction(target)
        _check_handler(handler, label="handler", allow_async=is_async)
        log = _resolve_logger(target, logger)

        def record(
            exc: BaseException, args: tuple[Any, ...], kwargs: Mapping[str, Any]
        ) -> Failure:
            return Failure(func=target, exception=exc, args=args, kwargs=kwargs)

        if is_async:

            @functools.wraps(target)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                try:
                    return await target(*args, **kwargs)
                except Exception as exc:
                    await _notify_async(
                        handler, record(exc, args, kwargs), log, "on_failure"
                    )
                    if reraise:
                        raise
                    return cast(R, None)

            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(target)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return target(*args, **kwargs)
            except Exception as exc:
                _notify_sync(handler, record(exc, args, kwargs), log, "on_failure")
                if reraise:
                    raise
                return cast(R, None)

        return wrapper

    return decorate


# ============================================================
# 输出捕获
# ============================================================


def capture_output[**P, R](
    func: Callable[P, R] | None = None,
    /,
    *,
    path: str | Path | None = None,
    logger: logging.Logger | str | None = None,
    echo: bool = False,
    also_stderr: bool = False,
    level: int = logging.INFO,
) -> Callable[P, R]:
    """捕获函数执行期间的 ``print`` 输出，写入日志文件或 logging。

    参数：
        path: 追加写入的文件，格式 ``[时间][函数名]:内容``；父目录自动创建。
        logger: 写到日志器；与 ``path`` 至少给一个。
        echo: 是否回显到真实 stdout；回显发生在函数结束后，不是流式输出。
        also_stderr: 是否连 stderr 一起捕获。
        level: 写 logging 时的级别。

    函数抛出异常时，已产生的输出仍会落盘，之后异常原样抛出。

    注意：``redirect_stdout`` 改的是进程级 ``sys.stdout``，多线程并发写同一段
    输出会互相串台；多线程场景请改用 ``logger`` 记录。
    """
    if path is None and logger is None:
        raise ValueError("capture_output 需要 path 或 logger 之一作为输出目标")

    if not isinstance(level, int):
        raise TypeError(f"level 必须是 logging 的整数常量，收到 {level!r}")
    destination = Path(path) if path is not None else None

    def decorate(target: Callable[P, R]) -> Callable[P, R]:
        _check_supported(target)
        log = _resolve_logger(target, logger)
        name = _short_name(target)

        def flush(text: str) -> None:
            if not text:
                return
            if echo:
                print(text, end="" if text.endswith("\n") else "\n")
            if destination is not None:
                _append_line(destination, name, text)
            if logger is not None:
                log.log(level, "%s 输出：\n%s", name, text.rstrip())

        if inspect.iscoroutinefunction(target):

            @functools.wraps(target)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                buffer = io.StringIO()
                stack = ExitStack()
                stack.enter_context(redirect_stdout(buffer))
                if also_stderr:
                    stack.enter_context(redirect_stderr(buffer))
                try:
                    with stack:
                        return await target(*args, **kwargs)
                finally:
                    flush(buffer.getvalue())

            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(target)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            buffer = io.StringIO()
            stack = ExitStack()
            stack.enter_context(redirect_stdout(buffer))
            if also_stderr:
                stack.enter_context(redirect_stderr(buffer))
            try:
                with stack:
                    return target(*args, **kwargs)
            finally:
                flush(buffer.getvalue())

        return wrapper

    return decorate(func) if func is not None else decorate


def save_print_to_log[**P, R](
    func: Callable[P, R] | None = None,
    /,
    *,
    path: str | Path = "log.txt",
    echo: bool = False,
) -> Callable[P, R]:
    """向后兼容的旧名：把 ``print`` 输出追加到 ``path``（默认当前目录 log.txt）。

    与旧实现的区别：用 ``redirect_stdout(io.StringIO())`` 代替「改写 sys.stdout +
    临时文件」，异常时不会留下没关闭的句柄，也不会把临时文件写到进程目录。
    """
    return capture_output(func, path=path, echo=echo)


# ============================================================
# 弃用标记
# ============================================================


def deprecated[**P, R](
    func: Callable[P, R] | None = None,
    /,
    *,
    reason: str = "",
    since: str | None = None,
    removal: str | None = None,
) -> Callable[P, R]:
    """标记函数已弃用：每次调用发出 ``DeprecationWarning``，并写入 ``__deprecated__``。

    ``__deprecated__`` 是 PEP 702 约定的属性，IDE 与静态检查器会据此提示。

    ::

        @deprecated(reason="改用 fetch_v2", since="0.3", removal="0.5")
        def fetch(): ...
    """

    def decorate(target: Callable[P, R]) -> Callable[P, R]:
        _check_supported(target)
        note = f"{_name_of(target)} 已弃用"
        if since:
            note += f"（自 {since} 起）"
        if reason:
            note += f"：{reason}"
        if removal:
            note += f"；计划在 {removal} 移除"

        if inspect.iscoroutinefunction(target):

            @functools.wraps(target)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                warnings.warn(note, DeprecationWarning, stacklevel=2)
                return await target(*args, **kwargs)

            async_wrapper.__deprecated__ = note  # type: ignore[attr-defined]
            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(target)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(note, DeprecationWarning, stacklevel=2)
            return target(*args, **kwargs)

        wrapper.__deprecated__ = note  # type: ignore[attr-defined]
        return wrapper

    return decorate(func) if func is not None else decorate
