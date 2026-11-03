"""``cs.decorator`` 的单元测试：重试策略、调用日志、失败通知、输出捕获、弃用标记。"""

from __future__ import annotations

import asyncio
import inspect
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs.decorator import (
    Failure,
    RetryPolicy,
    capture_output,
    deprecated,
    log_execution,
    logged,
    on_failure,
    retry,
    save_print_to_log,
)

# ============================================================
# retry
# ============================================================


def test_retry_returns_directly_when_success():
    calls = []

    @retry(attempts=3)
    def ok(value: int) -> int:
        calls.append(value)
        return value * 2

    assert ok(3) == 6
    assert calls == [3]


def test_retry_eventually_succeeds():
    counter = {"n": 0}

    @retry(attempts=3, delay=0.0, exceptions=(ValueError,))
    def flaky() -> str:
        counter["n"] += 1
        if counter["n"] < 3:
            raise ValueError("boom")
        return "ok"

    assert flaky() == "ok"
    assert counter["n"] == 3


def test_retry_exhausted_raises_last_exception_with_note():
    errors = []

    @retry(attempts=2, delay=0.0)
    def always() -> None:
        error = RuntimeError("nope")
        errors.append(error)
        raise error

    with pytest.raises(RuntimeError, match="nope") as info:
        always()

    assert info.value is errors[-1], "必须抛出最后一次的原始异常"
    assert len(errors) == 2
    notes = getattr(info.value, "__notes__", [])
    assert any("共尝试 2 次" in note for note in notes)


def test_retry_skips_unlisted_exceptions():
    counter = {"n": 0}

    @retry(attempts=5, exceptions=(ValueError,))
    def other() -> None:
        counter["n"] += 1
        raise TypeError("not retryable")

    with pytest.raises(TypeError):
        other()

    assert counter["n"] == 1


def test_retry_calls_hooks_with_attempt_numbers():
    retried: list[Failure] = []
    gave_up: list[Failure] = []

    @retry(attempts=3, delay=0.0, on_retry=retried.append, on_giveup=gave_up.append)
    def always() -> None:
        raise ValueError("x")

    with pytest.raises(ValueError):
        always()

    assert [failure.attempt for failure in retried] == [1, 2]
    assert [failure.attempt for failure in gave_up] == [3]
    assert gave_up[0].name.endswith("always")
    assert isinstance(gave_up[0].exception, ValueError)
    assert gave_up[0].func.__name__ == "always"


def test_retry_supports_bare_decorator():
    counter = {"n": 0}

    @retry
    def flaky() -> int:
        counter["n"] += 1
        if counter["n"] < 2:
            raise ValueError("once")
        return counter["n"]

    assert flaky() == 2


def test_retry_preserves_metadata():
    @retry(attempts=2)
    def target(a: int, *, b: int = 1) -> int:
        """目标函数文档。"""
        return a + b

    assert target.__name__ == "target"
    assert target.__doc__ == "目标函数文档。"

    signature = inspect.signature(target)
    assert list(signature.parameters) == ["a", "b"]
    assert signature.parameters["b"].default == 1
    assert signature.parameters["b"].kind is inspect.Parameter.KEYWORD_ONLY


def test_retry_rejects_async_generator():
    with pytest.raises(TypeError, match="异步生成器"):

        @retry(attempts=2)
        async def agen():
            yield 1


def test_retry_rejects_classes():
    with pytest.raises(TypeError, match="类"):

        @retry(attempts=2)
        class Thing:
            pass


def test_retry_rejects_async_hook_for_sync_function():
    async def hook(failure: Failure) -> None:  # pragma: no cover - 不应被调用
        raise AssertionError("不该执行")

    with pytest.raises(TypeError, match="协程函数"):

        @retry(attempts=2, on_retry=hook)
        def target() -> None:
            pass


def test_retry_async_target():
    counter = {"n": 0}

    @retry(attempts=3, delay=0.0, exceptions=(ValueError,))
    async def flaky() -> str:
        counter["n"] += 1
        if counter["n"] < 3:
            raise ValueError("boom")
        return "ok"

    assert asyncio.run(flaky()) == "ok"
    assert counter["n"] == 3


def test_retry_async_awaits_async_hook():
    seen: list[int] = []

    async def hook(failure: Failure) -> None:
        await asyncio.sleep(0)
        seen.append(failure.attempt)

    @retry(attempts=2, delay=0.0, on_retry=hook)
    async def flaky() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        asyncio.run(flaky())

    assert seen == [1]


def test_retry_logs_warning_on_each_attempt(caplog):
    @retry(attempts=2, delay=0.0)
    def flaky() -> None:
        raise ValueError("boom")

    with caplog.at_level(logging.WARNING), pytest.raises(ValueError):
        flaky()

    assert "第 1/2 次失败" in caplog.text


# ============================================================
# RetryPolicy
# ============================================================


def test_policy_wait_follows_backoff_and_cap():
    policy = RetryPolicy(attempts=5, delay=0.1, backoff=3.0, max_delay=0.5)
    assert [round(policy.wait(index), 4) for index in (1, 2, 3)] == [0.1, 0.3, 0.5]


def test_policy_wait_without_delay_is_zero():
    assert RetryPolicy(attempts=3).wait(2) == 0.0


def test_policy_jitter_stays_in_range():
    policy = RetryPolicy(delay=1.0, backoff=1.0, jitter=0.5)
    for _ in range(20):
        assert 0.5 <= policy.wait(1) <= 1.5


def test_policy_accepts_single_exception_class():
    policy = RetryPolicy(exceptions=ValueError)
    assert policy.exceptions == (ValueError,)
    assert policy.accepts(ValueError())
    assert not policy.accepts(TypeError())


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"attempts": 0}, ValueError),
        ({"backoff": 0.5}, ValueError),
        ({"jitter": 1.5}, ValueError),
        ({"delay": -1}, ValueError),
        ({"max_delay": -1}, ValueError),
        ({"exceptions": ()}, ValueError),
        ({"exceptions": (ValueError, "boom")}, TypeError),
    ],
)
def test_policy_rejects_invalid_config(kwargs, error):
    with pytest.raises(error):
        RetryPolicy(**kwargs)


# ============================================================
# logged
# ============================================================


def test_logged_reports_completion(caplog):
    @logged()
    def add(a: int, b: int) -> int:
        return a + b

    with caplog.at_level(logging.INFO):
        assert add(1, 2) == 3

    assert "add 完成" in caplog.text
    assert " ms" in caplog.text


def test_logged_keeps_args_hidden_by_default(caplog):
    @logged()
    def secret(password: str) -> None:
        pass

    with caplog.at_level(logging.DEBUG):
        secret("hunter2")

    assert "hunter2" not in caplog.text


def test_logged_can_report_args_result_and_time(caplog):
    @logged(log_args=True, log_result=True, level=logging.WARNING)
    def add(a: int, b: int) -> int:
        return a + b

    with caplog.at_level(logging.DEBUG):
        add(1, 2)

    assert "add(1, 2)" in caplog.text
    assert "-> 3" in caplog.text
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_logged_together_with_time_toggle(caplog):
    @logged(log_time=False)
    def ping() -> str:
        return "pong"

    with caplog.at_level(logging.INFO):
        ping()

    assert " ms" not in caplog.text


def test_logged_logs_exception_and_reraises(caplog):
    @logged()
    def bad() -> None:
        raise KeyError("missing")

    with caplog.at_level(logging.INFO), pytest.raises(KeyError):
        bad()

    assert "bad 失败" in caplog.text
    assert "KeyError" in caplog.text
    assert any(record.exc_info for record in caplog.records)


def test_logged_async_target(caplog):
    @logged(log_result=True)
    async def compute() -> int:
        await asyncio.sleep(0)
        return 42

    with caplog.at_level(logging.INFO):
        assert asyncio.run(compute()) == 42

    assert "compute 完成" in caplog.text


def test_log_execution_is_alias_of_logged():
    assert log_execution is logged


def test_logged_rejects_invalid_level():
    with pytest.raises(TypeError, match="level"):

        @logged(level="INFO")
        def target() -> None:
            pass


# ============================================================
# Failure / on_failure
# ============================================================


def test_failure_text_contains_traceback_and_call():
    def broken(value: int) -> None:
        raise ValueError(f"bad {value}")

    try:
        broken(7)
    except ValueError as exc:
        failure = Failure(
            func=broken, exception=exc, args=(7,), attempt=2, duration=0.5
        )

    text = failure.text()
    assert text.splitlines()[0].endswith("broken")
    assert "调用：7" in text
    assert "第 2 次" in text
    assert "500.00 ms" in text
    assert "Traceback" in text
    assert failure.name.endswith("broken")
    assert failure.short_name == "broken"


def test_on_failure_notifies_and_reraises():
    seen: list[Failure] = []

    @on_failure(seen.append)
    def job() -> None:
        raise ValueError("fail")

    with pytest.raises(ValueError):
        job()

    assert len(seen) == 1
    assert isinstance(seen[0].exception, ValueError)


def test_on_failure_can_swallow_exception():
    seen: list[Failure] = []

    @on_failure(seen.append, reraise=False)
    def job() -> None:
        raise ValueError("fail")

    assert job() is None
    assert len(seen) == 1


def test_on_failure_async_target():
    seen: list[Failure] = []

    @on_failure(seen.append, reraise=False)
    async def job() -> str:
        raise ValueError("fail")

    assert asyncio.run(job()) is None
    assert len(seen) == 1


def test_on_failure_keeps_going_when_handler_breaks(caplog):
    def broken_handler(failure: Failure) -> None:
        raise RuntimeError("handler boom")

    @on_failure(broken_handler)
    def job() -> None:
        raise ValueError("original")

    with caplog.at_level(logging.INFO), pytest.raises(ValueError, match="original"):
        job()

    assert "on_failure 回调执行失败" in caplog.text


def test_on_failure_requires_callable():
    with pytest.raises(TypeError, match="handler"):
        on_failure("not-callable")


# ============================================================
# capture_output
# ============================================================


def test_capture_output_writes_file(tmp_path: Path):
    log_file = tmp_path / "logs" / "build.log"

    @capture_output(path=log_file)
    def build() -> str:
        print("line 1")
        print("line 2")
        return "built"

    assert build() == "built"

    content = log_file.read_text(encoding="utf-8")
    assert "[build]:line 1" in content
    assert "line 2" in content
    assert content.startswith("[")


def test_capture_output_appends(tmp_path: Path):
    log_file = tmp_path / "run.log"

    @capture_output(path=log_file)
    def once(tag: str) -> None:
        print(tag)

    once("first")
    once("second")

    assert log_file.read_text(encoding="utf-8").count("]:") == 2


def test_capture_output_echo(tmp_path: Path, capsys):
    @capture_output(path=tmp_path / "x.log", echo=True)
    def talk() -> None:
        print("hello")

    talk()

    assert "hello" in capsys.readouterr().out


def test_capture_output_captures_stderr(tmp_path: Path):
    log_file = tmp_path / "err.log"

    @capture_output(path=log_file, also_stderr=True)
    def noisy() -> None:
        print("out")
        print("err", file=sys.stderr)

    noisy()

    content = log_file.read_text(encoding="utf-8")
    assert "out" in content
    assert "err" in content


def test_capture_output_flushes_partial_output_on_error(tmp_path: Path):
    log_file = tmp_path / "boom.log"

    @capture_output(path=log_file)
    def boom() -> None:
        print("partial")
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        boom()

    assert "partial" in log_file.read_text(encoding="utf-8")


def test_capture_output_writes_to_logger(tmp_path: Path, caplog):
    @capture_output(logger="cs.test.capture")
    def run() -> None:
        print("via logger")

    with caplog.at_level(logging.INFO):
        run()

    assert "via logger" in caplog.text


def test_capture_output_requires_a_sink():
    with pytest.raises(ValueError, match="path 或 logger"):

        @capture_output()
        def target() -> None:
            pass


def test_capture_output_async(tmp_path: Path):
    log_file = tmp_path / "async.log"

    @capture_output(path=log_file)
    async def run() -> int:
        print("async line")
        await asyncio.sleep(0)
        return 1

    assert asyncio.run(run()) == 1
    assert "async line" in log_file.read_text(encoding="utf-8")


def test_save_print_to_log_default_path(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @save_print_to_log
    def legacy() -> None:
        print("legacy output")

    legacy()

    assert "legacy output" in (tmp_path / "log.txt").read_text(encoding="utf-8")


def test_capture_output_restores_stdout_after_error(tmp_path: Path):
    original = sys.stdout

    @capture_output(path=tmp_path / "boom.log")
    def boom() -> None:
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        boom()

    assert sys.stdout is original


# ============================================================
# deprecated
# ============================================================


def test_deprecated_warns_and_sets_attribute():
    @deprecated(reason="改用 fetch_v2", since="0.3", removal="0.5")
    def fetch() -> str:
        return "v1"

    with pytest.warns(DeprecationWarning, match="fetch 已弃用"):
        assert fetch() == "v1"

    assert "0.5" in fetch.__deprecated__


def test_deprecated_supports_bare_decorator():
    @deprecated
    def ping() -> None:
        pass

    with pytest.warns(DeprecationWarning, match="ping 已弃用"):
        ping()


def test_deprecated_async_target():
    @deprecated(reason="换成 v2")
    async def fetch() -> str:
        return "v1"

    with pytest.warns(DeprecationWarning):
        assert asyncio.run(fetch()) == "v1"


def test_deprecated_keeps_metadata():
    @deprecated(reason="x")
    def fetch(a: int) -> int:
        """doc."""
        return a

    assert fetch.__name__ == "fetch"
    assert fetch.__doc__ == "doc."
