"""``cs.env`` 的单元测试。

覆盖：数据模型与 JSON 化、平台/硬件探测、命令安全执行、进程操作、
环境变量作用域、MCP 工具清单与调用。
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastmcp import Client

from cs.env import (
    AVX,
    Command,
    CommandError,
    CommandNotFoundError,
    Env,
    MemoryInfo,
    PlatformInfo,
    ProcessError,
    UnsafeCommandError,
    human_bytes,
    jsonable,
    run,
)
from cs.env.environ import is_sensitive, mask, patched, read, scoped
from cs.env.mcp_server import create_server
from cs.env.models import ProcessInfo
from cs.env.shell import split

PY = sys.executable


def _python(code: str) -> list[str]:
    return [PY, "-c", code]


# ---------------------------------------------------------------- 模型


def test_jsonable_handles_enum_path_and_nested():
    payload = jsonable(
        {
            "kind": Env.platform().kind,
            "path": Path("a/b"),
            "pair": (1, 2),
            "nested": MemoryInfo(1, 1, 0, 0, 0.0),
        }
    )
    assert payload["kind"] == Env.platform().kind.value
    assert payload["path"] in {"a/b", "a\\b"}
    assert payload["pair"] == [1, 2]
    assert payload["nested"]["total"] == 1


def test_human_bytes():
    assert human_bytes(0) == "0 B"
    assert human_bytes(1536) == "1.5 KB"
    assert human_bytes(1024**3) == "1.0 GB"


def test_avx_parse_is_lenient():
    assert AVX.parse("AVX512F") is AVX.AVX512
    assert AVX.parse("avx_2") is AVX.AVX2
    assert AVX.parse("sse4") is AVX.BASIC
    assert AVX.parse(None) is AVX.BASIC


def test_memory_to_dict_has_gb():
    info = MemoryInfo(total=8 * 1024**3, available=4, used=4, free=0, percent=50.0)
    assert info.to_dict()["total_gb"] == 8.0
    assert info.to_dict()["total"] == 8 * 1024**3


def test_process_dict_has_readable_time():
    item = ProcessInfo(
        pid=1,
        ppid=0,
        name="init",
        username="root",
        status="sleeping",
        cpu_percent=0.0,
        memory_rss=1024,
        memory_percent=0.1,
        create_time=1700000000.0,
        cmdline="/sbin/init",
    )
    payload = item.to_dict()
    assert payload["started"].startswith("2023-")
    assert payload["age"] > 0
    assert "init" in str(item)


# ---------------------------------------------------------------- 平台


def test_platform_info_is_cached_and_complete():
    first = Env.platform()
    assert first is Env.platform()  # @cache 复用同一个不可变对象
    assert isinstance(first, PlatformInfo)
    assert first.python_version.startswith("3.")
    assert Path(first.python_executable).exists()
    assert first.distro and first.machine


def test_cpu_memory_and_disks_are_sane():
    cpu = Env.cpu(interval=0)
    assert cpu.logical_cores >= 1
    assert 0 <= cpu.percent <= 100
    assert cpu.model

    memory = Env.memory()
    assert memory.total > 0
    assert 0 <= memory.percent <= 100
    assert memory.available <= memory.total

    disks = Env.disks()
    assert disks
    assert all(0 <= disk.percent <= 100 for disk in disks)
    assert all(disk.total > 0 for disk in disks)


def test_disk_usage_of_cwd():
    info = Env.disk_usage(".")
    assert info.total > 0
    assert 0 <= info.percent <= 100


def test_disk_io_samples_throughput():
    throughput = Env.disk_io(0.1)
    assert throughput.seconds == 0.1
    assert throughput.read_bytes >= 0
    assert throughput.read_rate >= 0


def test_gpu_runtime_never_raises():
    runtime = Env.gpus()
    assert isinstance(runtime.available, bool)
    assert runtime.count == len(runtime.devices)
    if not runtime.available:
        assert "未检测到" in str(runtime)


def test_snapshot_report_contains_sections():
    snapshot = Env.snapshot(top=2, interval=0)
    report = snapshot.report(top=2)
    assert "系统快照" in report
    assert "内存" in report
    assert len(snapshot.processes) <= 2


# ---------------------------------------------------------------- 命令


def test_split_handles_quotes_and_keeps_backslashes():
    assert split("git status --short") == ("git", "status", "--short")
    assert split('"a b/c.exe" --flag') == ("a b/c.exe", "--flag")
    assert split('python -c "print(1)"') == ("python", "-c", "print(1)")


@pytest.mark.parametrize("text", ["a | b", "a && b", "echo x > f", "a; b", "a $HOME"])
def test_split_rejects_shell_metacharacters(text):
    with pytest.raises(UnsafeCommandError, match="元字符"):
        split(text)


def test_run_list_and_string_forms():
    assert run(_python("print('hi')")).text == "hi"
    assert run(["git", "--version"]).ok
    assert run(f'"{PY}" -c "print(42)"').text == "42"


def test_run_nonzero_and_check():
    result = run(_python("import sys; sys.exit(3)"))
    assert not result.ok
    assert result.returncode == 3
    with pytest.raises(CommandError, match="退出码 3"):
        run(Command(_python("import sys; sys.exit(3)"), check=True))


def test_run_missing_executable():
    with pytest.raises(CommandNotFoundError, match="找不到可执行文件"):
        run(["cs-env-definitely-missing-binary"])


def test_run_timeout_returns_result_not_exception():
    result = run(_python("import time; time.sleep(5)"), timeout=0.3)
    assert result.timed_out
    assert result.returncode == 124
    assert not result.ok
    assert "未结束" in result.stderr


def test_run_stdin_replaces_pipes():
    result = run(_python("import sys; print(sys.stdin.read().upper())"), stdin="abc")
    assert result.text == "ABC"


def test_run_truncates_long_output():
    result = run(_python("print('x' * 5000)"), max_output=100)
    assert result.truncated
    assert len(result.stdout) < 400


def test_run_all_stops_on_error():
    results = Env.run_all(
        _python("print('a')"), _python("import sys; sys.exit(1)"), _python("print('c')")
    )
    assert [item.returncode for item in results] == [0, 1]
    assert len(Env.run_all(_python("print('a')"), _python("print('b')"))) == 2


def test_which():
    assert Env.which("python") is not None or Env.which("python3") is not None
    assert Env.which("cs-env-definitely-missing") is None


# ---------------------------------------------------------------- 进程


def test_processes_limit_sort_and_filter():
    items = Env.processes(limit=5, sort="memory")
    assert 0 < len(items) <= 5
    assert items == sorted(items, key=lambda item: item.memory_rss, reverse=True)

    current = Env.process(Env.pid())
    assert current.pid == os.getpid()
    assert current.name

    named = Env.processes(name=current.name, limit=5)
    assert named and all(current.name.lower() in item.name.lower() for item in named)


def test_process_missing_pid():
    from cs.env.errors import ProcessNotFoundError

    with pytest.raises(ProcessNotFoundError):
        Env.process(999_999_999)


@pytest.mark.parametrize("pid", [0, -1])
def test_kill_rejects_invalid_pid(pid):
    with pytest.raises(ProcessError, match="非法"):
        Env.kill(pid)


def test_kill_rejects_self_and_parent():
    with pytest.raises(ProcessError, match="当前进程"):
        Env.kill(os.getpid())
    with pytest.raises(ProcessError, match="父进程"):
        Env.kill(os.getppid())


def test_kill_terminates_child():
    child = subprocess.Popen(_python("import time; time.sleep(30)"))
    try:
        assert Env.kill(child.pid, force=True, timeout=5)
        child.wait(timeout=5)
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()


# ---------------------------------------------------------------- 环境变量


def test_env_get_set_drop():
    name = "CS_ENV_UNIT_TEST"
    assert Env.getenv(name) is None
    Env.setenv(name, "v1")
    assert Env.getenv(name) == "v1"
    assert Env.unsetenv(name)
    assert Env.getenv(name) is None
    assert not Env.unsetenv(name)


def test_patched_restores_exactly():
    name = "CS_ENV_PATCHED"
    os.environ.pop(name, None)
    with patched({name: "temp", "CS_ENV_MISSING": None}):
        assert os.environ[name] == "temp"
        assert "CS_ENV_MISSING" not in os.environ
    assert name not in os.environ


def test_scoped_none_is_noop():
    name = "CS_ENV_SCOPED"
    os.environ[name] = "keep"
    try:
        with scoped(name, None):
            assert os.environ[name] == "keep"
    finally:
        os.environ.pop(name, None)


def test_sensitive_masking():
    assert is_sensitive("GITHUB_TOKEN")
    assert is_sensitive("DB_PASSWORD")
    assert not is_sensitive("HOME")
    assert mask("secret") == "***"
    assert mask("") == ""
    assert read(mapping={"API_KEY": "abc", "HOME": "/root"})["API_KEY"] == "***"
    assert read(mapping={"API_KEY": "abc"}, reveal=True)["API_KEY"] == "abc"


# ---------------------------------------------------------------- MCP


def _call(tool: str, payload: dict) -> dict:
    async def go() -> dict:
        async with Client(create_server()) as client:
            result = await client.call_tool(tool, payload)
            return result.data

    return asyncio.run(go())


def test_mcp_exposes_expected_tools():
    async def go() -> set[str]:
        async with Client(create_server()) as client:
            return {tool.name for tool in await client.list_tools()}

    names = asyncio.run(go())
    assert {
        "env_platform",
        "env_cpu",
        "env_memory",
        "env_disks",
        "env_disk_usage",
        "env_gpus",
        "env_network",
        "env_disk_io",
        "env_snapshot",
        "env_processes",
        "env_process",
        "env_kill",
        "env_run",
        "env_which",
        "env_vars",
        "env_get",
    } <= names


def test_mcp_memory_and_platform():
    memory = _call("env_memory", {})
    assert memory["ok"] is True
    assert memory["total"] > 0
    assert memory["total_gb"] > 0

    platform = _call("env_platform", {})
    assert platform["ok"] is True
    assert platform["distro"]


def test_mcp_run_and_rejection():
    result = _call("env_run", {"command": PY, "args": ["-c", "print('mcp')"]})
    assert result["ok"] is True
    assert result["stdout"].strip() == "mcp"

    with pytest.raises(Exception, match="UnsafeCommandError"):
        _call("env_run", {"command": "echo hi > f"})


def test_mcp_processes_and_vars():
    payload = _call("env_processes", {"limit": 3})
    assert payload["ok"] is True
    assert 0 < payload["count"] <= 3

    os.environ["CS_ENV_MCP_SECRET"] = "topsecret"
    try:
        masked = _call("env_get", {"name": "CS_ENV_MCP_SECRET"})
        assert masked["value"] == "***"
        assert masked["exists"] is True
    finally:
        os.environ.pop("CS_ENV_MCP_SECRET", None)


def test_mcp_kill_rejects_self():
    with pytest.raises(Exception, match="当前进程"):
        _call("env_kill", {"pid": os.getpid()})
