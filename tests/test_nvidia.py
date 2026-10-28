import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs.nvidia import (
    NVIDIA,
    ClockReading,
    Clocks,
    GPUError,
    GPUNotFoundError,
    GPUProcess,
    GPUStatus,
    HealthReport,
    MemoryUsage,
    PciLink,
    PowerInfo,
    ProcessKind,
    PState,
    Severity,
    TemperatureLimits,
    ThrottleInfo,
    ThrottleReason,
    bytes_to,
    cuda_version_str,
    render_table,
    sample_status,
)
from cs.nvidia.__main__ import main
from cs.nvidia.monitor import SampleSeries, iter_status
from cs.nvidia.readers import diagnose

HAS_GPU = NVIDIA.available()
requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="需要可用的 NVIDIA GPU")


def _status(**overrides: object) -> GPUStatus:
    """构造一个可控的状态快照用于纯逻辑测试。"""
    base = {
        "index": 0,
        "uuid": "GPU-test",
        "name": "Test GPU",
        "timestamp": 1000.0,
        "temperature": 45.0,
        "power": PowerInfo(watts=10.0, limit_watts=100.0),
    }
    return GPUStatus(**{**base, **overrides})  # type: ignore[arg-type]


# ---------------------------------------------------------------- 换算与枚举


def test_bytes_and_cuda_helpers() -> None:
    assert bytes_to(1024**3) == 1.0
    assert bytes_to(1536 * 1024**2, "MiB") == 1536.0
    assert bytes_to(None) is None
    assert cuda_version_str(13040) == "13.4"
    assert cuda_version_str(12000) == "12.0"
    assert cuda_version_str(None) is None


@pytest.mark.parametrize(
    ("bits", "names"),
    [
        (0x0, ()),
        (0x1, ("GPU_IDLE",)),
        (0x4, ("SW_POWER_CAP",)),
        (0x400, ("RELIABILITY",)),
        (0x8 | 0x40, ("HW_SLOWDOWN", "HW_THERMAL_SLOWDOWN")),
    ],
)
def test_throttle_reason_bits(bits: int, names: tuple[str, ...]) -> None:
    reason = ThrottleReason.from_bits(bits)
    assert reason.names == names


def test_throttle_reason_ignores_unknown_bits() -> None:
    # 1 << 20 是驱动里尚未定义的新位，不应导致异常
    reason = ThrottleReason.from_bits(0x1 | (1 << 20))
    assert reason.names == ("GPU_IDLE",)
    assert reason & ThrottleReason.GPU_IDLE


def test_throttle_info_throttled_flag() -> None:
    idle_only = ThrottleInfo(
        bits=0x1, active=ThrottleReason.GPU_IDLE, supported=ThrottleReason(0x7FF)
    )
    assert idle_only.throttled is False
    capped = ThrottleInfo(
        bits=0x4, active=ThrottleReason.SW_POWER_CAP, supported=ThrottleReason(0x7FF)
    )
    assert capped.throttled is True
    assert capped.to_dict()["active"] == ["SW_POWER_CAP"]


def test_pstate_str() -> None:
    assert str(PState.P0) == "P0"
    assert str(PState.UNKNOWN) == "P?"
    assert PState.P8 > PState.P0


# ---------------------------------------------------------------- 值对象语义


def test_memory_usage_percent_and_dict() -> None:
    memory = MemoryUsage(total=8 * 1024**3, used=2 * 1024**3, free=6 * 1024**3)
    assert memory.percent == 25.0
    payload = memory.to_dict()
    assert payload["used"] == 2.0
    assert payload["percent"] == 25.0
    assert "2.0/8.0 GiB" in str(memory)


def test_power_and_clock_percent() -> None:
    assert PowerInfo(watts=50, limit_watts=200).percent == 25.0
    assert PowerInfo(watts=50).percent is None
    assert PowerInfo().percent is None
    assert ClockReading(current=1000, max=2000).percent == 50.0
    assert ClockReading().percent is None


def test_pci_degraded() -> None:
    assert (
        PciLink(generation=4, width=8, max_generation=4, max_width=16).degraded is True
    )
    assert (
        PciLink(generation=4, width=16, max_generation=4, max_width=16).degraded
        is False
    )
    assert PciLink().degraded is False


def test_status_idle_flags() -> None:
    idle = _status(
        utilization=_util(2), pstate=PState.P8, memory=MemoryUsage(100, 10, 90)
    )
    assert idle.is_idle is True
    busy = _status(utilization=_util(80), pstate=PState.P0)
    assert busy.is_idle is False


def _util(gpu: int):
    from cs.nvidia import Utilization

    return Utilization(gpu=gpu)


# ---------------------------------------------------------------- 健康检查


def test_diagnose_clean_status() -> None:
    status = _status(
        limits=TemperatureLimits(slowdown=95, shutdown=100),
        memory=MemoryUsage(100, 10, 90),
    )
    assert diagnose(status) == ()


def test_diagnose_high_temperature() -> None:
    status = _status(temperature=99.0, limits=TemperatureLimits(slowdown=95))
    issues = diagnose(status)
    assert [issue.code for issue in issues] == ["temperature.slowdown"]
    assert issues[0].level is Severity.CRITICAL


def test_diagnose_memory_pressure_and_throttle() -> None:
    status = _status(
        memory=MemoryUsage(total=100, used=92, free=8),
        throttle=ThrottleInfo(
            bits=0x8,
            active=ThrottleReason.HW_SLOWDOWN,
            supported=ThrottleReason(0xFF),
        ),
    )
    issues = {issue.code: issue for issue in diagnose(status)}
    assert issues["memory.pressure"].level is Severity.WARNING
    assert issues["clock.throttle"].level is Severity.WARNING


def test_diagnose_sw_power_cap_is_info() -> None:
    status = _status(
        throttle=ThrottleInfo(
            bits=0x4,
            active=ThrottleReason.SW_POWER_CAP,
            supported=ThrottleReason(0xFF),
        )
    )
    issues = diagnose(status)
    assert issues[0].level is Severity.INFO


def test_health_report_worst() -> None:
    from cs.nvidia import HealthIssue

    assert HealthReport.worst([]) is Severity.INFO
    assert (
        HealthReport.worst(
            [
                HealthIssue(Severity.INFO, "a", "a"),
                HealthIssue(Severity.CRITICAL, "c", "c"),
                HealthIssue(Severity.WARNING, "b", "b"),
            ]
        )
        is Severity.CRITICAL
    )


# ---------------------------------------------------------------- 采样


def test_iter_status_requires_bound() -> None:
    class Fake:
        def status(self) -> GPUStatus:
            return _status()

    with pytest.raises(ValueError, match="至少提供一个"):
        list(iter_status(Fake(), interval=0.1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="interval"):
        list(iter_status(Fake(), interval=0, count=1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="count"):
        list(iter_status(Fake(), count=0, duration=1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duration"):
        list(iter_status(Fake(), count=1, duration=-1))  # type: ignore[arg-type]


def test_iter_status_count_and_duration() -> None:
    class Fake:
        calls = 0

        def status(self) -> GPUStatus:
            Fake.calls += 1
            return _status()

    states = list(iter_status(Fake(), interval=0.01, count=3))  # type: ignore[arg-type]
    assert len(states) == 3
    assert Fake.calls == 3

    by_time = list(iter_status(Fake(), interval=0.01, duration=0.05))  # type: ignore[arg-type]
    assert 1 <= len(by_time) <= 10


def test_sample_series_stats_and_csv(tmp_path: Path) -> None:
    states = tuple(
        _status(
            timestamp=1000.0 + index,
            temperature=40.0 + index,
            utilization=_util(10 * index),
            memory=MemoryUsage(total=1024**3, used=index * 1024**2, free=0),
        )
        for index in range(3)
    )
    series = SampleSeries(
        index=0, uuid="GPU-test", name="Test GPU", interval=1.0, states=states
    )
    assert series.count == 3
    assert series.window == 2.0
    stats = series.stats()
    assert stats["temperature"] == {"min": 40.0, "max": 42.0, "avg": 41.0}
    assert stats["gpu_util"]["max"] == 20.0
    assert stats["mem_used"]["max"] == 2.0  # MiB，已换算

    target = series.to_csv(tmp_path / "gpu.csv")
    lines = target.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0].startswith("timestamp,iso_time,gpu_index")
    assert len(lines) == 4
    assert "40.0" in lines[1]

    payload = series.to_dict(include_states=True)
    assert payload["count"] == 3
    assert len(payload["states"]) == 3
    assert "采样 3 次" in str(series)


def test_sample_status_needs_handle_like() -> None:
    class Fake:
        index = 0
        uuid = "GPU-test"
        name = "Test GPU"

        def status(self) -> GPUStatus:
            return _status()

    series = sample_status(Fake(), interval=0.01, count=2)  # type: ignore[arg-type]
    assert series.count == 2


# ---------------------------------------------------------------- 呈现


def test_render_table() -> None:
    table = render_table(
        [
            _status(
                temperature=60.0,
                utilization=_util(42),
                memory=MemoryUsage(8 * 1024**3, 2 * 1024**3, 6 * 1024**3),
                pstate=PState.P2,
                clocks=Clocks(sm=ClockReading(1500, 3000)),
            )
        ]
    )
    assert "P2" in table
    assert "42%" in table
    assert "Test GPU" in table
    assert "60°C" in table
    assert render_table([]) == "没有可显示的 GPU"


def test_cli_rejects_bad_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise GPUNotFoundError("nope")

    monkeypatch.setattr("cs.nvidia.core._resolve", boom)
    assert main(["status", "9"]) == 1


def test_public_exports_are_importable() -> None:
    import cs.nvidia as module

    for name in module.__all__:
        assert getattr(module, name, None) is not None, name


def test_lazy_import_does_not_pull_optional_deps() -> None:
    code = (
        "import sys; import cs.nvidia; "
        "print('pynvml' in sys.modules, 'fastmcp' in sys.modules)"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False False"


# ---------------------------------------------------------------- 绑定层容错


class NVMLError_NotSupported(Exception):
    """模拟 pynvml 动态生成的异常类（按类名匹配）。"""


class NVMLError_GpuIsLost(Exception):
    pass


class NVMLError_DriverNotLoaded(Exception):
    pass


class NVMLError_NotFound(Exception):
    pass


class NVMLError_Unknown(Exception):
    pass


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (NVMLError_NotSupported("x"), "GPUNotSupportedError"),
        (NVMLError_NotFound("x"), "GPUNotFoundError"),
        (NVMLError_DriverNotLoaded("x"), "GPUUnavailableError"),
        (NVMLError_GpuIsLost("x"), "GPULostError"),
        (NVMLError_Unknown("x"), "GPUError"),
    ],
)
def test_translate_nvml_errors(error: Exception, expected: str) -> None:
    from cs.nvidia import bindings

    assert type(bindings._translate(error, "读取指标")).__name__ == expected


def test_optional_swallows_unsupported_but_not_lost() -> None:
    from cs.nvidia import GPULostError, bindings

    def unsupported() -> int:
        raise NVMLError_NotSupported("nope")

    assert bindings.optional(unsupported, default=None) is None
    assert bindings.optional(unsupported, default=7) == 7

    def lost() -> int:
        raise NVMLError_GpuIsLost("gone")

    with pytest.raises(GPULostError):
        bindings.optional(lost)


def test_as_text_handles_bytes() -> None:
    from cs.nvidia.bindings import as_text

    assert as_text(b"NVIDIA GeForce") == "NVIDIA GeForce"
    assert as_text("  x  ") == "x"
    assert as_text(b"") is None
    assert as_text(None) is None


def test_available_without_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """驱动缺失时 available() 返回 False，而不是抛异常。"""
    from cs.nvidia import GPUUnavailableError, bindings

    def broken() -> int:
        raise GPUUnavailableError("Driver Not Loaded")

    monkeypatch.setattr(bindings, "device_count", broken)
    assert NVIDIA.available() is False


# ---------------------------------------------------------------- 真实 GPU


@requires_gpu
def test_driver_and_list() -> None:
    driver = NVIDIA.driver()
    assert driver.gpu_count >= 1
    assert driver.driver_version
    assert driver.nvml_version

    infos = NVIDIA.list()
    assert len(infos) == driver.gpu_count
    first = infos[0]
    assert first.name and "NVIDIA" in first.name
    assert first.uuid.startswith("GPU-")
    assert first.memory_total and first.memory_total > 0
    assert first.architecture is not None


@requires_gpu
def test_status_snapshot() -> None:
    status = NVIDIA.status(0)
    assert status.uuid.startswith("GPU-")
    assert status.memory is not None
    assert status.memory.total > 0
    assert 0 <= status.memory.percent <= 100
    assert status.temperature is None or status.temperature > -40
    assert isinstance(status.pstate, PState)
    payload = status.to_dict()
    assert payload["uuid"] == status.uuid
    assert "throttle" in payload and "pci" in payload


@requires_gpu
def test_selector_variants() -> None:
    by_index = NVIDIA.gpu(0)
    by_uuid = NVIDIA.gpu(by_index.uuid)
    by_bus = NVIDIA.gpu(by_index.info().pci_bus_id or "pci:" + by_index.uuid)
    assert by_index.uuid == by_uuid.uuid == by_bus.uuid
    assert NVIDIA.gpu(str(0)).uuid == by_index.uuid


@requires_gpu
def test_selector_missing_gpu() -> None:
    with pytest.raises(GPUError):
        NVIDIA.gpu(99)
    with pytest.raises(GPUNotFoundError):
        NVIDIA.gpu(3.5)  # type: ignore[arg-type]


@requires_gpu
def test_processes_and_throttle() -> None:
    processes = NVIDIA.processes(0)
    assert isinstance(processes, list)
    for process in processes:
        assert isinstance(process, GPUProcess)
        assert process.pid > 0
        assert process.kind in ProcessKind

    throttle = NVIDIA.throttle(0)
    assert isinstance(throttle.bits, int)
    assert throttle.bits >= 0
    assert isinstance(throttle.active, ThrottleReason)


@requires_gpu
def test_health_and_topology() -> None:
    report = NVIDIA.health(0)
    assert report.level in Severity
    assert report.uuid.startswith("GPU-")
    assert isinstance(report.issues, tuple)

    matrix = NVIDIA.topologies()
    count = NVIDIA.count()
    assert len(matrix.levels) == count
    assert all(len(row) == count for row in matrix.levels)
    assert matrix.levels[0][0].value == "self"


@requires_gpu
def test_watch_and_export(tmp_path: Path) -> None:
    states = list(NVIDIA.watch(0, interval=0.1, count=2))
    assert len(states) == 2
    series = NVIDIA.sample(0, interval=0.1, count=2)
    assert series.count == 2
    target = series.to_csv(tmp_path / "out.csv")
    assert target.exists() and target.stat().st_size > 0


@requires_gpu
def test_cli_status_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gpus"]
    assert payload["gpus"][0]["uuid"].startswith("GPU-")


# ---------------------------------------------------------------- MCP


def test_mcp_tools_registered() -> None:
    pytest.importorskip("fastmcp")
    from fastmcp import Client

    from cs.nvidia.mcp_server import create_server

    server = create_server()

    async def collect() -> list[str]:
        async with Client(server) as client:
            return sorted(tool.name for tool in await client.list_tools())

    assert asyncio.run(collect()) == [
        "nvidia_driver",
        "nvidia_export_csv",
        "nvidia_health",
        "nvidia_info",
        "nvidia_list",
        "nvidia_processes",
        "nvidia_status",
        "nvidia_text",
        "nvidia_throttle",
        "nvidia_topology",
        "nvidia_watch",
    ]


@requires_gpu
def test_mcp_call_status_and_error() -> None:
    pytest.importorskip("fastmcp")
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    from cs.nvidia.mcp_server import create_server

    server = create_server()

    async def run() -> tuple[dict, str]:
        async with Client(server) as client:
            result = await client.call_tool("nvidia_status", {"gpu": "all"})
            payload = result.structured_content or json.loads(result.content[0].text)
            try:
                await client.call_tool("nvidia_watch", {"count": 0})
            except ToolError as exc:
                return payload, str(exc)
            raise AssertionError("应当抛出 ToolError")

    payload, message = asyncio.run(run())
    assert payload["ok"] is True
    assert payload["count"] >= 1
    assert payload["gpus"][0]["memory"]["total_bytes"] > 0
    assert "ValueError" in message
