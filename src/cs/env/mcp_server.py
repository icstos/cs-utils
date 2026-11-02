"""把 :mod:`cs.env` 暴露为 MCP 工具（FastMCP ≥ 4）。

启动方式::

    python -m cs.env serve                  # stdio，供 MCP 客户端拉起
    python -m cs.env serve --transport http --port 8000
    fastmcp run src/cs/env/mcp_server.py:mcp

约定：工具名统一 ``env_*``；只读工具标注 ``readOnlyHint``，结束进程标注
``destructiveHint``；业务异常统一转 ``ToolError``，不把 traceback 丢给模型。
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from .core import Env
from .environ import is_sensitive, mask
from .errors import EnvError
from .shell import Command

__all__ = ["create_server", "mcp"]

_SERVER_NAME = "cs-env"
_INSTRUCTIONS = """\
本服务器提供本机（运行 MCP 服务的机器）的操作系统级信息与操作：平台、CPU、
内存、磁盘、GPU、网络、进程，以及安全的命令执行与环境变量读取。

使用建议：
1. 先 env_platform / env_snapshot 了解环境，再决定后续动作；
2. 命令执行请把可执行文件与参数分开传（command + args），不要拼 shell 管道；
   需要管道时，把上一条的 stdout 作为下一条的 stdin；
3. 长命令显式给 timeout（秒）；超时不算错误，返回 timed_out=true；
4. env_kill 不可恢复，须先确认 pid 与进程名；
5. 环境变量中密码/令牌类字段默认返回 ***，确需真实值再传 reveal=true。
"""


def _guard[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """把领域异常转换成 MCP 可读错误。"""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except (EnvError, OSError, ValueError) as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


def _done(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def create_server(
    name: str = _SERVER_NAME, *, instructions: str = _INSTRUCTIONS
) -> FastMCP:
    """构造一个独立的服务器实例（便于测试或多实例部署）。"""
    server: FastMCP = FastMCP(name, instructions=instructions)

    # ---------------------------------------------------------- 平台 / 硬件
    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def env_platform() -> dict[str, Any]:
        """读取操作系统、发行版、架构与 Python 运行时信息。"""
        return _done(**Env.platform().to_dict())

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def env_cpu(interval: float = 0.1, per_core: bool = False) -> dict[str, Any]:
        """读取 CPU 型号、核心数、频率、指令集与占用率。

        interval 为采样窗口（秒），传 0 表示不阻塞但首次结果无意义；
        per_core=true 时额外返回每个逻辑核的占用率。
        """
        return _done(**Env.cpu(interval=interval, per_core=per_core).to_dict())

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def env_memory() -> dict[str, Any]:
        """读取物理内存与交换分区（字节 + GB）。"""
        return _done(**Env.memory().to_dict())

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def env_disks(all: bool = False) -> dict[str, Any]:
        """列出所有挂载点的容量与占用；all=true 时包含虚拟/临时文件系统。"""
        items = [disk.to_dict() for disk in Env.disks(all=all)]
        return _done(count=len(items), items=items)

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def env_disk_usage(path: str = ".") -> dict[str, Any]:
        """查看某个路径所在分区的容量与占用。"""
        return _done(**Env.disk_usage(path).to_dict())

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def env_gpus() -> dict[str, Any]:
        """读取 GPU 驱动、CUDA 版本与逐卡显存/利用率/温度/功耗。"""
        runtime = Env.gpus()
        return _done(**runtime.to_dict(), text=str(runtime))

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def env_network() -> dict[str, Any]:
        """读取网络累计收发量与在线网卡。"""
        return _done(**Env.network().to_dict())

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def env_disk_io(interval: float = 1.0) -> dict[str, Any]:
        """采样磁盘读写速率（会阻塞 interval 秒），返回累计量与字节/秒。"""
        throughput = Env.disk_io(interval)
        return _done(**throughput.to_dict(), text=str(throughput))

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def env_snapshot(
        top: int = 5, interval: float = 0.1, include_gpu: bool = True
    ) -> dict[str, Any]:
        """一次性返回整机快照：平台、CPU、内存、磁盘、GPU、网络与 Top 进程。"""
        snapshot = Env.snapshot(top=top, interval=interval, include_gpu=include_gpu)
        return _done(**snapshot.to_dict(), report=snapshot.report(top=top))

    # ---------------------------------------------------------- 进程
    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def env_processes(
        limit: int = 20, sort: str = "cpu", name: str | None = None
    ) -> dict[str, Any]:
        """列举进程。sort 取 cpu / memory / pid / name / age；name 为子串过滤。"""
        items = Env.processes(limit=limit, sort=sort, name=name)
        return _done(count=len(items), items=[item.to_dict() for item in items])

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def env_process(pid: int, interval: float = 0.1) -> dict[str, Any]:
        """查看单个进程的详情（含命令行、启动时间、显存占用）。"""
        return _done(**Env.process(pid, interval=interval).to_dict())

    @server.tool(annotations={"destructiveHint": True})
    @_guard
    def env_kill(pid: int, force: bool = False) -> dict[str, Any]:
        """结束进程（不可恢复）。force=true 直接强杀，否则先请求退出。"""
        return _done(pid=pid, killed=Env.kill(pid, force=force), force=force)

    # ---------------------------------------------------------- 命令 / 环境变量
    @server.tool()
    @_guard
    def env_run(
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        timeout: float = 30.0,
        stdin: str | None = None,
    ) -> dict[str, Any]:
        """安全执行一条命令并返回 stdout/stderr/退出码（不经 shell）。

        command 传可执行文件名，args 传参数列表；若只传 command 字符串，
        会按空白与引号分词，含 | & ; > $ 等 shell 元字符则直接报错。
        需要管道时，把上一条命令的 stdout 作为本条 stdin 传入。
        """
        source = (
            Command(argv=(command, *args)) if args else Command.from_string(command)
        )
        result = Env.run(source, cwd=cwd, timeout=timeout, stdin=stdin)
        return {"ok": result.ok, **result.to_dict()}

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def env_which(name: str) -> dict[str, Any]:
        """在 PATH 中查找可执行文件。"""
        found = Env.which(name)
        return _done(name=name, path=str(found) if found else None, found=bool(found))

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def env_vars(prefix: str | None = None, reveal: bool = False) -> dict[str, Any]:
        """读取环境变量；密码/令牌类字段默认脱敏为 ***，reveal=true 返回原值。"""
        values = Env.environ(prefix, reveal=reveal)
        return _done(count=len(values), reveal=reveal, items=values)

    @server.tool(annotations={"readOnlyHint": True})
    @_guard
    def env_get(name: str) -> dict[str, Any]:
        """读取单个环境变量（敏感名同样脱敏）。"""
        value = Env.getenv(name)
        shown = mask(value or "") if is_sensitive(name) else value
        return _done(name=name, value=shown, exists=value is not None)

    return server


mcp = create_server()
