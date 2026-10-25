"""MCP 场景的会话池。

MCP 工具是"一次调用一回合"的，如果每个操作都重新握手（TCP + 协商 + 认证 + 挂载），
既慢又啰嗦。这里的 :class:`SessionRegistry` 把连接变成具名句柄：

* ``samba_open`` 建会话 → 后续所有工具用 ``session="default"`` 复用；
* 空闲超时（默认 15 分钟）自动回收，进程退出时统一关闭；
* 同一名字重复 open 会顶替旧连接（换账号/换共享时不必先 close）。

线程安全：一把 ``RLock`` 保护整个注册表；每个会话内部的 SMB 调用由后端自己串行化。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .client import SMBClient
from .errors import SambaError, SambaNotFoundError
from .models import SMBOptions

__all__ = ["DEFAULT_SESSION", "LiveSession", "SessionRegistry"]

#: 默认会话名：工具不传 ``session`` 时用它
DEFAULT_SESSION = "default"


@dataclass(slots=True)
class LiveSession:
    """一个已连接的具名会话。"""

    name: str
    client: SMBClient
    created_at: datetime
    last_used: datetime

    def touch(self) -> None:
        self.last_used = datetime.now(UTC)

    @property
    def idle_seconds(self) -> float:
        return (datetime.now(UTC) - self.last_used).total_seconds()

    def info(self) -> dict[str, Any]:
        return {
            "session": self.name,
            "created_at": self.created_at.astimezone().isoformat(timespec="seconds"),
            "idle_seconds": round(self.idle_seconds, 1),
            **self.client.info().to_dict(),
        }

    def close(self) -> None:
        self.client.close()


class SessionRegistry:
    """具名连接池（进程内单例使用）。

    :param max_idle: 空闲多少秒后自动回收
    :param max_sessions: 同时存在的会话上限，超出时淘汰最久未用的
    :param client_factory: 客户端工厂，默认 :class:`SMBClient`（测试可注入假实现）
    """

    def __init__(
        self,
        *,
        max_idle: float = 900.0,
        max_sessions: int = 8,
        client_factory: Callable[[SMBOptions], SMBClient] = SMBClient,
    ) -> None:
        if max_idle <= 0:
            raise ValueError(f"max_idle 必须为正数，收到 {max_idle}")
        if max_sessions < 1:
            raise ValueError(f"max_sessions 必须 ≥ 1，收到 {max_sessions}")
        self.max_idle = max_idle
        self.max_sessions = max_sessions
        self._factory = client_factory
        self._lock = threading.RLock()
        self._sessions: dict[str, LiveSession] = {}

    # -- 生命周期
    def open(self, options: SMBOptions, *, name: str = DEFAULT_SESSION) -> LiveSession:
        """建立（或顶替）一个具名会话。"""
        key = name.strip() or DEFAULT_SESSION
        with self._lock:
            self._sweep()
            previous = self._sessions.pop(key, None)
            if previous is not None:
                previous.close()
            while len(self._sessions) >= self.max_sessions:
                oldest = min(self._sessions.values(), key=lambda item: item.last_used)
                self._sessions.pop(oldest.name, None)
                oldest.close()

            client = self._factory(options)
            now = datetime.now(UTC)
            session = LiveSession(
                name=key, client=client, created_at=now, last_used=now
            )
            self._sessions[key] = session
            return session

    def get(self, name: str = DEFAULT_SESSION) -> LiveSession:
        """取出会话并续期；不存在时给出"先 open"的可执行提示。"""
        key = (name or DEFAULT_SESSION).strip()
        with self._lock:
            self._sweep()
            session = self._sessions.get(key)
            if session is None:
                known = "、".join(sorted(self._sessions)) or "无"
                raise SambaNotFoundError(
                    f"session={key!r}",
                    f"会话不存在或已超时回收（现有会话：{known}），请先调用 samba_open",
                )
            session.touch()
            return session

    def close(self, name: str = DEFAULT_SESSION) -> bool:
        """关闭并移除会话；返回是否真的关闭了。"""
        key = (name or DEFAULT_SESSION).strip()
        with self._lock:
            session = self._sessions.pop(key, None)
            if session is None:
                return False
            session.close()
            return True

    def close_all(self) -> int:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            try:
                session.close()
            except SambaError:
                pass
        return len(sessions)

    # -- 观测
    def snapshot(self) -> list[dict[str, Any]]:
        """列出当前会话（轻量，不触发网络 IO）。"""
        with self._lock:
            self._sweep()
            return [
                {
                    "session": item.name,
                    "target": item.client.options.unc,
                    "share": item.client.options.share,
                    "username": item.client.options.username or "anonymous",
                    "connected": item.client.connected,
                    "created_at": item.created_at.astimezone().isoformat(
                        timespec="seconds"
                    ),
                    "idle_seconds": round(item.idle_seconds, 1),
                }
                for item in self._sessions.values()
            ]

    @property
    def names(self) -> list[str]:
        with self._lock:
            self._sweep()
            return sorted(self._sessions)

    # -- 内部
    def _sweep(self) -> None:
        """回收空闲超时的会话（调用方需持有锁）。"""
        expired = [
            item
            for item in self._sessions.values()
            if item.idle_seconds > self.max_idle
        ]
        for item in expired:
            self._sessions.pop(item.name, None)
            try:
                item.close()
            except SambaError:
                pass
