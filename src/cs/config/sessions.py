"""MCP 会话池：让一串 ``config_*`` 调用复用同一个 :class:`ConfigManager`。

LLM 的调用序列通常是「打开 → 若干次读/改 → 保存 → 关闭」，如果每次都从磁盘重建，
既慢又容易丢改动，所以这里按名字持有打开的配置。空闲超时自动回收，会话数有上限
（LRU 淘汰），避免长跑进程里越攒越多。
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import ConfigSessionError
from .manager import ConfigManager

__all__ = ["DEFAULT_SESSION", "ConfigRegistry", "LiveConfig"]

#: 默认会话名
DEFAULT_SESSION = "default"


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class LiveConfig:
    """一个已打开的配置会话。"""

    name: str
    manager: ConfigManager[Any]
    created_at: datetime = field(default_factory=_now)
    last_used: datetime = field(default_factory=_now)

    def touch(self) -> None:
        self.last_used = _now()

    @property
    def idle_seconds(self) -> float:
        return max(0.0, (_now() - self.last_used).total_seconds())

    def describe(self) -> dict[str, Any]:
        summary = self.manager.describe()
        return {
            "session": self.name,
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "idle_seconds": round(self.idle_seconds, 1),
            **summary,
        }

    def close(self) -> None:
        """配置会话没有需要释放的连接，接口保留是为了与 samba 的会话池对齐。"""


class ConfigRegistry:
    """线程安全的配置会话池。"""

    def __init__(
        self,
        *,
        max_idle: float = 1800.0,
        max_sessions: int = 8,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions 至少为 1")
        self.max_idle = max_idle
        self.max_sessions = max_sessions
        self._live: dict[str, LiveConfig] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------ 查询
    def __len__(self) -> int:
        with self._lock:
            self._sweep()
            return len(self._live)

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return isinstance(name, str) and name in self._live

    @property
    def names(self) -> list[str]:
        with self._lock:
            self._sweep()
            return list(self._live)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            self._sweep()
            return [live.describe() for live in self._live.values()]

    # ------------------------------------------------------------ 开关
    def open(
        self,
        path: str | Path,
        model: type | None = None,
        *,
        name: str = DEFAULT_SESSION,
        **kwargs: Any,
    ) -> LiveConfig:
        """打开（或顶替）一个会话并立即加载。"""
        manager: ConfigManager[Any] = ConfigManager(model, path, **kwargs)
        manager.load()
        live = LiveConfig(name=name, manager=manager)
        with self._lock:
            self._evict_if_needed(exclude=name)
            self._live[name] = live
        return live

    def get(self, name: str = DEFAULT_SESSION) -> LiveConfig:
        with self._lock:
            self._sweep()
            live = self._live.get(name)
            if live is None:
                existing = ", ".join(self._live) or "无"
                raise ConfigSessionError(
                    f"配置会话不存在或已超时回收（现有会话：{existing}），"
                    f"请先调用 config_open: session={name!r}"
                )
            live.touch()
            return live

    def close(self, name: str = DEFAULT_SESSION) -> bool:
        with self._lock:
            live = self._live.pop(name, None)
            if live is None:
                return False
            live.close()
            return True

    def close_all(self) -> int:
        with self._lock:
            names = list(self._live)
            for name in names:
                self.close(name)
            return len(names)

    # ------------------------------------------------------------ 内部
    def _sweep(self) -> None:
        """回收空闲超时的会话。"""
        if self.max_idle <= 0:
            return
        stale = [
            name
            for name, live in self._live.items()
            if live.idle_seconds > self.max_idle
        ]
        for name in stale:
            self.close(name)

    def _evict_if_needed(self, *, exclude: str) -> None:
        """会话数超上限时淘汰最久未使用的那个。"""
        while len(self._live) >= self.max_sessions:
            candidates: Iterator[tuple[str, LiveConfig]] = (
                (name, live) for name, live in self._live.items() if name != exclude
            )
            oldest = min(candidates, key=lambda item: item[1].last_used, default=None)
            if oldest is None:
                break
            self.close(oldest[0])
