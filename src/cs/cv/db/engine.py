"""本地 SQLite 引擎：连接参数、PRAGMA 与建表。

只管“怎么连、怎么建”，不掺业务逻辑。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import SQLModel, create_engine

from . import tables as _tables  # noqa: F401  导入即把所有表注册到 metadata

__all__ = ["MEMORY", "create_tables", "sqlite_engine"]

#: 内存库标识（进程退出即丢，适合测试）
MEMORY = ":memory:"


def sqlite_engine(
    db_file: str | Path = MEMORY, *, echo: bool = False, wal: bool = True
) -> Engine:
    """创建本地 SQLite 引擎。

    - 每个连接自动 ``PRAGMA foreign_keys=ON``（SQLite 默认是关的，外键与级联才不会失效）；
    - 文件库默认开 WAL，读写可以并发；内存库跳过；
    - 相对路径会转成绝对路径，避免工作目录变化后连到另一个文件。
    """
    location = str(db_file)
    if location in {MEMORY, "", "sqlite://"}:
        engine = create_engine("sqlite://", echo=echo)
        memory = True
    else:
        path = Path(location).expanduser().absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{path.as_posix()}", echo=echo)
        memory = False

    @event.listens_for(engine, "connect")
    def _setup(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        if wal and not memory:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    return engine


def create_tables(engine: Engine) -> None:
    """建表（幂等，已存在的表不动）。"""
    SQLModel.metadata.create_all(engine)
