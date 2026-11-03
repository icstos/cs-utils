"""SQLModel 基座：公共列、通用 CRUD、口令散列与事务作用域。

设计要点：

- :class:`TModelBase` 是所有实体的公共列集合（主键、代号、名称、状态、时间戳）；
- :class:`DataModel` 提供 ``by_id / all / first / count / search / create / save``，
  每个方法都接受 ``engine`` 或 ``session``：传 ``session`` 时**不接管**其生命周期，
  传 ``engine`` 时临时开一个 Session 用完即关；
- 口令是加盐 SHA-256 单次散列（与旧库数据保持一致，未引入新依赖）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any, ClassVar, Self

from sqlalchemy import func, or_
from sqlalchemy.engine import Engine
from sqlmodel import Field, Session, SQLModel, select

from ...my_types import Status

__all__ = [
    "SALT",
    "DataModel",
    "TModelBase",
    "compute_hash",
    "session_scope",
]

#: 静态盐（旧数据的散列口径，勿改，否则既有账号无法登录）
SALT = b"passwd"


def compute_hash(password: str, /) -> str:
    """加盐 SHA-256，返回 64 位十六进制摘要。"""
    return hashlib.sha256(SALT + password.encode()).hexdigest()


@contextmanager
def session_scope(
    engine: Engine | None = None, session: Session | None = None
) -> Iterator[Session]:
    """复用外部 ``session``，或按 ``engine`` 新建一个并在退出时自动关闭。"""
    if session is not None:
        yield session
        return
    if engine is None:
        raise ValueError("需要 engine 或 session 之一")
    with Session(engine) as owned:
        yield owned


class TModelBase(SQLModel):
    """所有实体表的公共列。"""

    id: int | None = Field(default=None, primary_key=True, description="主键")
    name: str = Field(
        default="", index=True, nullable=False, max_length=200, description="代号"
    )
    display_name: str = Field(
        default="", index=True, nullable=False, max_length=300, description="名称"
    )
    use_status: int = Field(
        default=Status.ACTIVE.value, index=True, description="状态：1 启用 / 2 弃用"
    )
    create_time: datetime = Field(
        default_factory=datetime.now, nullable=False, index=True, description="创建时间"
    )
    update_time: datetime = Field(
        default_factory=datetime.now,
        nullable=False,
        sa_column_kwargs={"onupdate": datetime.now},
        description="更新时间",
    )


class DataModel(SQLModel):
    """任意实体表都可直接使用的通用查询/写入方法。

    子类可通过 ``index_column_list`` 声明 :meth:`search` 默认检索的列，
    ``show_column_list`` 声明界面展示的列（表驱动 UI 用，不落库）。
    """

    index_column_list: ClassVar[tuple[str, ...]] = ("name", "display_name")
    show_column_list: ClassVar[tuple[str, ...]] = ("name", "display_name")

    # ---------------------------------------------------------- 查询
    @classmethod
    def by_id(
        cls,
        data_id: int,
        engine: Engine | None = None,
        session: Session | None = None,
    ) -> Self | None:
        """按主键取一条，不存在返回 ``None``。"""
        with session_scope(engine, session) as db:
            return db.get(cls, data_id)

    @classmethod
    def all(
        cls, engine: Engine | None = None, session: Session | None = None
    ) -> list[Self]:
        """取全表。"""
        with session_scope(engine, session) as db:
            return list(db.exec(select(cls)).all())

    @classmethod
    def first(
        cls, engine: Engine | None = None, session: Session | None = None
    ) -> Self | None:
        """取第一条，空表返回 ``None``。"""
        with session_scope(engine, session) as db:
            return db.exec(select(cls)).first()

    @classmethod
    def count(cls, engine: Engine | None = None, session: Session | None = None) -> int:
        """行数统计。"""
        with session_scope(engine, session) as db:
            return db.exec(select(func.count()).select_from(cls)).one()

    @staticmethod
    def _match(attr: Any, keyword: str, *, full: bool) -> Any:
        return attr == keyword if full else attr.ilike(f"%{keyword}%")

    @classmethod
    def search(
        cls,
        keyword: str | None = None,
        *,
        column_name: str | None = None,
        is_full_match: bool = False,
        engine: Engine | None = None,
        session: Session | None = None,
    ) -> list[Self]:
        """按关键字检索。

        - ``is_full_match=False``（默认）模糊匹配 ``LIKE %kw%``，``True`` 为精确相等；
        - 给了 ``column_name`` 只搜该列，否则搜 :attr:`index_column_list` 各列。
        """
        keyword_text = (keyword or "").strip()
        with session_scope(engine, session) as db:
            if not keyword_text:
                return list(db.exec(select(cls)).all())
            columns = (column_name,) if column_name else cls.index_column_list
            clause = or_(
                *(
                    cls._match(getattr(cls, name), keyword_text, full=is_full_match)
                    for name in columns
                )
            )
            return list(db.exec(select(cls).where(clause)).all())

    @classmethod
    def search_session(cls, session: Session, keyword: str | None = None) -> list[Self]:
        """:meth:`search` 的 session 版本（保留旧接口）。"""
        return cls.search(keyword, session=session)

    # ---------------------------------------------------------- 写入
    @classmethod
    def create(
        cls,
        data: dict[str, Any] | SQLModel | None = None,
        *,
        engine: Engine | None = None,
        session: Session | None = None,
    ) -> Self:
        """按 ``dict`` / 模型实例新建并提交，返回刷新后的对象。"""
        if data is None:
            raise ValueError("需要 dict 或模型实例作为初始值")
        with session_scope(engine, session) as db:
            obj = data if isinstance(data, cls) else cls.model_validate(data)
            db.add(obj)
            db.commit()
            db.refresh(obj)
            return obj

    def save(self, session: Session) -> Self:
        """提交当前对象并刷新（``session`` 由调用方管理）。"""
        session.add(self)
        session.commit()
        session.refresh(self)
        return self
