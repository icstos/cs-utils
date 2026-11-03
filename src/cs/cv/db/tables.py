"""标注平台的数据库表（SQLite + SQLModel）。

表一览：

==================  ==========================================================
``user``            标注员 / 管理员
``project_set``     项目集（项目分组）
``project``         项目（一次交付的标注任务集合）
``dataset``         数据集（图片 + 类别 + 标注 + 模型）
``category``        数据集内的类别表，``idx`` 即 YOLO 类别号
``img``             图片
``classify_anno``   分类标注（整图一个类别）
``detect_anno``     检测标注（水平矩形框）
``segment_anno``    分割标注（多边形，顶点以 JSON 存文本）
``model``           模型登记表
==================  ==========================================================

外键与删除策略：

- 归属树 ``project_set → project → dataset → img / category → *_anno`` 全部
  ``ON DELETE CASCADE`` —— 删容器即删其下内容；
- 指向用户的外键（``creator_user_id`` / ``head_user_id``）用 ``ON DELETE SET NULL``
  —— 删账号不删业务数据；
- ``model.dataset_id`` 用 ``ON DELETE SET NULL`` —— 权重文件独立于数据集存续；
- 唯一约束：``user(name, display_name)``、``category(dataset_id, idx)``、
  ``img(dataset_id, name)`` —— 同一数据集内类别号与图片名不重复。

时间字段统一用本地朴素 ``datetime``（与既有 anno.db 数据一致，界面直接展示）。
"""

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Optional

from sqlalchemy import Index, UniqueConstraint
from sqlmodel import Field, Relationship, Session, SQLModel, select

from ...my_types import ImgSize, TaskType, UserType
from ..geometry import Point
from .base import DataModel, TModelBase, compute_hash

__all__ = [
    "TABLES",
    "TCategory",
    "TClassifyAnno",
    "TDataset",
    "TDatasetTUserLink",
    "TDetectAnno",
    "TImg",
    "TModel",
    "TProject",
    "TProjectSet",
    "TProjectSetTUserLink",
    "TProjectTUserLink",
    "TSegmentAnno",
    "TUser",
    "pack_points",
    "unpack_points",
]

#: 父删子：ORM 级联 + 交给数据库 ``ON DELETE CASCADE``，避免逐行 UPDATE 置空
_CASCADE = {"cascade": "all, delete-orphan", "passive_deletes": True}


# ============================================================
# 多对多连接表
# ============================================================


class TProjectSetTUserLink(SQLModel, table=True):
    """项目集 ↔ 用户。"""

    __tablename__: ClassVar[str] = "project_set_user_link"

    project_set_id: int = Field(
        foreign_key="project_set.id", ondelete="CASCADE", primary_key=True
    )
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE", primary_key=True)


class TProjectTUserLink(SQLModel, table=True):
    """项目 ↔ 用户（项目成员）。"""

    __tablename__: ClassVar[str] = "project_user_link"

    project_id: int = Field(
        foreign_key="project.id", ondelete="CASCADE", primary_key=True
    )
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE", primary_key=True)


class TDatasetTUserLink(SQLModel, table=True):
    """数据集 ↔ 用户（标注人员）。"""

    __tablename__: ClassVar[str] = "dataset_user_link"

    dataset_id: int = Field(
        foreign_key="dataset.id", ondelete="CASCADE", primary_key=True
    )
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE", primary_key=True)


# ============================================================
# 用户
# ============================================================


class TUser(TModelBase, DataModel, table=True):
    """账号：登录信息、角色与负责的容器。"""

    __tablename__: ClassVar[str] = "user"
    __table_args__ = (
        UniqueConstraint("name", "display_name", name="uq_user_name_display"),
    )

    index_column_list: ClassVar[tuple[str, ...]] = ("name", "display_name")
    show_column_list: ClassVar[tuple[str, ...]] = (
        "name",
        "display_name",
        "user_type",
        "use_status",
        "create_time",
        "last_login_time",
    )

    password: str = Field(
        default_factory=lambda: compute_hash("123456"),
        nullable=False,
        max_length=128,
        description="口令散列（加盐 SHA-256）",
    )
    email: str = Field(
        default="user_name@email.com",
        nullable=False,
        index=True,
        max_length=200,
        description="邮箱",
    )
    user_type: int = Field(
        default=UserType.CUSTOMER.value, index=True, description="角色，见 UserType"
    )
    last_login_time: datetime = Field(
        default_factory=datetime.now, nullable=False, description="最后登录时间"
    )

    # 多对多
    project_sets: list["TProjectSet"] = Relationship(
        back_populates="users", link_model=TProjectSetTUserLink
    )
    projects: list["TProject"] = Relationship(
        back_populates="users", link_model=TProjectTUserLink
    )
    datasets: list["TDataset"] = Relationship(
        back_populates="users", link_model=TDatasetTUserLink
    )

    # 一对多：我负责的容器
    head_project_sets: list["TProjectSet"] = Relationship(
        back_populates="head_user",
        sa_relationship_kwargs={"foreign_keys": "[TProjectSet.head_user_id]"},
    )
    head_projects: list["TProject"] = Relationship(
        back_populates="head_user",
        sa_relationship_kwargs={"foreign_keys": "[TProject.head_user_id]"},
    )
    head_datasets: list["TDataset"] = Relationship(
        back_populates="head_user",
        sa_relationship_kwargs={"foreign_keys": "[TDataset.head_user_id]"},
    )

    @staticmethod
    def password_validator(value: str) -> str:
        """校验口令强度，不合规抛 ``ValueError``。"""
        import re

        message = "密码需要：最少8 字符，其中至少：1大写字母、1小写字线、1数字、1特殊字符（“+_-”）"
        for pattern in (r".{8}", r"[A-Z]", r"[a-z]", r"[0-9]", r"[^a-zA-Z0-9]"):
            if re.search(pattern, value) is None:
                raise ValueError(message)
        return value


# ============================================================
# 项目集 / 项目
# ============================================================


class TProjectSet(TModelBase, DataModel, table=True):
    """项目集：把多个项目归到一组。"""

    __tablename__: ClassVar[str] = "project_set"

    index_column_list: ClassVar[tuple[str, ...]] = ("name", "display_name", "detail")
    show_column_list: ClassVar[tuple[str, ...]] = ("name", "display_name")

    detail: str = Field(default="", max_length=2000, description="说明")
    creator_user_id: int | None = Field(
        default=None,
        foreign_key="user.id",
        ondelete="SET NULL",
        index=True,
        description="创建者",
    )
    head_user_id: int | None = Field(
        default=None,
        foreign_key="user.id",
        ondelete="SET NULL",
        index=True,
        description="负责人",
    )
    head_user: Optional["TUser"] = Relationship(
        back_populates="head_project_sets",
        sa_relationship_kwargs={"foreign_keys": "[TProjectSet.head_user_id]"},
    )

    projects: list["TProject"] = Relationship(
        back_populates="project_set", sa_relationship_kwargs=dict(_CASCADE)
    )
    users: list["TUser"] = Relationship(
        back_populates="project_sets", link_model=TProjectSetTUserLink
    )


class TProject(TModelBase, DataModel, table=True):
    """项目：一个交付单元，下挂若干数据集。"""

    __tablename__: ClassVar[str] = "project"

    index_column_list: ClassVar[tuple[str, ...]] = ("name", "display_name", "detail")
    show_column_list: ClassVar[tuple[str, ...]] = (
        "name",
        "display_name",
        "project_set_id",
        "creator_user_id",
        "head_user_id",
        "use_status",
        "detail",
    )

    detail: str = Field(default="", max_length=2000, description="说明")
    creator_user_id: int | None = Field(
        default=None,
        foreign_key="user.id",
        ondelete="SET NULL",
        index=True,
        description="创建者",
    )
    project_set_id: int | None = Field(
        default=None,
        foreign_key="project_set.id",
        ondelete="CASCADE",
        index=True,
        description="所属项目集",
    )
    project_set: Optional["TProjectSet"] = Relationship(back_populates="projects")

    head_user_id: int | None = Field(
        default=None,
        foreign_key="user.id",
        ondelete="SET NULL",
        index=True,
        description="负责人",
    )
    head_user: Optional["TUser"] = Relationship(
        back_populates="head_projects",
        sa_relationship_kwargs={"foreign_keys": "[TProject.head_user_id]"},
    )

    datasets: list["TDataset"] = Relationship(
        back_populates="project", sa_relationship_kwargs=dict(_CASCADE)
    )
    users: list["TUser"] = Relationship(
        back_populates="projects", link_model=TProjectTUserLink
    )


# ============================================================
# 数据集 / 类别 / 图片
# ============================================================


class TDataset(TModelBase, DataModel, table=True):
    """数据集：图片目录 + 类别表 + 标注 + 模型产物。"""

    __tablename__: ClassVar[str] = "dataset"

    index_column_list: ClassVar[tuple[str, ...]] = ("name", "display_name", "detail")
    show_column_list: ClassVar[tuple[str, ...]] = (
        "project_id",
        "name",
        "display_name",
        "task_type",
        "creator_user_id",
        "head_user_id",
        "users",
        "use_status",
    )

    task_type: int = Field(
        default=TaskType.DETECT.value, index=True, description="任务类型，见 TaskType"
    )
    detail: str = Field(default="", max_length=2000, description="说明")
    file_dir: str | None = Field(default=None, max_length=500, description="数据根目录")
    creator_user_id: int | None = Field(
        default=None,
        foreign_key="user.id",
        ondelete="SET NULL",
        index=True,
        description="创建者",
    )
    project_id: int | None = Field(
        default=None,
        foreign_key="project.id",
        ondelete="CASCADE",
        index=True,
        description="所属项目",
    )
    project: Optional["TProject"] = Relationship(back_populates="datasets")

    head_user_id: int | None = Field(
        default=None,
        foreign_key="user.id",
        ondelete="SET NULL",
        index=True,
        description="负责人",
    )
    head_user: Optional["TUser"] = Relationship(
        back_populates="head_datasets",
        sa_relationship_kwargs={"foreign_keys": "[TDataset.head_user_id]"},
    )

    imgs: list["TImg"] = Relationship(
        back_populates="dataset", sa_relationship_kwargs=dict(_CASCADE)
    )
    categorys: list["TCategory"] = Relationship(
        back_populates="dataset", sa_relationship_kwargs=dict(_CASCADE)
    )
    models: list["TModel"] = Relationship(back_populates="dataset")
    users: list["TUser"] = Relationship(
        back_populates="datasets", link_model=TDatasetTUserLink
    )

    # ---------------------------------------------------------- 路径
    @property
    def _base(self) -> Path:
        return Path(self.file_dir) if self.file_dir else Path()

    @property
    def img_dir(self) -> Path:
        """图片目录。"""
        return self._base / "images"

    @property
    def anno_db(self) -> Path:
        """标注数据库文件。"""
        return self._base / "anno.db"

    @property
    def anno_dir(self) -> Path:
        """标注目录。"""
        return self._base / "annos"

    @property
    def model_dir(self) -> Path:
        """模型目录。"""
        return self._base / "models"

    @property
    def cache_dir(self) -> Path:
        """缓存目录。"""
        return self._base / "cache"

    @property
    def data_yaml(self) -> Path:
        """YOLO 数据集配置。"""
        return self._base / "dataset_config.yaml"

    # ---------------------------------------------------------- 统计
    def get_category_names(self, session: Session) -> list[str]:
        """按 ``idx`` 升序返回类别名称（YOLO 的 names 列表）。"""
        dataset = session.get(TDataset, self.id)
        if dataset is None:
            return []
        return [c.display_name for c in sorted(dataset.categorys, key=lambda c: c.idx)]

    def get_category_name_list(self, session: Session) -> list[str]:
        """:meth:`get_category_names` 的旧名（保留兼容）。"""
        return self.get_category_names(session)

    _TASK_LABEL: ClassVar[dict[int, str]] = {
        TaskType.CLASSIFY.value: "图像分类",
        TaskType.DETECT.value: "目标检测",
        TaskType.SEGMENT.value: "图像分割",
    }
    _ANNO_ATTR: ClassVar[dict[int, str]] = {
        TaskType.CLASSIFY.value: "classify_annos",
        TaskType.DETECT.value: "detect_annos",
        TaskType.SEGMENT.value: "segment_annos",
    }

    def _count_by_category(self, session: Session) -> dict[str, int]:
        """按类别统计标注数量（分类/检测/分割通用）。"""
        dataset = session.get(TDataset, self.id)
        if dataset is None:
            return {}
        counts = {str(c.display_name): 0 for c in dataset.categorys}
        attr = self._ANNO_ATTR.get(self.task_type)
        if attr is None:
            return counts
        for img in dataset.imgs:
            for anno in getattr(img, attr, ()):
                category = anno.category
                if category is not None and category.display_name in counts:
                    counts[category.display_name] += 1
        return counts

    def v_msg(self, session: Session):
        """给 Flet 用的数据集概要面板（flet 按需导入）。"""
        import flet as ft

        dataset = session.get(TDataset, self.id)
        if dataset is None:
            return ft.Text("数据集不存在")

        img_count = session.exec(select(TImg).where(TImg.dataset_id == self.id)).all()
        rows = (
            ("数据集名称：", self.name),
            ("任务类别：", self._TASK_LABEL.get(self.task_type, "未知")),
            ("图像数量", str(len(img_count))),
            ("图像文件夹：", str(self.img_dir)),
            ("目标类别：", str([c.display_name for c in dataset.categorys])),
            ("任务描述：", self.detail),
            ("缺陷类别数量：", str(self._count_by_category(session))),
        )
        return ft.SelectionArea(
            ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Icon(icon=ft.Icons.FIBER_MANUAL_RECORD),
                            ft.Text(label),
                            ft.Text(value=value, color=ft.Colors.BLUE, max_lines=3),
                        ],
                        wrap=True,
                    )
                    for label, value in rows
                ]
            )
        )

    def load_img_anno(self, img_dir: Path) -> None:
        """占位：由导入流程实现（保持旧接口）。"""


class TCategory(TModelBase, DataModel, table=True):
    """数据集内的一个类别，``idx`` 与 YOLO 标签首列一致。"""

    __tablename__: ClassVar[str] = "category"
    __table_args__ = (
        UniqueConstraint("dataset_id", "idx", name="uq_category_dataset_idx"),
    )

    color: str = Field(default="#000000", max_length=9, description="调色板颜色")
    idx: int = Field(default=0, description="类别号（YOLO 标签首列）")

    dataset_id: int | None = Field(
        default=None,
        foreign_key="dataset.id",
        ondelete="CASCADE",
        index=True,
        description="所属数据集",
    )
    dataset: Optional["TDataset"] = Relationship(back_populates="categorys")

    classify_annos: list["TClassifyAnno"] = Relationship(
        back_populates="category", sa_relationship_kwargs=dict(_CASCADE)
    )
    detect_annos: list["TDetectAnno"] = Relationship(
        back_populates="category", sa_relationship_kwargs=dict(_CASCADE)
    )
    segment_annos: list["TSegmentAnno"] = Relationship(
        back_populates="category", sa_relationship_kwargs=dict(_CASCADE)
    )

    def __repr__(self) -> str:
        return f"{self.idx}: {self.name}-{self.display_name}"


class TImg(TModelBase, DataModel, table=True):
    """一张图片（``name`` 为文件名，在数据集内唯一）。"""

    __tablename__: ClassVar[str] = "img"
    __table_args__ = (
        UniqueConstraint("dataset_id", "name", name="uq_img_dataset_name"),
    )

    width: int = Field(default=-1, description="宽度（像素）")
    height: int = Field(default=-1, description="高度（像素）")

    dataset_id: int | None = Field(
        default=None,
        foreign_key="dataset.id",
        ondelete="CASCADE",
        index=True,
        description="所属数据集",
    )
    dataset: Optional["TDataset"] = Relationship(back_populates="imgs")

    classify_annos: list["TClassifyAnno"] = Relationship(
        back_populates="img", sa_relationship_kwargs=dict(_CASCADE)
    )
    detect_annos: list["TDetectAnno"] = Relationship(
        back_populates="img", sa_relationship_kwargs=dict(_CASCADE)
    )
    segment_annos: list["TSegmentAnno"] = Relationship(
        back_populates="img", sa_relationship_kwargs=dict(_CASCADE)
    )

    @property
    def img_size(self) -> ImgSize:
        """尺寸对象，便于与 :mod:`cs.cv.geometry` 互转。"""
        return ImgSize(self.width, self.height)


# ============================================================
# 标注
# ============================================================


def pack_points(points: Iterable[Point]) -> str:
    """把顶点序列打包成 JSON 文本 ``[[x, y], ...]``。"""
    return json.dumps([[round(p.x, 3), round(p.y, 3)] for p in points])


def unpack_points(raw: str | None) -> list[Point]:
    """从 JSON 文本还原顶点；空值返回空列表。"""
    if not raw:
        return []
    return [Point(float(x), float(y)) for x, y in json.loads(raw)]


class TClassifyAnno(TModelBase, DataModel, table=True):
    """分类标注：整图一个类别。

    ``conf`` 语义：``0.0~0.99`` 为模型置信度；``1`` 人工确认类别与位置都正确；
    ``2`` 类别对、位置可能不对；``3`` 位置对、类别可能不对；``4`` 都可能不对。
    """

    __tablename__: ClassVar[str] = "classify_anno"
    __table_args__ = (Index("ix_classify_anno_img_category", "img_id", "category_id"),)

    conf: float = Field(default=-1.0, description="置信度或人工确认等级")

    img_id: int | None = Field(
        default=None,
        foreign_key="img.id",
        ondelete="CASCADE",
        index=True,
        description="所属图片",
    )
    img: Optional["TImg"] = Relationship(back_populates="classify_annos")

    category_id: int | None = Field(
        default=None,
        foreign_key="category.id",
        ondelete="CASCADE",
        index=True,
        description="所属类别",
    )
    category: Optional["TCategory"] = Relationship(back_populates="classify_annos")

    @property
    def anno(self) -> Point:
        """转成几何图元（分类无位置，返回左上角点）。"""
        return Point(
            x=0,
            y=0,
            conf=self.conf,
            category_idx=self.category.idx if self.category is not None else -1,
        )


class TDetectAnno(TModelBase, DataModel, table=True):
    """检测标注：水平矩形框，存左上角与宽高（像素）。"""

    __tablename__: ClassVar[str] = "detect_anno"
    __table_args__ = (Index("ix_detect_anno_img_category", "img_id", "category_id"),)

    x: float = Field(default=0.0, description="左上角 x")
    y: float = Field(default=0.0, description="左上角 y")
    width: float = Field(default=0.0, description="宽")
    height: float = Field(default=0.0, description="高")
    conf: float = Field(default=-1.0, description="置信度或人工确认等级")

    img_id: int | None = Field(
        default=None,
        foreign_key="img.id",
        ondelete="CASCADE",
        index=True,
        description="所属图片",
    )
    img: Optional["TImg"] = Relationship(back_populates="detect_annos")

    category_id: int | None = Field(
        default=None,
        foreign_key="category.id",
        ondelete="CASCADE",
        index=True,
        description="所属类别",
    )
    category: Optional["TCategory"] = Relationship(back_populates="detect_annos")

    @property
    def anno(self):
        """转成 :class:`cs.cv.geometry.Box`。"""
        from ..geometry import Box

        return Box(
            start=Point(self.x, self.y),
            end=Point(self.x + self.width, self.y + self.height),
            category_idx=self.category.idx if self.category else -1,
            conf=self.conf,
            create_time=self.create_time,
        )

    def to_str(self, img_size: ImgSize) -> str:
        """转成一行 YOLO 检测标签。"""
        return self.anno.to_yolo(img_size)


class TSegmentAnno(TModelBase, DataModel, table=True):
    """分割标注：多边形，顶点以 JSON 文本存储（不再使用 pickle）。"""

    __tablename__: ClassVar[str] = "segment_anno"
    __table_args__ = (Index("ix_segment_anno_img_category", "img_id", "category_id"),)

    points: str = Field(default="[]", description="顶点 JSON：[[x, y], ...]")
    conf: float = Field(default=-1.0, description="置信度或人工确认等级")

    img_id: int | None = Field(
        default=None,
        foreign_key="img.id",
        ondelete="CASCADE",
        index=True,
        description="所属图片",
    )
    img: Optional["TImg"] = Relationship(back_populates="segment_annos")

    category_id: int | None = Field(
        default=None,
        foreign_key="category.id",
        ondelete="CASCADE",
        index=True,
        description="所属类别",
    )
    category: Optional["TCategory"] = Relationship(back_populates="segment_annos")

    @property
    def point_list(self) -> list[Point]:
        """顶点序列。"""
        return unpack_points(self.points)

    @property
    def anno(self):
        """转成 :class:`cs.cv.geometry.Polygon`。"""
        from ..geometry import Polygon

        return Polygon(
            points=tuple(self.point_list),
            category_idx=self.category.idx if self.category else -1,
            conf=self.conf,
            create_time=self.create_time,
        )

    def to_str(self, img_size: ImgSize) -> str:
        """转成一行 YOLO 分割标签。"""
        return self.anno.to_yolo(img_size)

    @classmethod
    def from_polygon(cls, polygon: Any, **extra: Any) -> "TSegmentAnno":
        """由 :class:`cs.cv.geometry.Polygon` 构造（自动打包顶点）。"""
        return cls(points=pack_points(polygon.points), conf=polygon.conf, **extra)


# ============================================================
# 模型登记
# ============================================================


class TModel(TModelBase, DataModel, table=True):
    """模型登记：记录某个数据集上训练/导入的权重。"""

    __tablename__: ClassVar[str] = "model"

    index_column_list: ClassVar[tuple[str, ...]] = ("name", "display_name")
    show_column_list: ClassVar[tuple[str, ...]] = (
        "name",
        "display_name",
        "model_type",
        "use_status",
        "create_time",
    )

    model_type: int = Field(
        default=TaskType.DETECT.value, index=True, description="模型任务，见 TaskType"
    )
    detail: str = Field(default="", max_length=2000, description="说明")
    weight_file: str | None = Field(
        default=None, max_length=500, description="权重文件名（相对 dataset.model_dir）"
    )

    dataset_id: int | None = Field(
        default=None,
        foreign_key="dataset.id",
        ondelete="SET NULL",
        index=True,
        description="所属数据集",
    )
    dataset: Optional["TDataset"] = Relationship(back_populates="models")


#: 需要建表的全部实体（``SQLModel.metadata`` 已自动收录，这里给显式清单）
TABLES: tuple[type[TModelBase], ...] = (
    TUser,
    TProjectSet,
    TProject,
    TDataset,
    TCategory,
    TImg,
    TClassifyAnno,
    TDetectAnno,
    TSegmentAnno,
    TModel,
)
