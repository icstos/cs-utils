"""标注平台的数据库层：SQLite + SQLModel。

三块内容：

- :mod:`cs.cv.db.base` —— 公共列（:class:`TModelBase`）与通用 CRUD（:class:`DataModel`）；
- :mod:`cs.cv.db.tables` —— 十张业务表与三张多对多连接表；
- :mod:`cs.cv.db.engine` —— 本地 SQLite 引擎、PRAGMA 与建表。

典型用法::

    from cs.cv.db import TDataset, TImg, create_tables, sqlite_engine

    engine = sqlite_engine("output/anno.db")
    create_tables(engine)

    dataset = TDataset.create({"name": "ds-001", "task_type": 2}, engine=engine)
    TImg(name="a.jpg", width=810, height=1080, dataset=dataset).save(session)

    for img in TDataset.by_id(dataset.id, engine=engine).imgs:
        print(img.name, img.img_size)
"""

from .base import SALT, DataModel, TModelBase, compute_hash, session_scope
from .engine import MEMORY, create_tables, sqlite_engine
from .tables import (
    TABLES,
    TCategory,
    TClassifyAnno,
    TDataset,
    TDatasetTUserLink,
    TDetectAnno,
    TImg,
    TModel,
    TProject,
    TProjectSet,
    TProjectSetTUserLink,
    TProjectTUserLink,
    TSegmentAnno,
    TUser,
    pack_points,
    unpack_points,
)

__all__ = [
    "MEMORY",
    "SALT",
    "TABLES",
    "DataModel",
    "TCategory",
    "TClassifyAnno",
    "TDataset",
    "TDatasetTUserLink",
    "TDetectAnno",
    "TImg",
    "TModel",
    "TModelBase",
    "TProject",
    "TProjectSet",
    "TProjectSetTUserLink",
    "TProjectTUserLink",
    "TSegmentAnno",
    "TUser",
    "compute_hash",
    "create_tables",
    "pack_points",
    "session_scope",
    "sqlite_engine",
    "unpack_points",
]
