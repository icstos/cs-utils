"""cs.cv.db（标注平台数据库表）的测试。

全部用内存 SQLite 跑，不落盘、不依赖外部环境；只验证表设计约束
（主键/外键/唯一/索引）与实体 ↔ 几何图元的往返。
"""

import sys
from pathlib import Path

import pytest
from sqlalchemy import inspect
from sqlmodel import Session, SQLModel, select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs.cv import Box, ImgSize, Point, Polygon
from cs.cv.db import (
    TABLES,
    TCategory,
    TClassifyAnno,
    TDataset,
    TDetectAnno,
    TImg,
    TModel,
    TProject,
    TProjectSet,
    TSegmentAnno,
    TUser,
    compute_hash,
    create_tables,
    pack_points,
    session_scope,
    sqlite_engine,
    unpack_points,
)
from cs.my_types import Status, TaskType, UserType

SIZE = ImgSize(200, 100)


@pytest.fixture
def engine():
    eng = sqlite_engine()
    create_tables(eng)
    return eng


@pytest.fixture
def session(engine):
    with Session(engine) as sess:
        yield sess


def _dataset(session: Session, **extra) -> TDataset:
    ds = TDataset(name="ds", display_name="数据集", file_dir="output/ds", **extra)
    session.add(ds)
    session.commit()
    session.refresh(ds)
    return ds


# ============================================================
# 表结构
# ============================================================


def test_all_tables_registered(engine):
    names = {t.__tablename__ for t in TABLES}
    assert names == {
        "user",
        "project_set",
        "project",
        "dataset",
        "category",
        "img",
        "classify_anno",
        "detect_anno",
        "segment_anno",
        "model",
    }
    assert names <= set(SQLModel.metadata.tables)


def test_common_columns_and_defaults(session):
    user = TUser(name="u")
    session.add(user)
    session.commit()
    session.refresh(user)
    assert user.id is not None
    assert user.use_status == Status.ACTIVE.value
    assert user.user_type == UserType.CUSTOMER.value
    assert isinstance(user.create_time, type(user.update_time))
    assert user.password == compute_hash("123456")


def test_update_time_touched_on_change(session):
    user = TUser(name="u")
    session.add(user)
    session.commit()
    session.refresh(user)
    before = user.update_time
    user.display_name = "改名"
    session.add(user)
    session.commit()
    session.refresh(user)
    assert user.update_time >= before


def test_unique_constraints(engine, session):
    ds = _dataset(session)
    session.add(TCategory(name="a", idx=0, dataset=ds))
    session.commit()
    with pytest.raises(Exception, match="UNIQUE"):
        session.add(TCategory(name="b", idx=0, dataset=ds))
        session.commit()
    session.rollback()

    session.add(TImg(name="a.jpg", dataset_id=ds.id))
    session.commit()
    with pytest.raises(Exception, match="UNIQUE"):
        session.add(TImg(name="a.jpg", dataset_id=ds.id))
        session.commit()
    session.rollback()


def test_foreign_keys_and_ondelete(engine):
    insp = inspect(engine)
    assert insp.get_foreign_keys("category")[0]["options"] == {"ondelete": "CASCADE"}
    assert insp.get_foreign_keys("model")[0]["options"] == {"ondelete": "SET NULL"}
    detect = {
        fk["referred_table"]: fk["options"]
        for fk in insp.get_foreign_keys("detect_anno")
    }
    assert detect == {
        "img": {"ondelete": "CASCADE"},
        "category": {"ondelete": "CASCADE"},
    }
    # 复合索引覆盖 img_id + category_id
    index_columns = [i["column_names"] for i in insp.get_indexes("detect_anno")]
    assert ["img_id", "category_id"] in index_columns


def test_foreign_keys_enforced(engine, session):
    """SQLite 默认关外键，引擎必须打开，否则孤立数据会悄悄写进去。"""
    session.add(TImg(name="x.jpg", dataset_id=12345))
    with pytest.raises(Exception, match="FOREIGN KEY"):
        session.commit()
    session.rollback()


# ============================================================
# 通用 CRUD
# ============================================================


def test_create_and_query(engine):
    user = TUser.create({"name": "shawn", "display_name": "Shawn"}, engine=engine)
    assert user.id is not None
    assert TUser.by_id(user.id, engine=engine).name == "shawn"
    assert TUser.by_id(99999, engine=engine) is None
    assert [u.name for u in TUser.all(engine=engine)] == ["shawn"]
    assert TUser.first(engine=engine).display_name == "Shawn"
    assert TUser.count(engine=engine) == 1


def test_create_requires_data(engine):
    with pytest.raises(ValueError):
        TUser.create(None, engine=engine)


def test_search_fuzzy_and_exact(engine):
    TUser.create({"name": "alice", "display_name": "爱丽丝"}, engine=engine)
    TUser.create({"name": "bob", "display_name": "鲍勃"}, engine=engine)
    assert [u.name for u in TUser.search("ali", engine=engine)] == ["alice"]
    assert [u.name for u in TUser.search("爱丽", engine=engine)] == ["alice"]
    assert [
        u.name for u in TUser.search("alice", is_full_match=True, engine=engine)
    ] == ["alice"]
    assert TUser.search("ali", is_full_match=True, engine=engine) == []
    assert len(TUser.search("alice", column_name="name", engine=engine)) == 1
    assert TUser.search("爱丽", column_name="name", engine=engine) == []
    assert len(TUser.search(None, engine=engine)) == 2


def test_search_session_and_save(engine, session):
    user = TUser(name="carol")
    assert user.save(session).id is not None
    assert [u.name for u in TUser.search_session(session, "car")] == ["carol"]


def test_session_scope_requires_engine_or_session():
    with pytest.raises(ValueError), session_scope():
        pass


# ============================================================
# 关系与级联
# ============================================================


def _tree(session: Session):
    user = TUser(name="u", display_name="负责人")
    session.add(user)
    session.commit()
    session.refresh(user)

    ps = TProjectSet(name="ps", head_user=user, creator_user_id=user.id)
    session.add(ps)
    session.commit()
    prj = TProject(name="p", project_set=ps, head_user=user)
    session.add(prj)
    session.commit()
    ds = TDataset(
        name="d", project=prj, head_user=user, task_type=TaskType.DETECT.value
    )
    session.add(ds)
    session.commit()
    session.refresh(ds)

    cat = TCategory(name="person", display_name="人", idx=0, dataset=ds)
    session.add(cat)
    session.commit()
    img = TImg(name="a.jpg", width=200, height=100, dataset=ds)
    session.add(img)
    session.commit()
    session.refresh(img)
    session.refresh(cat)
    return user, ps, prj, ds, cat, img


def test_bidirectional_relationships(session):
    user, ps, prj, ds, cat, img = _tree(session)
    session.expire_all()

    assert [x.name for x in session.get(TUser, user.id).head_project_sets] == ["ps"]
    assert [x.name for x in session.get(TUser, user.id).head_projects] == ["p"]
    assert [x.name for x in session.get(TUser, user.id).head_datasets] == ["d"]
    assert session.get(TProjectSet, ps.id).head_user.display_name == "负责人"
    assert session.get(TProjectSet, ps.id).creator_user_id == user.id
    assert [x.name for x in session.get(TProject, prj.id).datasets] == ["d"]
    assert session.get(TDataset, ds.id).project.name == "p"
    assert session.get(TImg, img.id).dataset.name == "d"
    assert session.get(TCategory, cat.id).dataset.name == "d"
    assert [x.name for x in session.get(TDataset, ds.id).imgs] == ["a.jpg"]


def test_many_to_many(session):
    user, _, prj, ds, _, _ = _tree(session)
    user.projects.append(prj)
    user.datasets.append(ds)
    session.add(user)
    session.commit()
    session.expire_all()
    assert [x.name for x in session.get(TUser, user.id).projects] == ["p"]
    assert [x.name for x in session.get(TProject, prj.id).users] == ["u"]
    assert [x.name for x in session.get(TDataset, ds.id).users] == ["u"]


def test_cascade_delete_dataset(session):
    _, _, _, ds, cat, img = _tree(session)
    session.add(TDetectAnno(x=1, y=1, width=10, height=10, img=img, category=cat))
    session.add(TSegmentAnno(points="[[1,1],[20,1],[20,20]]", img=img, category=cat))
    session.add(TClassifyAnno(conf=1.0, img=img, category=cat))
    session.add(TModel(name="m", dataset=ds))
    session.commit()

    ds_id, img_id = ds.id, img.id
    session.delete(session.get(TDataset, ds_id))
    session.commit()

    assert session.exec(select(TImg).where(TImg.dataset_id == ds_id)).all() == []
    assert (
        session.exec(select(TCategory).where(TCategory.dataset_id == ds_id)).all() == []
    )
    assert (
        session.exec(select(TDetectAnno).where(TDetectAnno.img_id == img_id)).all()
        == []
    )
    assert (
        session.exec(select(TSegmentAnno).where(TSegmentAnno.img_id == img_id)).all()
        == []
    )
    # 模型用 SET NULL：记录保留，权重不随数据集消失
    assert [(m.name, m.dataset_id) for m in session.exec(select(TModel)).all()] == [
        ("m", None)
    ]


def test_delete_user_sets_null(session):
    user, ps, prj, ds, _, _ = _tree(session)
    session.delete(session.get(TUser, user.id))
    session.commit()
    session.expire_all()
    assert session.get(TProjectSet, ps.id).head_user_id is None
    assert session.get(TProject, prj.id).head_user_id is None
    assert session.get(TDataset, ds.id).head_user_id is None
    assert session.get(TProjectSet, ps.id).name == "ps"


# ============================================================
# 几何 ↔ 实体
# ============================================================


def test_detect_anno_roundtrip(session):
    _, _, _, _ds, _cat, img = _tree(session)
    box = Box(Point(10, 20), Point(110, 70), category_idx=0, conf=0.9)
    anno = box.to_sql_model(img, session=session)
    session.add(anno)
    session.commit()
    session.refresh(anno)

    back = anno.anno
    assert (back.x_min, back.y_min, back.width, back.height) == pytest.approx(
        (10, 20, 100, 50)
    )
    assert back.conf == pytest.approx(0.9)
    assert back.category_idx == 0
    assert anno.to_str(ImgSize(200, 100)) == box.to_yolo(ImgSize(200, 100))
    assert anno.conf == pytest.approx(0.9)


def test_segment_anno_roundtrip(session):
    _, _, _, _, _, img = _tree(session)
    poly = Polygon(
        (Point(10, 10), Point(60, 10), Point(60, 40), Point(10, 40)),
        category_idx=0,
        conf=0.7,
    )
    anno = poly.to_sql_model(img, session=session)
    session.add(anno)
    session.commit()
    session.refresh(anno)

    assert anno.point_list == list(poly.points)
    assert anno.anno.points == poly.points
    assert anno.to_str(SIZE) == poly.to_yolo(SIZE)
    assert anno.anno.area == pytest.approx(poly.area)


def test_classify_anno(session):
    _, _, _, _, cat, img = _tree(session)
    anno = TClassifyAnno(conf=2.0, img=img, category=cat)
    session.add(anno)
    session.commit()
    session.refresh(anno)
    point = anno.anno
    assert (point.x, point.y) == (0, 0)
    assert point.category_idx == 0
    assert point.conf == pytest.approx(2.0)


def test_pack_unpack_points():
    points = [Point(1.23456, 2.0), Point(3, 4)]
    packed = pack_points(points)
    assert unpack_points(packed) == [Point(1.235, 2.0), Point(3, 4)]
    assert unpack_points("") == []
    assert unpack_points(None) == []


def test_segment_from_polygon():
    poly = Polygon((Point(0, 0), Point(5, 0), Point(5, 5)), conf=0.5)
    anno = TSegmentAnno.from_polygon(poly)
    assert anno.conf == pytest.approx(0.5)
    assert anno.point_list == list(poly.points)


def test_img_size_property(session):
    _, _, _, _, _, img = _tree(session)
    assert img.img_size == ImgSize(200, 100)


# ============================================================
# 数据集辅助
# ============================================================


def test_dataset_paths():
    ds = TDataset(name="d")
    assert ds.img_dir == Path("images")
    assert ds.anno_db == Path("anno.db")
    assert ds.anno_dir == Path("annos")
    assert ds.model_dir == Path("models")
    assert ds.cache_dir == Path("cache")
    assert ds.data_yaml == Path("dataset_config.yaml")

    ds.file_dir = "output/ds"
    assert ds.img_dir == Path("output/ds/images")
    assert ds.anno_db == Path("output/ds/anno.db")


def test_dataset_category_names(session):
    _, _, _, ds, _, _ = _tree(session)
    session.add(TCategory(name="bus", display_name="公交车", idx=1, dataset=ds))
    session.commit()
    session.expire_all()
    dataset = session.get(TDataset, ds.id)
    assert dataset.get_category_names(session) == ["人", "公交车"]
    assert dataset.get_category_name_list(session) == ["人", "公交车"]
    assert TDataset.by_id(99999, session=session) is None


def test_dataset_missing_returns_empty(session):
    ghost = TDataset(id=424242, name="ghost")
    assert ghost.get_category_names(session) == []


# ============================================================
# 口令
# ============================================================


def test_compute_hash():
    assert compute_hash("123456") == compute_hash("123456")
    assert compute_hash("123456") != compute_hash("1234567")
    assert len(compute_hash("x")) == 64


@pytest.mark.parametrize(
    ("password", "ok"),
    [
        ("Abcdef1!", True),
        ("abcdef1!", False),
        ("ABCDEF1!", False),
        ("Abcdefg!", False),
        ("Abcd1", False),
    ],
)
def test_password_validator(password, ok):
    if ok:
        assert TUser.password_validator(password) == password
    else:
        with pytest.raises(ValueError, match="密码需要"):
            TUser.password_validator(password)


# ============================================================
# 引擎
# ============================================================


def test_memory_engine_has_all_tables(engine):
    assert set(SQLModel.metadata.tables) >= {"user", "img", "detect_anno"}
    with engine.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1


def test_file_engine_creates_parent_and_wal(tmp_path: Path):
    db_file = tmp_path / "nested" / "anno.db"
    eng = sqlite_engine(db_file)
    create_tables(eng)
    with Session(eng) as sess:
        TUser.create({"name": "file-user"}, session=sess)
        assert TUser.count(session=sess) == 1
    assert db_file.exists()
    with eng.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"
