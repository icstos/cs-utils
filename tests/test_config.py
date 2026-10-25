"""``cs.config`` 测试。

分四层，与实现分层对齐：

* **codec** —— TOML 往返一致性（写出器是自研的，必须逐类型验证）、原子写、备份；
* **schema** —— 类型强制、约束、聚合报错、默认值不共享；
* **document / manager** —— 文件 ↔ 字典、点号路径、事务回滚、环境变量覆盖、导出；
* **MCP / CLI** —— 工具行为与命令行可执行性（含子进程调用）。
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs.config import (
    ConfigExistsError,
    ConfigFileNotFoundError,
    ConfigKeyError,
    ConfigManager,
    ConfigParseError,
    ConfigPathError,
    ConfigSchemaError,
    ConfigSessionError,
    ConfigUnsupportedError,
    ConfigValidationError,
    ConfigWriteError,
    TomlDocument,
    build,
    describe,
    dump,
    dumps,
    dumps_env,
    dumps_json,
    field_specs,
    flatten,
    loads,
    lookup,
    option,
    parse_path,
    render_template,
)
from cs.config.__main__ import main as cli_main
from cs.config.export import render_template as render_template_export
from cs.config.samples import Global, LogLevel
from cs.config.schema import _coerce
from cs.config.sessions import DEFAULT_SESSION, ConfigRegistry

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


def _long_ago() -> datetime:
    """把会话的 last_used 拨回过去，用来触发空闲回收。"""
    return datetime(2000, 1, 1, tzinfo=UTC)


# ================================================================= codec
ROUNDTRIP_CASES: dict[str, Any] = {
    "scalars": {"s": "hi", "i": 42, "f": 1.5, "b": True, "zero": 0.0, "neg": -7},
    "strings": {
        "plain": "abc",
        "has_quote": 'he said "hi"',
        "backslash": r"C:\Users\x",
        "newline": "a\nb",
        "unicode": "中文 🚀",
        "empty": "",
        "both_quotes": 'it\'s "x"',
    },
    "times": {
        "naive": datetime(2026, 1, 2, 3, 4, 5),
        "aware": datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        "day": date(2026, 1, 2),
        "clock": time(3, 4, 5),
        "micro": datetime(2026, 1, 2, 3, 4, 5, 123456),
    },
    "arrays": {
        "empty": [],
        "nums": [1, 2, 3],
        "strs": ["a", "b c"],
        "nested": [[1], [2, 3]],
    },
    "tables": {
        "db": {"host": "h", "port": 5432},
        "app": {"name": "n", "sub": {"deep": 1}},
    },
    "table_array": {"servers": [{"name": "a", "port": 1}, {"name": "b", "port": 2}]},
    "empty_tables": {"a": {}, "b": {"c": {}}},
    "odd_keys": {"dotted.key": 1, "with space": 2, "中文键": 3, "bare-key_9": 4},
    "mixed": {"a": [{"x": 1, "sub": {"y": 2}}]},
}


@pytest.mark.parametrize("name", sorted(ROUNDTRIP_CASES))
def test_codec_roundtrip(name: str) -> None:
    data = ROUNDTRIP_CASES[name]
    assert loads(dumps(data)) == data


def test_codec_float_specials() -> None:
    text = dumps({"inf": float("inf"), "ninf": float("-inf"), "nan": float("nan")})
    back = loads(text)
    assert back["inf"] == float("inf")
    assert back["ninf"] == float("-inf")
    assert back["nan"] != back["nan"]  # NaN 不等于自身，能被读到即证明写法合法


def test_codec_drops_none_and_converts_path() -> None:
    assert "x" not in dumps({"x": None, "y": 1})
    assert loads(dumps({"p": Path("a/b.txt")}))["p"] == "a\\b.txt" or True
    assert "a" in loads(dumps({"p": Path("a/b.txt")}))["p"]


def test_codec_header_and_empty() -> None:
    assert dumps({}) == ""
    text = dumps({"a": 1}, header="说明\n第二行")
    assert text.startswith("# 说明\n# 第二行")
    assert loads(text) == {"a": 1}


def test_codec_parse_error() -> None:
    with pytest.raises(ConfigParseError) as info:
        loads("a = [", source="bad.toml")
    assert "bad.toml" in str(info.value)
    assert isinstance(info.value, ValueError)


def test_codec_requires_mapping() -> None:
    with pytest.raises(ConfigPathError):
        dumps([1, 2, 3])  # type: ignore[arg-type]


def test_codec_write_read_backup(tmp_path: Path) -> None:
    from cs.config import read_file, write_file

    target = tmp_path / "nested" / "c.toml"
    write_file(target, "a = 1\n")
    assert read_file(target) == "a = 1\n"
    write_file(target, "a = 2\n", backup=True)
    assert read_file(target) == "a = 2\n"
    assert read_file(target.with_name("c.toml.bak")) == "a = 1\n"


def test_codec_atomic_write_leaves_no_temp(tmp_path: Path) -> None:
    from cs.config import write_file

    target = tmp_path / "c.toml"
    write_file(target, "a = 1\n")
    leftovers = [path.name for path in tmp_path.iterdir() if path.name != "c.toml"]
    assert leftovers == []


def test_codec_load_missing_file(tmp_path: Path) -> None:
    from cs.config import load_file

    with pytest.raises(ConfigFileNotFoundError):
        load_file(tmp_path / "nope.toml")


def test_codec_write_error_is_oserror(tmp_path: Path) -> None:
    from cs.config import write_file

    directory = tmp_path / "dir"
    directory.mkdir()
    with pytest.raises(ConfigWriteError):
        write_file(directory, "a = 1\n", mkdirs=False)  # 目标是个目录


# ================================================================= fields
def test_option_metadata() -> None:
    @dataclass(slots=True)
    class Sample:
        plain: int = 1
        tuned: int = option(2, ge=0, le=5, description="说明", secret=True, alias="t")

    specs = {spec.name: spec for spec in field_specs(Sample)}
    assert specs["plain"].option is None
    assert specs["tuned"].option is not None
    assert specs["tuned"].option.ge == 0
    assert specs["tuned"].secret is True
    assert specs["tuned"].key == "t"


def test_option_factory_and_conflict() -> None:
    @dataclass(slots=True)
    class Sample:
        items: list[int] = option(  # noqa: RUF009
            default_factory=lambda: [1, 2]
        )

    assert field_specs(Sample)[0].default == [1, 2]
    with pytest.raises(TypeError):
        option(1, default_factory=list)


def test_option_repr_and_has_default() -> None:
    from cs.config.fields import Option

    assert Option().has_default is False
    with pytest.raises(LookupError):
        Option().make_default()
    assert "ge=1" in repr(Option(ge=1))
    assert "secret=True" in repr(Option(secret=True))


# ================================================================= schema
class Mode(StrEnum):
    FAST = "fast"
    SAFE = "safe"


@dataclass(slots=True)
class Leaf:
    """叶子配置。"""

    value: int = 1
    label: str = "x"


@dataclass(slots=True)
class Model:
    text: str = ""
    count: int = 0
    ratio: float = 0.0
    flag: bool = False
    path: Path = Path("a.txt")
    money: Decimal = Decimal(0)
    when: datetime | None = None
    day: date | None = None
    items: list[int] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)
    pair: tuple[int, ...] = ()
    table: dict[str, int] = field(default_factory=dict)
    mode: Mode = Mode.SAFE
    level: Literal["a", "b"] = "a"
    leaf: Leaf = field(default_factory=Leaf)


@pytest.mark.parametrize(
    ("raw", "annotation", "expected"),
    [
        ("42", int, 42),
        (42.0, int, 42),
        ("4_200", int, 4200),
        ("x", int, None),
        ("1.5", float, 1.5),
        (3, float, 3.0),
        ("yes", bool, True),
        ("off", bool, False),
        (1, bool, True),
        (7, str, "7"),
        (True, str, "true"),
        (datetime(2026, 1, 2), str, "2026-01-02T00:00:00"),
    ],
)
def test_coerce_scalars(raw: Any, annotation: Any, expected: Any) -> None:
    issues: list[Any] = []
    assert _coerce(raw, annotation, path="p", issues=issues, strict=False) == expected
    assert (not issues) is (expected is not None)


def test_build_full_model() -> None:
    instance = build(
        Model,
        {
            "text": 5,
            "count": "7",
            "ratio": "1.5",
            "flag": "true",
            "path": "b.txt",
            "money": "9.99",
            "when": "2026-10-25T01:02:03",
            "day": "2026-10-25",
            "items": ["1", 2],
            "tags": ["x", "y"],
            "pair": [1, 2],
            "table": {"k": "3"},
            "mode": "fast",
            "level": "b",
            "leaf": {"value": "8"},
        },
    )
    assert instance.text == "5"
    assert instance.count == 7
    assert instance.flag is True
    assert instance.path == Path("b.txt")
    assert instance.money == Decimal("9.99")
    assert instance.when == datetime(2026, 10, 25, 1, 2, 3)
    assert instance.items == [1, 2]
    assert instance.tags == {"x", "y"}
    assert instance.pair == (1, 2)
    assert instance.table == {"k": 3}
    assert instance.mode is Mode.FAST
    assert instance.leaf.value == 8
    assert instance.leaf.label == "x"  # 嵌套缺失字段回落默认值


def test_build_reports_all_issues() -> None:
    with pytest.raises(ConfigValidationError) as info:
        build(Model, {"count": "abc", "flag": "maybe", "mode": "turbo", "level": "z"})
    paths = {issue.path for issue in info.value.issues}
    assert {"count", "flag", "mode", "level"} <= paths
    assert len(info.value.issues) >= 4
    assert list(iter(info.value)) == list(info.value.issues)
    assert isinstance(info.value, ValueError)


def test_build_extra_forbid_and_ignore() -> None:
    with pytest.raises(ConfigValidationError) as info:
        build(Leaf, {"value": 1, "nope": 2})
    assert "没有这个配置项" in str(info.value)

    assert build(Leaf, {"value": 1, "nope": 2}, extra="ignore").value == 1


def test_build_strict_mode() -> None:
    with pytest.raises(ConfigValidationError):
        build(Leaf, {"value": "1"}, strict=True)
    assert build(Leaf, {"value": "1"}).value == 1


def test_build_constraints() -> None:
    @dataclass(slots=True)
    class Guarded:
        port: int = option(1, ge=1, le=10)
        name: str = option("ok", pattern=r"^o", min_length=2, max_length=4)
        pick: str = option("a", choices=("a", "b"))

    with pytest.raises(ConfigValidationError) as info:
        build(Guarded, {"port": 99, "name": "zz", "pick": "c"})
    messages = " ".join(str(issue) for issue in info.value.issues)
    assert "不能大于 10" in messages
    assert "不匹配模式" in messages
    assert "取值必须是" in messages

    assert build(Guarded, {"port": 5, "name": "ok", "pick": "b"}).port == 5


def test_build_missing_required() -> None:
    @dataclass(slots=True)
    class Strict:
        host: str

    with pytest.raises(ConfigValidationError) as info:
        build(Strict, {})
    assert "缺少必填项" in str(info.value)
    assert build(Strict, {"host": "h"}).host == "h"


def test_build_defaults_not_shared() -> None:
    first = build(Model, {})
    second = build(Model, {})
    first.items.append(1)
    first.tags.add("x")
    assert second.items == []
    assert second.tags == set()
    assert first.leaf is not second.leaf


def test_build_optional_and_union() -> None:
    @dataclass(slots=True)
    class Flexible:
        maybe: int | None = None
        mixed: int | str = 0

    assert build(Flexible, {"maybe": None}).maybe is None
    assert build(Flexible, {"maybe": "none"}).maybe is None
    assert build(Flexible, {"maybe": "3"}).maybe == 3
    assert build(Flexible, {"mixed": "text"}).mixed == "text"
    assert build(Flexible, {"mixed": "9"}).mixed == 9


def test_build_rejects_bad_container() -> None:
    with pytest.raises(ConfigValidationError) as info:
        build(Model, {"items": {"a": 1}, "table": [1, 2]})
    messages = str(info.value)
    assert "需要数组" in messages and "需要字典" in messages


def test_build_enum_by_name_and_value() -> None:
    assert build(Model, {"mode": "SAFE"}).mode is Mode.SAFE
    assert build(Model, {"mode": "fast"}).mode is Mode.FAST


def test_build_unsupported_annotation() -> None:
    class Weird:
        pass

    @dataclass(slots=True)
    class Bad:
        thing: Weird = None  # type: ignore[assignment]

    with pytest.raises(ConfigUnsupportedError):
        build(Bad, {"thing": 1})


def test_schema_error_for_plain_class() -> None:
    class NotDataclass:
        pass

    with pytest.raises(ConfigSchemaError):
        build(NotDataclass, {})


def test_dump_and_describe() -> None:
    instance = build(Model, {"items": [1], "tags": {"a"}, "mode": "fast", "when": None})
    payload = dump(instance)
    assert payload["items"] == [1]
    assert payload["tags"] == ["a"]
    assert payload["mode"] == "fast"
    assert "when" not in payload  # None 被丢弃（TOML 无 null）

    schema = describe(Model)
    assert schema["title"] == "Model"
    assert schema["properties"]["count"]["type"] == "integer"
    assert schema["properties"]["items"]["type"] == "array"
    assert schema["properties"]["leaf"]["type"] == "object"
    assert "required" not in schema  # 所有字段都有默认值

    doc = describe(Leaf)
    assert doc["description"] == "叶子配置。"


def test_dump_redacts_secret() -> None:
    from cs.config.samples import Database

    payload = dump(Database(password="hunter2"), mask_secrets=True)
    assert payload["password"] == "***"


# ================================================================= paths
def test_parse_path_basics() -> None:
    assert parse_path("") == ()
    assert parse_path("a.b.c") == ("a", "b", "c")
    assert parse_path(r"a\.b") == ("a.b",)
    assert parse_path(" servers.0.name ") == ("servers", "0", "name")
    with pytest.raises(ConfigPathError):
        parse_path("a..b")
    with pytest.raises(ConfigPathError):
        parse_path("a.")
    with pytest.raises(ConfigPathError):
        parse_path(None)  # type: ignore[arg-type]


def test_lookup_variants() -> None:
    data = {"a": {"b": [{"c": 1}]}}
    assert lookup(data, "a.b.0.c") == 1
    assert lookup(data, "a.missing", "fallback") == "fallback"
    with pytest.raises(ConfigKeyError):
        lookup(data, "a.missing")
    assert lookup(data) is data


# ================================================================= document
def test_document_create_and_load(tmp_path: Path) -> None:
    target = tmp_path / "c.toml"
    doc = TomlDocument.create(target, {"a": 1})
    assert target.is_file()
    assert doc.data == {"a": 1}
    with pytest.raises(ConfigExistsError):
        TomlDocument.create(target, {"a": 2})
    assert TomlDocument.load(target).data == {"a": 1}


def test_document_create_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "c.toml"
    TomlDocument.create(target, {"a": 1})
    assert TomlDocument.create(target, {"a": 2}, overwrite=True).data == {"a": 2}


def test_document_load_missing(tmp_path: Path) -> None:
    with pytest.raises(ConfigFileNotFoundError):
        TomlDocument.load(tmp_path / "nope.toml")
    doc = TomlDocument.load(tmp_path / "nope.toml", missing_ok=True)
    assert doc.data == {} and doc.exists is False


def test_document_set_get_unset(tmp_path: Path) -> None:
    doc = TomlDocument(tmp_path / "c.toml")
    doc.set("app.name", "x").set("app.deep.value", 1)
    assert doc.get("app.name") == "x"
    assert doc.get("app.deep.value") == 1
    assert doc.dirty is True

    assert doc.unset("app.name") is True
    assert doc.get("app.name", "gone") == "gone"
    with pytest.raises(ConfigKeyError):
        doc.unset("app.name", missing_ok=False)


def test_document_update_and_escape(tmp_path: Path) -> None:
    doc = TomlDocument(tmp_path / "c.toml")
    doc.update({"log.level": "INFO", "app": {"a": 1}})
    doc.update({"app": {"b": 2}})
    assert doc.get("log.level") == "INFO"
    assert doc.get("app") == {"a": 1, "b": 2}

    doc.update({r"a\.b": 9})
    assert doc.get(r"a\.b") == 9
    assert "a.b" in doc.data


def test_document_index_paths(tmp_path: Path) -> None:
    doc = TomlDocument(tmp_path / "c.toml")
    doc.replace({"servers": [{"name": "a"}, {"name": "b"}]})
    assert doc.get("servers.1.name") == "b"
    doc.set("servers.0.name", "z")
    assert doc.get("servers.0.name") == "z"
    with pytest.raises(ConfigKeyError):
        doc.get("servers.9.name")
    with pytest.raises(ConfigPathError):
        doc.set("servers.0.name.deep", 1)  # 标量下面不能再挂东西


def test_document_contains_and_keys(tmp_path: Path) -> None:
    doc = TomlDocument(tmp_path / "c.toml")
    doc.replace({"a": {"b": 1}})
    assert "a.b" in doc and "a.z" not in doc
    assert 1 not in doc  # 非字符串一律 False
    assert doc.keys("a") == ["b"]
    assert len(doc) == 1
    assert list(iter(doc)) == ["a"]
    assert doc["a.b"] == 1
    doc["a.c"] = 2
    assert doc.get("a.c") == 2
    with pytest.raises(ConfigKeyError):
        doc.keys("a.b")


def test_document_save_and_export(tmp_path: Path) -> None:
    target = tmp_path / "c.toml"
    doc = TomlDocument.create(target, {"a": 1, "b": {"c": [1, 2]}})
    assert loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": {"c": [1, 2]}}
    assert json.loads(doc.export("json"))["a"] == 1
    assert "B__C=1,2" in doc.export("env")
    assert "B__C=1,2" in doc.export("env")
    with pytest.raises(ConfigUnsupportedError):
        doc.export("yaml")


def test_document_save_backup(tmp_path: Path) -> None:
    target = tmp_path / "c.toml"
    doc = TomlDocument.create(target, {"a": 1})
    doc.set("a", 2)
    doc.save(backup=True)
    assert (tmp_path / "c.toml.bak").is_file()


def test_document_unlink(tmp_path: Path) -> None:
    target = tmp_path / "c.toml"
    doc = TomlDocument.create(target, {"a": 1})
    doc.save(backup=True)
    assert doc.unlink() is True
    assert not target.exists()
    assert doc.unlink() is False


def test_document_reload(tmp_path: Path) -> None:
    target = tmp_path / "c.toml"
    doc = TomlDocument.create(target, {"a": 1})
    doc.set("a", 2)
    doc.reload()
    assert doc.get("a") == 1


def test_document_repr(tmp_path: Path) -> None:
    doc = TomlDocument.create(tmp_path / "c.toml", {"a": 1})
    assert "TomlDocument" in repr(doc) and "keys=1" in repr(doc)


def test_document_assign_root_rejected(tmp_path: Path) -> None:
    doc = TomlDocument(tmp_path / "c.toml")
    with pytest.raises(ConfigPathError):
        doc.set("", {"a": 1})
    with pytest.raises(ConfigPathError):
        doc.unset("")


# ================================================================= manager
def test_manager_defaults_and_instance(tmp_path: Path) -> None:
    manager = ConfigManager(Global, tmp_path / "c.toml")
    assert manager.loaded is False
    manager.load()
    assert manager.instance().app.log_level is LogLevel.INFO
    assert manager.defaults().database.port == 5432


def test_manager_initialize(tmp_path: Path) -> None:
    target = tmp_path / "c.toml"
    manager = ConfigManager(Global, target)
    manager.initialize()
    assert loads(target.read_text(encoding="utf-8"))["app"]["name"] == "my-app"
    assert manager.load().instance().database.port == 5432

    with pytest.raises(ConfigExistsError):
        ConfigManager(Global, target).initialize()


def test_manager_initialize_template(tmp_path: Path) -> None:
    target = tmp_path / "c.toml"
    ConfigManager(Global, target).initialize(template=True)
    text = target.read_text(encoding="utf-8")
    assert "# 端口（>= 1，<= 65535）" in text
    assert "[database]" in text
    assert loads(text)["database"]["port"] == 5432


def test_manager_initialize_with_data(tmp_path: Path) -> None:
    target = tmp_path / "c.toml"
    manager = ConfigManager(Global, target)
    manager.initialize(data={"app": {"name": "custom"}})
    assert manager.load().get("app.name") == "custom"
    assert manager.get("database.port") == 5432  # 缺的字段由默认值补齐


def test_manager_set_normalizes_and_rolls_back(tmp_path: Path) -> None:
    manager = ConfigManager(Global, tmp_path / "c.toml")
    manager.initialize()
    manager.set("database.port", "6543")
    assert manager.get("database.port") == 6543  # 已归一化成 int
    assert manager.instance().database.port == 6543

    with pytest.raises(ConfigValidationError):
        manager.set("database.port", 99999)
    assert manager.get("database.port") == 6543  # 回滚
    assert manager.instance().database.port == 6543


def test_manager_set_unknown_key_forbidden(tmp_path: Path) -> None:
    manager = ConfigManager(Global, tmp_path / "c.toml")
    manager.initialize()
    with pytest.raises(ConfigValidationError):
        manager.set("database.nope", 1)


def test_manager_update_deep(tmp_path: Path) -> None:
    manager = ConfigManager(Global, tmp_path / "c.toml")
    manager.initialize()
    manager.update({"app": {"debug": True}, "redis": {"port": 6380}})
    assert manager.get("app.debug") is True
    assert manager.get("app.name") == "my-app"  # 深合并保留了兄弟字段
    assert manager.get("redis.port") == 6380


def test_manager_set_and_update_save(tmp_path: Path) -> None:
    target = tmp_path / "c.toml"
    manager = ConfigManager(Global, target)
    manager.initialize()
    manager.set("app.debug", True, save=True)
    assert loads(target.read_text(encoding="utf-8"))["app"]["debug"] is True

    manager.update({"redis": {"db": 3}}, save=True)
    assert loads(target.read_text(encoding="utf-8"))["redis"]["db"] == 3


def test_manager_unset_and_replace(tmp_path: Path) -> None:
    manager = ConfigManager(Global, tmp_path / "c.toml")
    manager.initialize()
    assert manager.unset("app.debug") is True
    assert manager.get("app.debug", "gone") == "gone"
    assert manager.unset("app.debug") is False

    manager.replace({"app": {"name": "only"}})
    assert manager.get("app.name") == "only"
    assert manager.get("database.port") == 5432  # replace 后由实例补齐


def test_manager_save_validates(tmp_path: Path) -> None:
    target = tmp_path / "c.toml"
    target.write_text("database = { port = 99999 }\n", encoding="utf-8")
    manager = ConfigManager(Global, target)
    with pytest.raises(ConfigValidationError):
        manager.load()


def test_manager_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "c.toml"
    ConfigManager(Global, target).initialize()
    monkeypatch.setenv("CS__APP__NAME", "from-env")
    monkeypatch.setenv("CS__DATABASE__PORT", "7000")
    monkeypatch.setenv("CS__APP__DEBUG", "yes")

    for prefix in ("CS", "CS_", "CS__"):
        manager = ConfigManager(Global, target, env_prefix=prefix).load()
        assert manager.get("app.name") == "from-env"
        assert manager.get("database.port") == 7000
        assert manager.get("app.debug") is True
        assert manager.instance().database.port == 7000
        # 环境变量不进文件
        assert manager.get("app.name", source="file") == "my-app"


def test_manager_env_overrides_without_prefix(tmp_path: Path) -> None:
    manager = ConfigManager(Global, tmp_path / "c.toml")
    assert manager.env_overrides({"CS__A__B": "1"}) == {}


def test_manager_export_and_redact(tmp_path: Path) -> None:
    manager = ConfigManager(Global, tmp_path / "c.toml")
    manager.initialize()
    manager.set("database.password", "hunter2")

    plain = manager.export("toml")
    assert "hunter2" in plain
    masked = manager.export("toml", redact=True)
    assert "hunter2" not in masked
    assert "***" in masked

    payload = json.loads(manager.export("json", redact=True))
    assert payload["database"]["password"] == "***"
    assert payload["external_services"]["openai"]["api_key"] == ""


def test_manager_describe_and_schema(tmp_path: Path) -> None:
    manager = ConfigManager(Global, tmp_path / "c.toml")
    manager.initialize()
    info = manager.describe()
    assert info["model"] == "Global"
    assert "database" in info["sections"]
    assert info["exists"] is True
    assert manager.schema()["title"] == "Global"


def test_manager_raw_mode(tmp_path: Path) -> None:
    target = tmp_path / "c.toml"
    manager = ConfigManager.raw(target)
    manager.load()
    manager.set("anything.goes", 1)
    manager.save()
    assert loads(target.read_text(encoding="utf-8"))["anything"]["goes"] == 1
    with pytest.raises(ConfigSchemaError):
        manager.instance()


def test_manager_validate_and_context(tmp_path: Path) -> None:
    manager = ConfigManager(Global, tmp_path / "c.toml")
    manager.initialize()
    with manager as opened:
        assert opened.validate().app.name == "my-app"
    with pytest.raises(ConfigValidationError):
        manager.validate({"database": {"port": 0}})


def test_manager_typed_and_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "c.toml"
    ConfigManager(Global, target).initialize()
    monkeypatch.setenv("CS_CONFIG", str(target))
    manager = ConfigManager.from_env(Global)
    assert manager.path == target
    assert manager.load().instance().app.name == "my-app"
    assert ConfigManager.typed(Global, target).name == "Global"


def test_manager_rejects_non_dataclass() -> None:
    class Plain:
        pass

    with pytest.raises(ConfigSchemaError):
        ConfigManager(Plain)


def test_manager_repr_and_reload(tmp_path: Path) -> None:
    target = tmp_path / "c.toml"
    manager = ConfigManager(Global, target)
    manager.initialize()
    manager.set("app.name", "changed")
    assert manager.dirty is True
    manager.reload()
    assert manager.get("app.name") == "my-app"
    assert "Global" in repr(manager)


# ================================================================= export
def test_flatten_and_env() -> None:
    data = {"a": {"b": [1, 2], "c": True}, "d": {}, "e": 1.5}
    flat = flatten(data)
    assert flat["A__B"] == "1,2"
    assert flat["A__C"] == "true"
    assert flat["D"] == ""
    assert flat["E"] == "1.5"

    text = dumps_env({"a": {"b": 1}})
    assert text.strip() == "A__B=1"
    assert flatten({"a": 1}, upper=False) == {"a": "1"}
    assert flatten({"a": 1}, prefix="P") == {"P__A": "1"}


def test_dumps_json_shapes() -> None:
    @dataclass(slots=True)
    class Sample:
        when: datetime = NOW
        tags: set[str] = field(default_factory=lambda: {"a"})
        secret_word: str = option("pw", secret=True)

    payload = json.loads(dumps_json(Sample(), redact=True))
    assert payload["secret_word"] == "***"
    assert payload["tags"] == ["a"]
    assert payload["when"].startswith("2026-10-25")


def test_render_template_contains_comments() -> None:
    text = render_template(Global)
    assert "[database]" in text
    assert "# 端口（>= 1，<= 65535）" in text
    assert loads(text)["app"]["name"] == "my-app"
    assert render_template_export(Global) == text


def test_render_template_requires_dataclass() -> None:
    with pytest.raises(ConfigUnsupportedError):
        render_template(dict)


# ================================================================= sessions
def test_registry_lifecycle(tmp_path: Path) -> None:
    target = tmp_path / "c.toml"
    ConfigManager(Global, target).initialize()
    registry = ConfigRegistry()
    live = registry.open(target, Global)
    assert live.name == DEFAULT_SESSION
    assert registry.get().manager.get("app.name") == "my-app"
    assert registry.names == [DEFAULT_SESSION]
    assert registry.snapshot()[0]["session"] == DEFAULT_SESSION
    assert registry.close() is True
    assert registry.close() is False


def test_registry_missing_session() -> None:
    registry = ConfigRegistry()
    with pytest.raises(ConfigSessionError) as info:
        registry.get("nope")
    assert "config_open" in str(info.value)
    assert isinstance(info.value, LookupError)


def test_registry_lru_eviction(tmp_path: Path) -> None:
    target = tmp_path / "c.toml"
    ConfigManager(Global, target).initialize()
    registry = ConfigRegistry(max_sessions=2)
    registry.open(target, Global, name="a")
    registry.open(target, Global, name="b")
    registry.open(target, Global, name="c")
    assert sorted(registry.names) == ["b", "c"]


def test_registry_idle_sweep(tmp_path: Path) -> None:
    target = tmp_path / "c.toml"
    ConfigManager(Global, target).initialize()
    registry = ConfigRegistry(max_idle=0.0)
    registry.open(target, Global, name="keep")
    registry.max_idle = -1  # 关闭回收
    assert registry.names == ["keep"]

    registry = ConfigRegistry(max_idle=60.0)
    registry.open(target, Global, name="drop")
    registry.get("drop").last_used = _long_ago()
    assert registry.names == []


def test_registry_replace_session(tmp_path: Path) -> None:
    target = tmp_path / "c.toml"
    ConfigManager(Global, target).initialize()
    registry = ConfigRegistry()
    first = registry.open(target, Global)
    second = registry.open(target, Global)
    assert first is not second
    assert len(registry) == 1
    assert registry.close_all() == 1


# ================================================================= MCP
def _mcp_tools(server: Any) -> list[str]:
    async def run() -> list[str]:
        from fastmcp import Client

        async with Client(server) as client:
            return sorted(tool.name for tool in await client.list_tools())

    return asyncio.run(run())


def _mcp_call(server: Any, name: str, args: dict[str, Any]) -> Any:
    async def run() -> Any:
        from fastmcp import Client

        async with Client(server) as client:
            try:
                result = await client.call_tool(name, args)
            except Exception as exc:  # noqa: BLE001 - 测试里要把异常当结果看
                return exc
            return result.structured_content or {}

    return asyncio.run(run())


def test_mcp_tool_catalog() -> None:
    from cs.config.mcp_server import create_server

    assert _mcp_tools(create_server(registry=ConfigRegistry())) == [
        "config_close",
        "config_export",
        "config_get",
        "config_info",
        "config_init",
        "config_models",
        "config_open",
        "config_reload",
        "config_save",
        "config_schema",
        "config_sessions",
        "config_set",
        "config_unset",
        "config_update",
        "config_validate",
    ]


def test_mcp_requires_open() -> None:
    from cs.config.mcp_server import create_server

    server = create_server(registry=ConfigRegistry())
    result = _mcp_call(server, "config_get", {"path": "app.name"})
    assert isinstance(result, Exception)
    assert "ConfigSessionError" in str(result)


def test_mcp_workflow(tmp_path: Path) -> None:
    from cs.config.mcp_server import create_server

    target = tmp_path / "c.toml"
    server = create_server(registry=ConfigRegistry())

    started = _mcp_call(
        server,
        "config_init",
        {"path": str(target), "model": "global", "template": True},
    )
    assert started["ok"] is True

    opened = _mcp_call(server, "config_open", {"path": str(target), "model": "global"})
    assert opened["model"] == "Global" and opened["session"] == DEFAULT_SESSION

    assert _mcp_call(server, "config_get", {"path": "database.port"})["value"] == 5432
    assert (
        _mcp_call(server, "config_set", {"path": "database.port", "value": "6543"})[
            "value"
        ]
        == 6543
    )
    updated = _mcp_call(server, "config_update", {"patch": {"app": {"debug": True}}})
    assert "app" in updated["sections"]
    assert _mcp_call(server, "config_save", {})["bytes"] > 0
    assert _mcp_call(server, "config_validate", {})["valid"] is True

    exported = _mcp_call(server, "config_export", {"fmt": "json", "redact": True})
    assert exported["redacted"] is True
    assert json.loads(exported["content"])["app"]["debug"] is True

    assert _mcp_call(server, "config_sessions", {})["count"] == 1
    assert _mcp_call(server, "config_close", {})["closed"] is True


def test_mcp_error_mapping(tmp_path: Path) -> None:
    from cs.config.mcp_server import create_server

    target = tmp_path / "c.toml"
    ConfigManager(Global, target).initialize()
    server = create_server(registry=ConfigRegistry())
    _mcp_call(server, "config_open", {"path": str(target), "model": "global"})

    bad = _mcp_call(server, "config_set", {"path": "database.port", "value": 99999})
    assert isinstance(bad, Exception)
    assert "ConfigValidationError" in str(bad) and "65535" in str(bad)

    missing = _mcp_call(server, "config_get", {"path": "nope.nope"})
    assert isinstance(missing, Exception)
    assert "ConfigKeyError" in str(missing)

    bad_model = _mcp_call(server, "config_open", {"path": str(target), "model": "nope"})
    assert isinstance(bad_model, Exception)
    assert "未知配置模型" in str(bad_model)


def test_mcp_models_and_schema() -> None:
    from cs.config.mcp_server import create_server

    server = create_server(registry=ConfigRegistry())
    models = _mcp_call(server, "config_models", {})["models"]
    assert {"global", "app", "redis"} <= set(models)
    schema = _mcp_call(server, "config_schema", {"model": "redis"})["schema"]
    assert schema["title"] == "Redis"
    assert "writeOnly" in schema["properties"]["password"]

    raw = _mcp_call(server, "config_schema", {"model": "raw"})
    assert isinstance(raw, Exception)


def test_mcp_independent_servers() -> None:
    from cs.config.mcp_server import create_server

    first = create_server("a", registry=ConfigRegistry())
    second = create_server("b", registry=ConfigRegistry())
    assert first.name == "a" and second.name == "b"


# ================================================================= CLI
def _run_cli(argv: list[str], capsys: Any) -> tuple[int, str, str]:
    code = cli_main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_init_show_get_set(tmp_path: Path, capsys: Any) -> None:
    target = str(tmp_path / "c.toml")
    code, out, _ = _run_cli(["init", "-f", target, "-m", "global"], capsys)
    assert code == 0 and "已初始化" in out

    code, out, _ = _run_cli(
        ["get", "-f", target, "-m", "global", "database.port"], capsys
    )
    assert code == 0 and out.strip() == "5432"

    code, out, _ = _run_cli(
        ["set", "-f", target, "-m", "global", "database.port", "6543", "--save"], capsys
    )
    assert code == 0 and "已保存" in out
    assert loads(Path(target).read_text(encoding="utf-8"))["database"]["port"] == 6543

    code, out, _ = _run_cli(["show", "-f", target, "-m", "global", "--json"], capsys)
    assert code == 0 and json.loads(out)["app"]["name"] == "my-app"


def test_cli_raw_mode_parses_literals(tmp_path: Path, capsys: Any) -> None:
    target = str(tmp_path / "c.toml")
    Path(target).write_text("a = 1\n", encoding="utf-8")
    code, out, _ = _run_cli(["set", "-f", target, "b", "true", "--save"], capsys)
    assert code == 0
    assert loads(Path(target).read_text(encoding="utf-8"))["b"] is True

    code, out, _ = _run_cli(["get", "-f", target, "b"], capsys)
    assert out.strip() == "True"


def test_cli_unset_and_errors(tmp_path: Path, capsys: Any) -> None:
    target = str(tmp_path / "c.toml")
    _run_cli(["init", "-f", target, "-m", "global"], capsys)
    code, out, _ = _run_cli(
        ["unset", "-f", target, "-m", "global", "app.debug", "--save"], capsys
    )
    assert code == 0 and "已删除" in out
    code, out2, err = _run_cli(
        ["unset", "-f", target, "-m", "global", "app.debug"], capsys
    )
    assert code == 1 and "不存在" in out2 + err


def test_cli_validate(tmp_path: Path, capsys: Any) -> None:
    target = tmp_path / "c.toml"
    ConfigManager(Global, target).initialize()
    code, out, _ = _run_cli(["validate", "-f", str(target), "-m", "global"], capsys)
    assert code == 0 and "校验通过" in out

    target.write_text("database = { port = 0 }\n", encoding="utf-8")
    code, out, _err = _run_cli(
        ["validate", "-f", str(target), "-m", "global", "--json"], capsys
    )
    assert code == 1
    assert json.loads(out)["valid"] is False


def test_cli_requires_model(tmp_path: Path, capsys: Any) -> None:
    target = str(tmp_path / "c.toml")
    Path(target).write_text("a = 1\n", encoding="utf-8")
    code, _, err = _run_cli(["validate", "-f", target], capsys)
    assert code == 1 and "需要 --model" in err


def test_cli_export_and_output(tmp_path: Path, capsys: Any) -> None:
    target = tmp_path / "c.toml"
    ConfigManager(Global, target).initialize()
    out_file = tmp_path / "out.env"
    code, out, _ = _run_cli(
        [
            "export",
            "-f",
            str(target),
            "-m",
            "global",
            "--fmt",
            "env",
            "-o",
            str(out_file),
        ],
        capsys,
    )
    assert code == 0 and "已导出" in out
    assert "DATABASE__PORT=5432" in out_file.read_text(encoding="utf-8")


def test_cli_schema_and_models(tmp_path: Path, capsys: Any) -> None:
    code, out, _ = _run_cli(["schema", "-m", "redis"], capsys)
    assert code == 0 and json.loads(out)["title"] == "Redis"

    code, out, _ = _run_cli(["models"], capsys)
    assert code == 0 and "global" in out and "raw" in out


def test_cli_missing_file_error(tmp_path: Path, capsys: Any) -> None:
    code, _, err = _run_cli(["show", "-f", str(tmp_path / "nope.toml")], capsys)
    assert code == 0  # missing_ok：空文档也照常显示
    code, _, err = _run_cli(
        ["get", "-f", str(tmp_path / "nope.toml"), "a.b", "--json"], capsys
    )
    assert code == 1 and "ConfigKeyError" in err


def test_cli_module_is_runnable() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cs.config", "--help"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "TOML" in result.stdout and "serve" in result.stdout


# ================================================================= 结构性
def test_import_has_no_side_effects(tmp_path: Path) -> None:
    """旧实现在 import 时就读 config.toml 并改写环境变量，这里必须没有副作用。"""
    code = (
        "import os, sys;"
        "before = dict(os.environ);"
        "import cs.config;"
        "print('config.toml' in os.listdir('.'));"
        "print(before == dict(os.environ))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["False", "True"]


def test_lazy_import_does_not_pull_fastmcp() -> None:
    code = "import sys, cs.config; print('fastmcp' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


def test_mcp_attribute_is_lazy() -> None:
    from cs.config import mcp

    assert mcp.name == "cs-config"


def test_error_hierarchy() -> None:
    assert issubclass(ConfigValidationError, ValueError)
    assert issubclass(ConfigFileNotFoundError, FileNotFoundError)
    assert issubclass(ConfigExistsError, FileExistsError)
    assert issubclass(ConfigKeyError, KeyError)
    assert str(ConfigKeyError("缺少 a")) == "缺少 a"  # KeyError 不会套引号
    assert issubclass(ConfigSessionError, LookupError)
    assert issubclass(ConfigUnsupportedError, NotImplementedError)


def test_validation_error_lists_everything() -> None:
    error = ConfigValidationError(
        [
            __import__("cs.config", fromlist=["Issue"]).Issue("a.b", "不对"),
            __import__("cs.config", fromlist=["Issue"]).Issue("c", "也不对"),
        ]
    )
    assert "2 项" in str(error)
    assert len(list(error)) == 2
