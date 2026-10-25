"""把 :mod:`cs.config` 暴露为 MCP 工具（FastMCP ≥ 4）。

启动方式::

    python -m cs.config serve                     # stdio，供 MCP 客户端拉起
    python -m cs.config serve --transport http --port 8000

约定
----
* 工具名统一 ``config_*``；写操作默认不落盘（``save=False``），要落盘必须显式要求；
* 会话模型：先 ``config_open`` 得到会话（默认名 ``default``），后续工具复用该配置，
  不必每次传路径；空闲 30 分钟自动回收；
* ``model`` 用 ``samples.MODELS`` 里的名字（``global`` / ``app`` / ``redis`` …），
  ``raw`` 表示不做 schema 校验、纯 TOML 读写；
* 成功返回 ``{"ok": true, ...}``，业务异常统一转成 ``ToolError``；
* ``secret=True`` 的字段在导出时默认打码，避免密码出现在模型上下文里。
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from .errors import ConfigError, ConfigSchemaError, ConfigValidationError
from .manager import ConfigManager
from .samples import get_model, model_names
from .sessions import DEFAULT_SESSION, ConfigRegistry, LiveConfig

__all__ = ["create_server", "mcp"]

_SERVER_NAME = "cs-config"
_MAX_EXPORT = 200_000
_INSTRUCTIONS = """\
本服务器读写 TOML 配置文件：初始化、加载、查询、更新、保存、导出（TOML/JSON/env），
并可对配置做基于 dataclass schema 的类型校验。

使用建议：
1. 先 config_open(path, model="global") 打开配置文件；之后所有工具默认复用
   "default" 会话，多份配置可用 session 参数区分；
2. model 取 config_models 里列出的名字；传 "raw" 表示不做校验的纯 TOML 操作；
3. 改配置用 config_set / config_update（会立刻按 schema 校验，坏值当场报错），
   要真正写到磁盘再调 config_save —— 不调就不会动用户的文件；
4. 想让人从零开始填，用 config_init(path, model="global", template=true)，
   会生成一份带注释的骨架；
5. 回答"某项配置现在是什么"用 config_get；看整体结构用 config_info 或 config_export；
6. 导出默认不是脱敏的，需要贴给外部看时传 redact=true，密码会变成 ***；
7. config_schema 可以查某个模型有哪些字段、类型、默认值和约束。
"""


def _guard[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """把领域异常转换成 MCP 客户端能看懂的错误消息。"""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except ConfigValidationError as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
        except (ConfigError, OSError, ValueError, KeyError) as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


def _done(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def _target(pool: ConfigRegistry, session: str) -> ConfigManager[Any]:
    live: LiveConfig = pool.get(session)
    return live.manager


def _resolve_model(name: str) -> type | None:
    """``raw`` → 无 schema；其余按名字取样本模型。"""
    key = (name or "raw").strip().lower()
    if key in {"raw", "none", ""}:
        return None
    try:
        return get_model(key)
    except KeyError as exc:
        raise ConfigSchemaError(str(exc)) from exc


def create_server(
    name: str = _SERVER_NAME,
    *,
    instructions: str = _INSTRUCTIONS,
    registry: ConfigRegistry | None = None,
) -> FastMCP:
    """构造服务器实例；``registry`` 可注入，便于测试隔离。"""
    pool = registry if registry is not None else ConfigRegistry()
    app = FastMCP(name, instructions=instructions)

    # ------------------------------------------------------------ 会话
    @app.tool(annotations={"readOnlyHint": False})
    @_guard
    def config_open(
        path: str,
        model: str = "global",
        session: str = DEFAULT_SESSION,
        env_prefix: str | None = None,
        strict: bool = False,
        extra: str = "forbid",
    ) -> dict[str, Any]:
        """打开一份 TOML 配置并校验，返回概况。

        model="raw" 表示不做 schema 校验；env_prefix 形如 "CS_" 时会读取
        CS__APP__DEBUG=true 这类环境变量并叠加到文件内容之上。
        同一 session 名重复调用会顶替旧会话。
        """
        live = pool.open(
            path,
            _resolve_model(model),
            name=session,
            env_prefix=env_prefix,
            strict=strict,
            extra="forbid" if extra == "forbid" else "ignore",
        )
        return _done(**live.describe())

    @app.tool(annotations={"readOnlyHint": False})
    @_guard
    def config_close(session: str = DEFAULT_SESSION) -> dict[str, Any]:
        """关闭会话（未保存的改动会丢失，先调 config_save）。"""
        return _done(session=session, closed=pool.close(session))

    @app.tool(annotations={"readOnlyHint": True})
    @_guard
    def config_sessions() -> dict[str, Any]:
        """列出当前打开的配置会话。"""
        items = pool.snapshot()
        return _done(count=len(items), sessions=items)

    @app.tool(annotations={"readOnlyHint": True})
    @_guard
    def config_info(session: str = DEFAULT_SESSION) -> dict[str, Any]:
        """查看会话对应文件的信息（路径、模型、顶层小节、是否已改动）。"""
        live = pool.get(session)
        return _done(**live.describe())

    # ------------------------------------------------------------ 读
    @app.tool(annotations={"readOnlyHint": True})
    @_guard
    def config_get(
        path: str = "",
        session: str = DEFAULT_SESSION,
        source: str = "effective",
    ) -> dict[str, Any]:
        """按点号路径读取配置；path 留空返回整份配置。

        路径支持数组下标（servers.0.name）与转义点（a\\.b）。
        source="file" 只看文件，source="effective"（默认）叠加了环境变量覆盖。
        """
        manager = _target(pool, session)
        value = manager.get(path, source="file" if source == "file" else "effective")
        return _done(path=path or "<root>", value=value)

    @app.tool(annotations={"readOnlyHint": True})
    @_guard
    def config_export(
        fmt: str = "toml",
        session: str = DEFAULT_SESSION,
        redact: bool = True,
        source: str = "file",
    ) -> dict[str, Any]:
        """导出配置文本；fmt 可选 toml / json / env。redact=True 会把密码字段打码。"""
        manager = _target(pool, session)
        text = manager.export(
            fmt, redact=redact, source="file" if source == "file" else "effective"
        )
        truncated = len(text) > _MAX_EXPORT
        return _done(
            format=fmt,
            redacted=redact,
            truncated=truncated,
            content=text[:_MAX_EXPORT],
            bytes=len(text),
        )

    # ------------------------------------------------------------ 写
    @app.tool(annotations={"readOnlyHint": False})
    @_guard
    def config_set(
        path: str,
        value: Any,
        session: str = DEFAULT_SESSION,
        save: bool = False,
    ) -> dict[str, Any]:
        """设置单个配置项；会立刻按 schema 校验，非法值当场报错。

        save=True 才写盘，否则只改内存里的副本。
        """
        manager = _target(pool, session)
        manager.set(path, value)
        return _done(
            path=path,
            value=manager.get(path, source="file"),
            saved=_save(manager, save),
        )

    @app.tool(annotations={"readOnlyHint": False})
    @_guard
    def config_update(
        patch: Mapping[str, Any],
        session: str = DEFAULT_SESSION,
        deep: bool = True,
        save: bool = False,
    ) -> dict[str, Any]:
        """深合并一段配置（嵌套字典逐层合并，数组和标量整体替换）。"""
        manager = _target(pool, session)
        manager.update(patch, deep=deep)
        return _done(
            applied=sorted(patch),
            sections=sorted(manager.document.data),
            saved=_save(manager, save),
        )

    @app.tool(annotations={"readOnlyHint": False})
    @_guard
    def config_unset(
        path: str,
        session: str = DEFAULT_SESSION,
        save: bool = False,
    ) -> dict[str, Any]:
        """删除一个配置项（删除后该字段回落到模型默认值）。"""
        manager = _target(pool, session)
        removed = manager.unset(path)
        return _done(path=path, removed=removed, saved=_save(manager, save))

    @app.tool(annotations={"readOnlyHint": False})
    @_guard
    def config_save(
        session: str = DEFAULT_SESSION,
        backup: bool = True,
    ) -> dict[str, Any]:
        """校验并写盘；backup=True 会把原文件另存为 .bak。"""
        manager = _target(pool, session)
        path = manager.save(backup=backup)
        return _done(path=str(path), bytes=path.stat().st_size if path.exists() else 0)

    @app.tool(annotations={"readOnlyHint": False})
    @_guard
    def config_reload(session: str = DEFAULT_SESSION) -> dict[str, Any]:
        """丢弃内存改动，从磁盘重新读取。"""
        manager = _target(pool, session)
        manager.reload()
        return _done(**manager.describe())

    # ------------------------------------------------------------ 初始化/校验
    @app.tool(annotations={"readOnlyHint": False})
    @_guard
    def config_init(
        path: str,
        model: str = "global",
        overwrite: bool = False,
        template: bool = True,
    ) -> dict[str, Any]:
        """初始化一个新的配置文件。

        template=True 写出带注释的骨架（适合给人改），否则写纯默认值。
        文件已存在且 overwrite=False 时会报错，不会静默覆盖。
        """
        resolved = _resolve_model(model)
        manager = ConfigManager(resolved, path)
        manager.initialize(
            overwrite=overwrite, template=template, data=None if resolved else {}
        )
        return _done(path=str(manager.path), model=model, template=template)

    @app.tool(annotations={"readOnlyHint": True})
    @_guard
    def config_validate(session: str = DEFAULT_SESSION) -> dict[str, Any]:
        """校验当前配置；通过返回 ok，不通过返回逐条问题清单。"""
        manager = _target(pool, session)
        try:
            manager.validate()
        except ConfigValidationError as exc:
            return {
                "ok": False,
                "valid": False,
                "issues": [
                    {"path": issue.path, "message": issue.message}
                    for issue in exc.issues
                ],
            }
        return _done(valid=True, model=manager.name)

    @app.tool(annotations={"readOnlyHint": True})
    @_guard
    def config_schema(model: str = "global") -> dict[str, Any]:
        """查看某个模型有哪些字段：类型、默认值、约束、是否敏感。"""
        resolved = _resolve_model(model)
        if resolved is None:
            raise ConfigSchemaError("model='raw' 没有 schema，请指定具体模型名")
        return _done(model=model, schema=ConfigManager(resolved).schema())

    @app.tool(annotations={"readOnlyHint": True})
    @_guard
    def config_models() -> dict[str, Any]:
        """列出可用的配置模型名。"""
        return _done(models=model_names(), raw_available=True)

    return app


def _save(manager: ConfigManager[Any], save: bool) -> bool:
    """按需落盘（``save`` 为真才动磁盘）。"""
    if not save:
        return False
    manager.save()
    return True


mcp = create_server()
