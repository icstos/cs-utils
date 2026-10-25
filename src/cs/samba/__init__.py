"""``cs.samba`` —— SMB/Samba 共享操作 + FastMCP 服务。

对外主入口是 :class:`SMBClient`::

    from cs.samba import SMBClient

    with SMBClient.open(server="nas", share="share",
                        username="u", password="p") as client:
        client.list_dir("docs")                 # 列目录 -> list[SMBEntry]
        client.mtime("docs/a.pdf")              # 最后更新时间
        client.download("docs/a.pdf", "a.pdf")  # 下载
        client.upload("b.pdf", "docs/b.pdf")    # 上传
        client.mkdir("docs/2026")               # 建目录（可多级）

也可以从 URL / 环境变量构造::

    SMBClient.from_url("smb://u:p@nas:445/share/docs")
    SMBClient.from_env()      # 读 SMB_HOST / SMB_SHARE / SMB_USER / SMB_PASSWORD

MCP 能力按需加载，未安装 ``fastmcp`` 时类方法照常可用::

    from cs.samba import create_server
    create_server().run()

依赖 ``smbprotocol``（SMB 2.0.2 ~ 3.1.1，微软维护），支持签名与 SMB3 加密；
共享枚举（SRVSVC）不在支持范围内——SMB2 没有原生命令，需要额外的 RPC 通道。
"""

from __future__ import annotations

from typing import Any

from .client import ProgressHook, SMBClient, SortKey
from .errors import (
    SambaAccessError,
    SambaAuthError,
    SambaConfigError,
    SambaConnectionError,
    SambaDependencyError,
    SambaError,
    SambaExistsError,
    SambaIsADirectoryError,
    SambaNotADirectoryError,
    SambaNotFoundError,
    SambaOperationError,
    SambaPathError,
    SambaProtocolError,
    SambaShareError,
    SambaTimeoutError,
    SambaUnsupportedError,
)
from .models import (
    FILE_ATTRIBUTE,
    ServerInfo,
    SMBEntry,
    SMBOptions,
    TransferResult,
    as_utc,
    dialect_name,
    human_age,
    human_size,
)
from .paths import RemotePath
from .views import display_width, pad, render_table, render_tree

__all__ = [
    "FILE_ATTRIBUTE",
    "ProgressHook",
    "RemotePath",
    "SMBClient",
    "SMBEntry",
    "SMBOptions",
    "SambaAccessError",
    "SambaAuthError",
    "SambaConfigError",
    "SambaConnectionError",
    "SambaDependencyError",
    "SambaError",
    "SambaExistsError",
    "SambaIsADirectoryError",
    "SambaNotADirectoryError",
    "SambaNotFoundError",
    "SambaOperationError",
    "SambaPathError",
    "SambaProtocolError",
    "SambaShareError",
    "SambaTimeoutError",
    "SambaUnsupportedError",
    "ServerInfo",
    "SortKey",
    "TransferResult",
    "as_utc",
    "create_server",
    "dialect_name",
    "display_width",
    "human_age",
    "human_size",
    "pad",
    "render_table",
    "render_tree",
]


def create_server(*args: Any, **kwargs: Any) -> Any:
    """创建 FastMCP 服务器实例（惰性导入 ``fastmcp``）。"""
    from .mcp_server import create_server as _create_server

    return _create_server(*args, **kwargs)


def __getattr__(name: str) -> Any:
    """``cs.samba.mcp`` 惰性暴露，避免可选依赖在 import 期被拉起。"""
    if name == "mcp":
        from .mcp_server import mcp as server

        return server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
