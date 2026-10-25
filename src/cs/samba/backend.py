"""后端抽象：SMB 原语的最小接口。

:class:`SmbBackend` 只描述"协议能做什么"，不关心重试、递归、覆盖策略等业务语义——
那些都在 :class:`cs.samba.client.SMBClient` 里。把接口抽出来有两个好处：

1. 可以换成别的实现（例如纯内存假后端）来跑离线测试；
2. 业务逻辑与 ``smbprotocol`` 解耦，升级依赖时改动面收敛在一个文件里。
"""

from __future__ import annotations

from datetime import datetime
from typing import BinaryIO, Literal, Protocol, runtime_checkable

from .models import ServerInfo, SMBEntry
from .paths import RemotePath

__all__ = ["OpenMode", "SmbBackend"]

type OpenMode = Literal["create", "overwrite", "append"]


@runtime_checkable
class SmbBackend(Protocol):
    """SMB 原语接口（共享内相对路径 + 已建连接）。"""

    @property
    def connected(self) -> bool:
        """是否已连上服务器并挂载共享。"""
        ...

    def connect(self) -> ServerInfo:
        """建立连接，返回协商结果。"""
        ...

    def close(self) -> None:
        """断开连接（可重复调用）。"""
        ...

    def use_share(self, share: str) -> ServerInfo:
        """切换到另一个共享。"""
        ...

    def stat(self, path: RemotePath) -> SMBEntry:
        """读取单个路径的属性。"""
        ...

    def list_dir(self, path: RemotePath) -> list[SMBEntry]:
        """列出目录的直接子项（不含 ``.`` / ``..``）。"""
        ...

    def open_read(self, path: RemotePath) -> BinaryIO:
        """打开只读文件句柄（调用方负责 close）。"""
        ...

    def open_write(self, path: RemotePath, *, mode: OpenMode) -> BinaryIO:
        """打开写入句柄：``create`` 要求不存在、``overwrite`` 截断、``append`` 追加。"""
        ...

    def mkdir(self, path: RemotePath) -> None:
        """创建单层目录，已存在时报错。"""
        ...

    def remove(self, path: RemotePath) -> None:
        """删除文件（不存在时报错）。"""
        ...

    def rmdir(self, path: RemotePath) -> None:
        """删除空目录，非空时报错。"""
        ...

    def rename(
        self, source: RemotePath, target: RemotePath, *, overwrite: bool
    ) -> None:
        """重命名/移动，跨目录亦可（同一共享内）。"""
        ...

    def set_mtime(self, path: RemotePath, when: datetime) -> None:
        """修改文件的最后写入时间。"""
        ...
