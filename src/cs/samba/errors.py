"""``cs.samba`` 的异常体系。

设计原则与 :mod:`cs.pdf` / :mod:`cs.nvidia` 保持一致：所有异常都继承
:class:`SambaError`，调用方既能精确捕获子类，也能用 ``except SambaError`` 兜底；
同时尽量复用内建异常语义（``FileNotFoundError`` / ``PermissionError`` …），
让不熟悉本模块的代码也能凭直觉处理::

    try:
        client.download("a/b.zip", "b.zip")
    except FileNotFoundError:   # 远程文件不存在
        ...
    except PermissionError:     # 认证失败或无权限
        ...
"""

from __future__ import annotations

__all__ = [
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
]


class SambaError(Exception):
    """SMB 操作异常基类。"""


# ---------------------------------------------------------------- 配置 / 依赖
class SambaConfigError(SambaError, ValueError):
    """连接参数非法（端口越界、共享名带斜杠、URL 解析失败等）。"""


class SambaPathError(SambaError, ValueError):
    """远程路径非法（试图用 ``..`` 跳出共享根、含非法字符等）。"""


class SambaDependencyError(SambaError, ImportError):
    """缺少 ``smbprotocol`` 依赖。"""

    def __init__(self, purpose: str = "SMB 操作") -> None:
        super().__init__(
            f"{purpose} 需要依赖 smbprotocol，请安装：pip install 'cs-utils[samba]'"
        )


# ---------------------------------------------------------------- 连接 / 认证
class SambaConnectionError(SambaError, ConnectionError):
    """无法建立或维持连接（TCP 不通、协商失败、连接被对端关闭）。"""

    def __init__(self, server: str, port: int, reason: str) -> None:
        self.server = server
        self.port = port
        self.reason = reason
        super().__init__(f"无法连接 SMB 服务器 {server}:{port} —— {reason}")


class SambaTimeoutError(SambaConnectionError, TimeoutError):
    """连接或读写超时。"""


class SambaAuthError(SambaError, PermissionError):
    """认证失败：用户名/密码错误、账号被禁用、不支持该认证方式。"""

    def __init__(self, server: str, username: str, reason: str = "认证失败") -> None:
        self.server = server
        self.username = username
        self.reason = reason
        who = username or "<匿名>"
        super().__init__(f"{server} 拒绝了用户 {who} 的认证 —— {reason}")


class SambaShareError(SambaError, FileNotFoundError):
    """共享不存在，或当前账号无权访问该共享。"""

    def __init__(
        self, server: str, share: str, reason: str = "共享不存在或无权访问"
    ) -> None:
        self.server = server
        self.share = share
        self.reason = reason
        super().__init__(f"无法挂载共享 \\\\{server}\\{share} —— {reason}")


# ---------------------------------------------------------------- 文件操作
class SambaNotFoundError(SambaError, FileNotFoundError):
    """远程路径不存在。"""

    def __init__(self, path: str, reason: str = "路径不存在") -> None:
        self.path = path
        super().__init__(f"{reason}: {path}")


class SambaExistsError(SambaError, FileExistsError):
    """目标已存在且未允许覆盖。"""

    def __init__(self, path: str, reason: str = "目标已存在") -> None:
        self.path = path
        super().__init__(f"{reason}: {path}（如需覆盖请设置 overwrite=True）")


class SambaAccessError(SambaError, PermissionError):
    """路径存在但当前账号没有权限。"""

    def __init__(self, path: str, reason: str = "拒绝访问") -> None:
        self.path = path
        super().__init__(f"{reason}: {path}")


class SambaIsADirectoryError(SambaError, IsADirectoryError):
    """期望文件却拿到目录。"""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"路径是目录而非文件: {path}")


class SambaNotADirectoryError(SambaError, NotADirectoryError):
    """期望目录却拿到文件。"""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"路径是文件而非目录: {path}")


class SambaOperationError(SambaError, OSError):
    """其它 SMB 层操作失败（目录非空、被占用、磁盘满等）。"""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class SambaProtocolError(SambaError):
    """服务器返回了无法预期的响应，或服务端不支持该能力。"""


class SambaUnsupportedError(SambaProtocolError, NotImplementedError):
    """该服务器/共享不支持此操作（如 SMB1 服务端、只读共享）。"""
