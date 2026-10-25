"""共享内相对路径的规范化与校验。

SMB 的路径分隔符是反斜杠，而用户/脚本里 ``a/b``、``a\\b``、``\\\\nas\\share\\a``
都可能出现。:class:`RemotePath` 负责把它们统一成 **相对共享根的 POSIX 形式**，
并在入口处挡掉 ``..`` 越界与 NUL 等非法字符，后续所有 SMB 调用都只接受这种形式。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from .errors import SambaPathError
from .models import StrPath, ensure_remote_path

__all__ = ["PathSpec", "RemotePath"]

_ILLEGAL = frozenset("\x00")


@dataclass(frozen=True, slots=True, order=True)
class RemotePath:
    """共享内的相对路径（不可变值对象）。

    构造即完成规范化::

        RemotePath("a//b/./c").raw              # 'a/b/c'
        RemotePath(r"docs\\a.txt").raw          # 'docs/a.txt'
        RemotePath.parse(r"\\\\nas\\share\\x", server="nas", share="share").raw  # 'x'
        RemotePath("..")                        # 抛 SambaPathError
    """

    #: 规范化后的 POSIX 路径；共享根为空字符串
    raw: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", _normalize(self.raw))

    # -- 构造
    @classmethod
    def parse(
        cls,
        value: StrPath | RemotePath = "",
        *,
        server: str | None = None,
        share: str | None = None,
    ) -> RemotePath:
        """解析任意写法的路径。

        传入 ``server`` / ``share`` 时，``\\\\server\\share\\a`` 这类 UNC 路径会被
        校验并剥掉前缀；只写 ``\\\\server\\share`` 则得到共享根。
        """
        if isinstance(value, RemotePath):  # 已是规范化结果，直接复用
            return value
        raw = ensure_remote_path(value).replace("\\", "/")
        if raw.startswith("//"):
            segments = [part for part in raw.split("/") if part]
            if len(segments) < 2:
                raise SambaPathError(f"UNC 路径至少需要 \\\\server\\share：{value!r}")
            host, name, *rest = segments
            if server is not None and host.casefold() != server.strip().casefold():
                raise SambaPathError(
                    f"路径中的服务器 {host!r} 与当前连接 {server!r} 不一致"
                )
            if share and name.casefold() != share.casefold():
                raise SambaPathError(
                    f"路径中的共享 {name!r} 与当前连接 {share!r} 不一致"
                )
            return cls("/".join(rest))
        return cls(raw)

    def join(self, *parts: StrPath | RemotePath) -> RemotePath:
        """拼接子路径（每段都会重新校验）。"""
        base = self.raw
        for part in parts:
            piece = (
                part.posix
                if isinstance(part, RemotePath)
                else _normalize(ensure_remote_path(part).replace("\\", "/"))
            )
            if piece:
                base = f"{base}/{piece}" if base else piece
        return RemotePath(base)

    def child(self, name: str) -> RemotePath:
        return self.join(name)

    # -- 派生属性
    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(self.raw.split("/")) if self.raw else ()

    @property
    def is_root(self) -> bool:
        return not self.raw

    @property
    def depth(self) -> int:
        """层级深度，共享根为 0。"""
        return len(self.parts)

    @property
    def name(self) -> str:
        return self.parts[-1] if self.raw else ""

    @property
    def stem(self) -> str:
        return PurePosixPath(self.name).stem

    @property
    def suffix(self) -> str:
        return PurePosixPath(self.name).suffix

    @property
    def parent(self) -> RemotePath:
        return RemotePath("/".join(self.parts[:-1])) if self.raw else self

    @property
    def wire(self) -> str:
        """发给 SMB 服务端的字符串形式（反斜杠、根为空）。"""
        return self.raw.replace("/", "\\")

    @property
    def posix(self) -> str:
        return self.raw

    # -- 语法糖
    def __truediv__(self, other: StrPath) -> RemotePath:
        return self.join(other)

    def __str__(self) -> str:
        return self.raw or "."

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.raw!r})"


def _normalize(raw: str) -> str:
    """折叠分隔符与 ``.``，并拒绝 ``..`` 与 NUL。"""
    if _ILLEGAL & set(raw):
        raise SambaPathError(f"远程路径不能包含 NUL 字符：{raw!r}")
    cleaned = raw.replace("\\", "/")
    segments: list[str] = []
    for segment in cleaned.split("/"):
        if segment in {"", "."}:
            continue
        if segment == "..":
            raise SambaPathError(f"远程路径不能包含 '..'：{raw!r}")
        segments.append(segment)
    return "/".join(segments)


#: 客户端方法接受的远程路径：字符串、路径对象，或已解析好的 :class:`RemotePath`
type PathSpec = StrPath | RemotePath
