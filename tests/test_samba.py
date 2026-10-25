"""``cs.samba`` 测试。

分两部分：

* **离线**：:class:`FakeBackend` 实现 :class:`~cs.samba.backend.SmbBackend` 协议，
  在内存里模拟目录/文件，用于验证路径、递归、覆盖、续传、会话池、MCP 工具等业务逻辑；
* **在线**：本机开着 SMB 服务时，用真实 ``smbprotocol`` 走一遍协商、认证失败、
  共享不存在、匿名枚举，确认协议层与异常翻译在真机上也成立。
"""

from __future__ import annotations

import asyncio
import io
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs.samba import (
    FILE_ATTRIBUTE,
    RemotePath,
    SambaAccessError,
    SambaAuthError,
    SambaConfigError,
    SambaError,
    SambaExistsError,
    SambaIsADirectoryError,
    SambaNotADirectoryError,
    SambaNotFoundError,
    SambaOperationError,
    SambaPathError,
    SambaShareError,
    SMBClient,
    SMBEntry,
    SMBOptions,
    TransferResult,
    as_utc,
    dialect_name,
    human_age,
    human_size,
    render_table,
    render_tree,
)
from cs.samba.__main__ import _build_parser, _options, main
from cs.samba.backend import SmbBackend
from cs.samba.models import ServerInfo
from cs.samba.sessions import DEFAULT_SESSION, SessionRegistry
from cs.samba.transport import from_filetime, to_filetime
from cs.samba.views import display_width, pad

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


# ================================================================= 假后端
class _Sink(io.BytesIO):
    """写入缓冲：关闭时把内容落到假后端的文件表里。"""

    def __init__(self, backend: FakeBackend, key: str, initial: bytes) -> None:
        super().__init__(initial)
        self._backend = backend
        self._key = key
        if initial:
            self.seek(len(initial))

    def close(self) -> None:
        if not self.closed:
            self._backend.files[self._key] = self.getvalue()
            self._backend.mtimes[self._key] = datetime.now(UTC)
        super().close()


class FakeBackend:
    """内存版 SMB 后端：实现 :class:`SmbBackend` 协议。"""

    def __init__(
        self,
        options: SMBOptions,
        *,
        dirs: tuple[str, ...] = (),
        files: dict[str, bytes] | None = None,
    ) -> None:
        self.options = options
        self.share = options.share
        self.connected = False
        self.connects = 0
        self.closes = 0
        self.dirs: set[str] = {""} | {d.strip("/") for d in dirs}
        self.files: dict[str, bytes] = {
            key.strip("/"): value for key, value in (files or {}).items()
        }
        self.mtimes: dict[str, datetime] = {key: NOW for key in self.files}
        self.dir_mtimes: dict[str, datetime] = {key: NOW for key in self.dirs}
        for path in list(self.files):
            parts = path.split("/")[:-1]
            for index in range(1, len(parts) + 1):
                self.dirs.add("/".join(parts[:index]))

    # -- 连接
    def connect(self) -> ServerInfo:
        self.connects += 1
        self.connected = True
        return ServerInfo(
            server=self.options.server,
            port=self.options.port,
            share=self.share,
            dialect="SMB 3.1.1",
            server_guid="fake-guid",
            username=self.options.effective_user,
            signing=False,
            encrypted=False,
            connected_at=datetime.now(UTC),
        )

    def close(self) -> None:
        self.closes += 1
        self.connected = False

    def use_share(self, share: str) -> ServerInfo:
        self.share = share
        self.connected = False
        return self.connect()

    # -- 查询
    def stat(self, path: RemotePath) -> SMBEntry:
        key = path.posix
        if key in self.dirs:
            return SMBEntry(
                name=path.name,
                path=key,
                is_dir=True,
                attributes=FILE_ATTRIBUTE["directory"],
                mtime=self.dir_mtimes.get(key, NOW),
            )
        if key in self.files:
            return SMBEntry(
                name=path.name,
                path=key,
                is_dir=False,
                size=len(self.files[key]),
                attributes=FILE_ATTRIBUTE["normal"],
                mtime=self.mtimes.get(key, NOW),
            )
        raise SambaNotFoundError(key or "/")

    def list_dir(self, path: RemotePath) -> list[SMBEntry]:
        key = path.posix
        if key not in self.dirs:
            raise SambaNotADirectoryError(key or "/")
        prefix = f"{key}/" if key else ""
        names = {
            item[len(prefix) :].split("/")[0]
            for item in self.files
            if item.startswith(prefix)
        }
        names |= {
            item[len(prefix) :].split("/")[0]
            for item in self.dirs
            if item != key and item.startswith(prefix)
        }
        return [self.stat(RemotePath(f"{prefix}{name}")) for name in sorted(names)]

    # -- 读写
    def open_read(self, path: RemotePath) -> io.BytesIO:
        key = path.posix
        if key in self.dirs:
            raise SambaIsADirectoryError(key)
        if key not in self.files:
            raise SambaNotFoundError(key)
        return io.BytesIO(self.files[key])

    def open_write(self, path: RemotePath, *, mode: str) -> _Sink:
        key = path.posix
        if key in self.dirs:
            raise SambaIsADirectoryError(key)
        if mode == "create" and key in self.files:
            raise SambaExistsError(key)
        parent = path.parent.posix
        if parent and parent not in self.dirs:
            raise SambaNotFoundError(parent)
        initial = self.files.get(key, b"") if mode == "append" else b""
        return _Sink(self, key, initial)

    # -- 修改
    def mkdir(self, path: RemotePath) -> None:
        key = path.posix
        if key in self.dirs:
            raise SambaExistsError(key)
        if key in self.files:
            raise SambaNotADirectoryError(key)
        parent = path.parent.posix
        if parent and parent not in self.dirs:
            raise SambaNotFoundError(parent)
        self.dirs.add(key)
        self.dir_mtimes[key] = datetime.now(UTC)

    def remove(self, path: RemotePath) -> None:
        key = path.posix
        if key in self.dirs:
            raise SambaIsADirectoryError(key)
        if key not in self.files:
            raise SambaNotFoundError(key)
        del self.files[key]
        self.mtimes.pop(key, None)

    def rmdir(self, path: RemotePath) -> None:
        key = path.posix
        if key not in self.dirs:
            raise SambaNotFoundError(key)
        prefix = f"{key}/"
        if any(item.startswith(prefix) for item in self.dirs) or any(
            item.startswith(prefix) for item in self.files
        ):
            raise SambaOperationError(key, "目录非空")
        self.dirs.discard(key)
        self.dir_mtimes.pop(key, None)

    def rename(
        self, source: RemotePath, target: RemotePath, *, overwrite: bool
    ) -> None:
        src, dst = source.posix, target.posix
        if src in self.files:
            if dst in self.files and not overwrite:
                raise SambaExistsError(dst)
            self.files[dst] = self.files.pop(src)
            self.mtimes[dst] = self.mtimes.pop(src, datetime.now(UTC))
            return
        if src in self.dirs:
            if dst in self.dirs and not overwrite:
                raise SambaExistsError(dst)
            moved = {
                item: value
                for item, value in self.files.items()
                if item.startswith(f"{src}/")
            }
            for item, value in moved.items():
                self.files[f"{dst}/{item[len(src) + 1 :]}"] = value
                del self.files[item]
            self.dirs.add(dst)
            return
        raise SambaNotFoundError(src)

    def set_mtime(self, path: RemotePath, when: datetime) -> None:
        key = path.posix
        if key not in self.files:
            raise SambaNotFoundError(key)
        self.mtimes[key] = when


# ================================================================= fixtures
@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend(
        SMBOptions(server="nas", share="share", username="u", password="p"),
        dirs=("docs", "docs/sub", "empty"),
        files={
            "docs/a.txt": b"hello smb",
            "docs/b.pdf": b"%PDF-1.4 fake",
            "docs/sub/c.log": b"line1\nline2\n",
            "readme.md": b"# hi\n",
            ".hidden": b"secret",
        },
    )


@pytest.fixture
def client(backend: FakeBackend) -> SMBClient:
    return SMBClient(backend.options, backend=backend)


# ================================================================= 连接参数
def test_options_validation() -> None:
    with pytest.raises(SambaConfigError):
        SMBOptions(server="  ")
    with pytest.raises(SambaConfigError):
        SMBOptions(server=r"\\nas\share")
    with pytest.raises(SambaConfigError):
        SMBOptions(server="nas", share="share/docs")
    with pytest.raises(SambaConfigError):
        SMBOptions(server="nas", port=0)
    with pytest.raises(SambaConfigError):
        SMBOptions(server="nas", port=70000)
    with pytest.raises(SambaConfigError):
        SMBOptions(server="nas", timeout=0)
    with pytest.raises(SambaConfigError):
        SMBOptions(server="nas", retries=-1)
    with pytest.raises(SambaConfigError):
        SMBOptions(server="nas", backoff=-0.1)


def test_options_derived_and_masking() -> None:
    options = SMBOptions(
        server="nas", share="/share/", username="u", password="secret", domain="CORP"
    )
    assert options.share == "share"
    assert options.unc == r"\\nas\share"
    assert options.effective_user == r"CORP\u"
    assert options.display == "u@nas:445/share"
    payload = options.to_dict()
    assert "password" not in payload
    assert payload["has_password"] is True
    assert "secret" not in str(options)
    assert SMBOptions(server="nas").is_anonymous


def test_options_from_url() -> None:
    options, sub = SMBOptions.from_url("smb://CORP;u:p%40ss@nas:1445/share/a/b")
    assert options.server == "nas"
    assert options.port == 1445
    assert options.share == "share"
    assert options.username == "u"
    assert options.domain == "CORP"
    assert options.password == "p@ss"
    assert sub == "a/b"

    with pytest.raises(SambaConfigError):
        SMBOptions.from_url("http://nas/share")
    with pytest.raises(SambaConfigError):
        SMBOptions.from_url("smb:///share")


def test_options_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {
        "SMB_HOST": "nas",
        "SMB_SHARE": "share",
        "SMB_USER": "u",
        "SMB_PASSWORD": "p",
        "SMB_PORT": "1445",
        "SMB_ENCRYPT": "true",
    }
    options = SMBOptions.from_env(env)
    assert options.port == 1445
    assert options.encrypt is True
    assert options.share == "share"
    assert options.replace(share="other").share == "other"

    with pytest.raises(SambaConfigError):
        SMBOptions.from_env({})
    with pytest.raises(SambaConfigError):
        SMBOptions.from_env({"SMB_HOST": "nas", "SMB_PORT": "abc"})


# ================================================================= 路径
def test_remote_path_normalization() -> None:
    assert RemotePath("a//b/./c").raw == "a/b/c"
    assert RemotePath(r"docs\sub\c.log").raw == "docs/sub/c.log"
    assert RemotePath("/leading/slash/").raw == "leading/slash"
    assert RemotePath("").is_root is True
    assert RemotePath("").wire == ""
    assert RemotePath("a/b").wire == r"a\b"


def test_remote_path_properties() -> None:
    path = RemotePath("docs/report.tar.gz")
    assert path.parts == ("docs", "report.tar.gz")
    assert path.name == "report.tar.gz"
    assert path.stem == "report.tar"
    assert path.suffix == ".gz"
    assert path.parent.raw == "docs"
    assert path.depth == 2
    assert str(path) == "docs/report.tar.gz"
    assert repr(path) == "RemotePath('docs/report.tar.gz')"


def test_remote_path_rejects_escape() -> None:
    with pytest.raises(SambaPathError):
        RemotePath("../etc/passwd")
    with pytest.raises(SambaPathError):
        RemotePath("a/../../b")
    with pytest.raises(SambaPathError):
        RemotePath("a\x00b")
    with pytest.raises(SambaPathError):
        RemotePath("a") / ".."


def test_remote_path_unc() -> None:
    assert RemotePath.parse(
        r"\\nas\share\docs\a.txt", server="nas", share="share"
    ).raw == ("docs/a.txt")
    assert RemotePath.parse(r"\\nas\share", server="nas", share="share").is_root
    with pytest.raises(SambaPathError):
        RemotePath.parse(r"\\other\share\a", server="nas", share="share")
    with pytest.raises(SambaPathError):
        RemotePath.parse(r"\\nas\other\a", server="nas", share="share")
    with pytest.raises(SambaPathError):
        RemotePath.parse("//nas")


def test_remote_path_accepts_itself() -> None:
    path = RemotePath("docs")
    assert RemotePath.parse(path) is path
    assert (path / RemotePath("a")).raw == "docs/a"


# ================================================================= 模型
def test_time_helpers() -> None:
    naive = datetime(2026, 10, 24, 12, 0)
    assert as_utc(naive) == NOW
    assert as_utc(NOW) == NOW
    assert as_utc(None) is None

    stamp = to_filetime(NOW)
    assert from_filetime(stamp) == NOW
    assert from_filetime(0) is None
    assert from_filetime(0xFFFFFFFFFFFFFFFF) is None


def test_format_helpers() -> None:
    assert human_size(0) == "0 B"
    assert human_size(1536) == "1.5 KiB"
    assert human_size(1024**3) == "1.0 GiB"
    assert human_size(None) == "?"
    assert human_age(timedelta(seconds=30)) == "30 秒前"
    assert human_age(timedelta(hours=3)) == "3 小时前"
    assert human_age(timedelta(days=-2)) == "2 天后"
    assert human_age(None) == "未知"
    assert dialect_name(0x0311) == "SMB 3.1.1"
    assert dialect_name("SMB 2.1") == "SMB 2.1"
    assert dialect_name(0x9999).startswith("未知方言")


def test_entry_properties() -> None:
    entry = SMBEntry(
        name=".env",
        path=".env",
        is_dir=False,
        size=10,
        mtime=NOW,
        attributes=FILE_ATTRIBUTE["readonly"],
    )
    assert entry.is_file and entry.is_hidden and entry.is_readonly
    assert entry.flags == ["readonly"]
    assert entry.stem == ".env"
    payload = entry.to_dict()
    assert payload["type"] == "file"
    assert payload["mtime"].startswith("2026-10-24")
    assert payload["mtime_epoch"] == NOW.timestamp()
    assert payload["attributes"] == ["readonly"]
    assert "docs/a.txt" in str(SMBEntry("a.txt", "docs/a.txt", False, 1, NOW))

    bare = SMBEntry("x", "x", False)
    assert bare.to_dict()["mtime"] is None
    assert bare.age() is None


def test_transfer_result_text() -> None:
    result = TransferResult(
        "download", "a/b.bin", "b.bin", 2048, 1.0, resumed_bytes=1024
    )
    assert result.speed == 2048
    assert result.speed_human.endswith("/s")
    assert "续传" in str(result)
    assert result.to_dict()["action"] == "download"


# ================================================================= 客户端基础
def test_client_connection_lifecycle(client: SMBClient, backend: FakeBackend) -> None:
    assert client.connected
    assert client.info().share == "share"
    assert repr(client).endswith("connected>")
    client.close()
    assert not client.connected
    assert backend.closes == 1
    client.close()  # 幂等
    assert client.reconnect().dialect == "SMB 3.1.1"
    assert client.connected


def test_client_classmethods(backend: FakeBackend) -> None:
    opened = SMBClient.open(
        server="nas", share="share", username="u", password="p", auto_connect=False
    )
    assert opened.options.effective_user == "u"
    assert opened.connected is False

    from_url = SMBClient.from_url("smb://u:p@nas/share/docs", auto_connect=False)
    assert from_url.options.share == "share"
    assert from_url.default_path == "docs"

    merged = SMBClient.open(
        SMBOptions(server="nas", share="a"), share="b", auto_connect=False
    )
    assert merged.options.share == "b"


def test_client_with_fake_backend_via_with_backend(backend: FakeBackend) -> None:
    client = SMBClient.with_backend(backend.options, backend)
    assert client.connected is False
    assert client.info().share == "share"
    assert client.connected is True


def test_client_context_manager(backend: FakeBackend) -> None:
    options = backend.options
    with SMBClient(options, backend=backend) as client:
        assert client.list_dir("docs")
    assert backend.closes == 1


def test_client_use_share(client: SMBClient, backend: FakeBackend) -> None:
    assert client.use_share("other") is client
    assert client.share == "other"
    assert backend.share == "other"


def test_client_resolve(client: SMBClient) -> None:
    assert client.resolve("").is_root
    assert client.resolve(r"docs\a.txt").raw == "docs/a.txt"
    assert client.resolve(r"\\nas\share\docs").raw == "docs"
    with pytest.raises(SambaPathError):
        client.resolve(r"\\nas\other\docs")


def test_client_requires_share() -> None:
    from cs.samba.transport import SmbprotocolBackend

    with pytest.raises(SambaConfigError):
        SmbprotocolBackend(SMBOptions(server="nas")).connect()


# ================================================================= 浏览
def test_list_dir_sorting_and_filtering(client: SMBClient) -> None:
    entries = client.list_dir("docs")
    assert [entry.name for entry in entries] == ["sub", "a.txt", "b.pdf"]

    only_pdf = client.list_dir("docs", pattern="*.PDF")
    assert [entry.name for entry in only_pdf] == ["b.pdf"]

    root = client.list_dir("")
    assert ".hidden" not in {entry.name for entry in root}
    assert ".hidden" in {
        entry.name for entry in client.list_dir("", include_hidden=True)
    }

    by_size = client.list_dir("docs", sort="size")
    assert [entry.name for entry in by_size] == ["sub", "a.txt", "b.pdf"]
    assert client.list_dir("docs", sort=None)


def test_stat_and_helpers(client: SMBClient) -> None:
    entry = client.stat("docs/a.txt")
    assert entry.size == len("hello smb")
    assert entry.is_file
    assert client.exists("docs/a.txt")
    assert client.exists("docs")
    assert not client.exists("docs/missing.txt")
    assert client.is_file("docs/a.txt")
    assert client.is_dir("docs")
    assert not client.is_dir("docs/a.txt")
    assert client.size("docs/a.txt") == 9
    assert client.size("docs") == 0
    assert client.stat("").is_dir


def test_mtime_and_age(client: SMBClient, backend: FakeBackend) -> None:
    backend.mtimes["docs/a.txt"] = datetime.now(UTC) - timedelta(hours=2)
    mtime = client.mtime("docs/a.txt")
    assert mtime.tzinfo is not None
    assert client.age("docs/a.txt") > timedelta(hours=1)

    backend.mtimes["docs/a.txt"] = None  # type: ignore[assignment]
    with pytest.raises(SambaOperationError):
        client.mtime("docs/a.txt")


def test_touch(client: SMBClient, backend: FakeBackend) -> None:
    when = datetime(2020, 1, 1, 9, 30, tzinfo=UTC)
    client.touch("docs/a.txt", when)
    assert backend.mtimes["docs/a.txt"] == when
    with pytest.raises(SambaNotFoundError):
        client.touch("docs/none.txt")


def test_iter_entries_and_find(client: SMBClient) -> None:
    paths = [entry.path for entry in client.iter_entries("docs")]
    assert paths == ["docs/sub", "docs/sub/c.log", "docs/a.txt", "docs/b.pdf"]
    assert [entry.path for entry in client.iter_entries("docs", max_depth=0)] == [
        "docs/sub",
        "docs/a.txt",
        "docs/b.pdf",
    ]
    hits = client.find("*.log", "docs")
    assert [entry.name for entry in hits] == ["c.log"]
    assert client.find("sub", "docs", include_dirs=True)[0].is_dir
    assert client.find("*.log", "docs", recursive=False) == []


def test_du_latest_summary(client: SMBClient, backend: FakeBackend) -> None:
    assert client.du("docs") == 9 + 13 + 12
    backend.mtimes["docs/b.pdf"] = datetime.now(UTC)
    assert client.latest("docs").name == "b.pdf"
    assert "3 个文件" in client.summary("docs")
    with pytest.raises(SambaNotFoundError):
        client.latest("empty")


# ================================================================= 修改
def test_mkdir(client: SMBClient, backend: FakeBackend) -> None:
    assert client.mkdir("docs/2026/10/24") == 3
    assert "docs/2026/10/24" in backend.dirs
    assert client.mkdir("docs/2026") == 0
    with pytest.raises(SambaExistsError):
        client.mkdir("docs/2026", exist_ok=False)
    with pytest.raises(SambaNotFoundError):
        client.mkdir("docs/missing/deep", parents=False)
    with pytest.raises(SambaNotADirectoryError):
        client.mkdir("readme.md/sub")
    assert client.mkdir("") == 0


def test_remove(client: SMBClient, backend: FakeBackend) -> None:
    assert client.remove("docs/a.txt") is True
    assert "docs/a.txt" not in backend.files
    assert client.remove("docs/a.txt", missing_ok=True) is False
    with pytest.raises(SambaNotFoundError):
        client.remove("docs/a.txt")
    with pytest.raises(SambaIsADirectoryError):
        client.remove("docs")


def test_rmdir(client: SMBClient, backend: FakeBackend) -> None:
    assert client.rmdir("empty") == 1
    assert "empty" not in backend.dirs
    assert client.rmdir("empty", missing_ok=True) == 0
    with pytest.raises(SambaNotFoundError):
        client.rmdir("empty")
    with pytest.raises(SambaOperationError):
        client.rmdir("docs")
    with pytest.raises(SambaNotADirectoryError):
        client.rmdir("readme.md")
    with pytest.raises(SambaOperationError):
        client.rmdir("")

    removed = client.rmdir("docs", recursive=True)
    assert removed == 5  # sub + sub/c.log + a.txt + b.pdf + docs 本身
    assert "docs" not in backend.dirs and "docs/sub" not in backend.dirs


def test_rename(client: SMBClient, backend: FakeBackend) -> None:
    assert client.rename("docs/a.txt", "docs/renamed.txt").raw == "docs/renamed.txt"
    assert "docs/renamed.txt" in backend.files
    with pytest.raises(SambaNotFoundError):
        client.rename("docs/none.txt", "docs/x.txt")
    backend.files["docs/dup.txt"] = b"x"
    with pytest.raises(SambaExistsError):
        client.rename("docs/dup.txt", "docs/renamed.txt")
    client.rename("docs/dup.txt", "docs/renamed.txt", overwrite=True)
    assert client.move("docs/renamed.txt", "docs/moved.txt").raw == "docs/moved.txt"


# ================================================================= 传输
def test_download_basic(client: SMBClient, tmp_path: Path) -> None:
    target = tmp_path / "out" / "a.txt"
    result = client.download("docs/a.txt", target)
    assert target.read_bytes() == b"hello smb"
    assert result.size == 9
    assert result.action == "download"
    assert not list(target.parent.glob("*.part"))


def test_download_overwrite_and_resume(client: SMBClient, tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"old")
    with pytest.raises(SambaExistsError):
        client.download("docs/a.txt", target)
    client.download("docs/a.txt", target, overwrite=True)
    assert target.read_bytes() == b"hello smb"

    target.write_bytes(b"hello")  # 只下了一半
    result = client.download("docs/a.txt", target, resume=True)
    assert target.read_bytes() == b"hello smb"
    assert result.resumed_bytes == 5
    assert result.size == 4


def test_download_into_directory(client: SMBClient, tmp_path: Path) -> None:
    result = client.download("docs/a.txt", tmp_path)
    assert (tmp_path / "a.txt").exists()
    assert result.target.endswith("a.txt")


def test_download_errors(client: SMBClient, tmp_path: Path) -> None:
    with pytest.raises(SambaNotFoundError):
        client.download("docs/none.txt", tmp_path / "x")
    with pytest.raises(SambaIsADirectoryError):
        client.download("docs", tmp_path / "x")
    client.download("docs/a.txt", tmp_path / "a.txt", verify=True)


def test_upload_basic(client: SMBClient, backend: FakeBackend, tmp_path: Path) -> None:
    source = tmp_path / "up.bin"
    source.write_bytes(b"0123456789")
    result = client.upload(source, "docs/up.bin")
    assert backend.files["docs/up.bin"] == b"0123456789"
    assert result.size == 10
    assert result.action == "upload"


def test_upload_overwrite_and_resume(
    client: SMBClient, backend: FakeBackend, tmp_path: Path
) -> None:
    source = tmp_path / "up.bin"
    source.write_bytes(b"0123456789")
    client.upload(source, "docs/up.bin", overwrite=True)

    with pytest.raises(SambaExistsError):
        client.upload(source, "docs/up.bin")

    backend.files["docs/up.bin"] = b"01234"  # 远端只传了一半
    result = client.upload(source, "docs/up.bin", resume=True)
    assert backend.files["docs/up.bin"] == b"0123456789"
    assert result.resumed_bytes == 5
    assert result.size == 5

    same = client.upload(source, "docs/up.bin", resume=True)
    assert same.size == 0


def test_upload_creates_parents(
    client: SMBClient, backend: FakeBackend, tmp_path: Path
) -> None:
    source = tmp_path / "deep.txt"
    source.write_bytes(b"deep")
    client.upload(source, "new/a/b/deep.txt")
    assert "new/a/b" in backend.dirs
    assert backend.files["new/a/b/deep.txt"] == b"deep"
    assert client.upload(source, "docs/", overwrite=True).target == "docs/deep.txt"


def test_upload_errors(client: SMBClient, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        client.upload(tmp_path / "nope", "docs/x")
    with pytest.raises(SambaIsADirectoryError):
        client.upload(tmp_path, "docs/x")


def test_upload_verify(client: SMBClient, tmp_path: Path) -> None:
    source = tmp_path / "v.txt"
    source.write_bytes(b"verify")
    client.upload(source, "docs/v.txt", verify=True)


def test_upload_dir_and_download_dir(
    client: SMBClient, backend: FakeBackend, tmp_path: Path
) -> None:
    local = tmp_path / "tree"
    (local / "nested").mkdir(parents=True)
    (local / "a.txt").write_bytes(b"aaa")
    (local / "nested" / "b.txt").write_bytes(b"bbb")

    results = client.upload_dir(local, "docs")
    assert len(results) == 2
    assert backend.files["docs/tree/nested/b.txt"] == b"bbb"

    out = tmp_path / "pulled"
    downloaded = client.download_dir("docs/tree", out, overwrite=True)
    assert len(downloaded) == 2
    assert (out / "tree" / "nested" / "b.txt").read_bytes() == b"bbb"


def test_copy(client: SMBClient, backend: FakeBackend) -> None:
    result = client.copy("docs/a.txt", "docs/copy.txt")
    assert backend.files["docs/copy.txt"] == b"hello smb"
    assert result.action == "copy"
    with pytest.raises(SambaExistsError):
        client.copy("docs/a.txt", "docs/copy.txt")
    with pytest.raises(SambaIsADirectoryError):
        client.copy("docs", "docs/copy2")


def test_progress_hook(client: SMBClient, tmp_path: Path) -> None:
    seen: list[tuple[int, int]] = []
    target = tmp_path / "p.bin"
    client.download(
        "docs/a.txt", target, progress=lambda done, total: seen.append((done, total))
    )
    assert seen[-1] == (9, 9)


# ================================================================= 文本
def test_read_write_text(client: SMBClient, backend: FakeBackend) -> None:
    assert client.read_text("docs/a.txt") == "hello smb"
    assert client.read_bytes("docs/a.txt") == b"hello smb"
    assert client.read_text("docs/a.txt", max_bytes=5) == "hello"

    written = client.write_text("docs/new.txt", "第一行\n第二行\n")
    assert written > 0
    assert client.read_text("docs/new.txt") == "第一行\n第二行\n"

    backend.files["bom.txt"] = "\ufeff带 BOM".encode()
    assert client.read_text("bom.txt") == "带 BOM"

    client.write_text("docs/crlf.txt", "a\nb\n", newline="\r\n")
    assert backend.files["docs/crlf.txt"] == b"a\r\nb\r\n"


def test_write_bytes_overwrite_flag(client: SMBClient, backend: FakeBackend) -> None:
    client.write_bytes("docs/x.bin", b"\x00\x01")
    assert backend.files["docs/x.bin"] == b"\x00\x01"
    with pytest.raises(SambaExistsError):
        client.write_bytes("docs/x.bin", b"\x02", overwrite=False)
    client.write_bytes("docs/x.bin", b"\x03", overwrite=True)
    assert backend.files["docs/x.bin"] == b"\x03"


def test_writes_create_directories(client: SMBClient, backend: FakeBackend) -> None:
    client.write_text("brand/new/file.txt", "hi")
    assert "brand/new" in backend.dirs
    assert client.read_text("brand/new/file.txt") == "hi"
    assert client.ensure_dir("brand/other").raw == "brand/other"


# ================================================================= 视图
def test_display_helpers() -> None:
    assert display_width("abc") == 3
    assert display_width("中文") == 4
    assert pad("中", 4) == "中  "
    assert pad("a", 3, align="right") == "  a"


def test_render_table_and_tree(client: SMBClient) -> None:
    table = render_table(client.list_dir("docs"))
    assert "a.txt" in table and "dir" in table
    assert render_table([]) == "（空目录）"

    tree = render_tree(client, "docs", max_depth=2)
    assert tree.startswith(client.unc)
    assert "sub/" in tree and "c.log" in tree


# ================================================================= 会话池
def test_session_registry_lifecycle(backend: FakeBackend) -> None:
    registry = SessionRegistry(
        client_factory=lambda options: SMBClient(options, backend=backend)
    )
    live = registry.open(backend.options)
    assert registry.names == [DEFAULT_SESSION]

    listed = registry.snapshot()
    assert listed[0]["target"] == r"\\nas\share"
    assert listed[0]["connected"] is True

    assert registry.get().client is live.client
    assert registry.close() is True
    assert registry.close() is False
    assert registry.names == []


def test_session_registry_errors(backend: FakeBackend) -> None:
    registry = SessionRegistry(
        client_factory=lambda options: SMBClient(options, backend=backend)
    )
    with pytest.raises(SambaNotFoundError):
        registry.get("nope")
    with pytest.raises(ValueError):
        SessionRegistry(max_idle=0)
    with pytest.raises(ValueError):
        SessionRegistry(max_sessions=0)


def test_session_registry_idle_sweep(backend: FakeBackend) -> None:
    registry = SessionRegistry(
        max_idle=5.0, client_factory=lambda options: SMBClient(options, backend=backend)
    )
    live = registry.open(backend.options, name="idle")
    live.last_used = datetime.now(UTC) - timedelta(seconds=60)
    assert registry.names == []
    with pytest.raises(SambaNotFoundError):
        registry.get("idle")


def test_session_registry_lru_eviction(backend: FakeBackend) -> None:
    registry = SessionRegistry(
        max_sessions=2,
        client_factory=lambda options: SMBClient(options, backend=backend),
    )
    registry.open(backend.options, name="a")
    registry.open(backend.options, name="b")
    registry.open(backend.options, name="c")
    assert sorted(registry.names) == ["b", "c"]
    assert registry.close_all() == 2


def test_session_reopen_replaces(backend: FakeBackend) -> None:
    registry = SessionRegistry(
        client_factory=lambda options: SMBClient(options, backend=backend)
    )
    first = registry.open(backend.options, name="same")
    second = registry.open(backend.options.replace(share="other"), name="same")
    assert first.client is not second.client
    assert second.client.share == "other"
    assert backend.closes >= 1


# ================================================================= MCP
def _mcp_call(server: Any, tool: str, arguments: dict[str, Any]) -> Any:
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    async def run() -> Any:
        async with Client(server) as client:
            return await client.call_tool(tool, arguments)

    try:
        return asyncio.run(run())
    except ToolError as exc:
        return exc


def _mcp_tools(server: Any) -> list[str]:
    from fastmcp import Client

    async def run() -> list[str]:
        async with Client(server) as client:
            return sorted(tool.name for tool in await client.list_tools())

    return asyncio.run(run())


def test_mcp_tool_catalog() -> None:
    from cs.samba.mcp_server import create_server

    tools = _mcp_tools(create_server(registry=SessionRegistry()))
    assert tools == [
        "samba_close",
        "samba_download",
        "samba_find",
        "samba_info",
        "samba_list",
        "samba_mkdir",
        "samba_mtime",
        "samba_open",
        "samba_read_text",
        "samba_remove",
        "samba_rename",
        "samba_rmdir",
        "samba_sessions",
        "samba_stat",
        "samba_tree",
        "samba_upload",
        "samba_write_text",
    ]


def test_mcp_requires_open() -> None:
    from cs.samba.mcp_server import create_server

    server = create_server(registry=SessionRegistry())
    result = _mcp_call(server, "samba_list", {})
    assert isinstance(result, Exception)
    assert "samba_open" in str(result)


def test_mcp_workflow(backend: FakeBackend, tmp_path: Path) -> None:
    from cs.samba.mcp_server import create_server

    registry = SessionRegistry(
        client_factory=lambda options: SMBClient(options, backend=backend)
    )
    server = create_server(registry=registry)

    opened = _mcp_call(
        server,
        "samba_open",
        {"server": "nas", "share": "share", "username": "u", "password": "p"},
    )
    assert opened.structured_content["ok"] is True
    assert opened.structured_content["dialect"] == "SMB 3.1.1"

    listed = _mcp_call(server, "samba_list", {"path": "docs"})
    assert listed.structured_content["count"] == 3
    assert listed.structured_content["total_bytes"] == 9 + 13

    mtime = _mcp_call(server, "samba_mtime", {"path": "docs/a.txt"})
    assert mtime.structured_content["age_human"]
    assert mtime.structured_content["mtime_utc"].startswith("2026-10-24")

    tree = _mcp_call(server, "samba_tree", {"path": "docs", "max_depth": 1})
    assert "sub/" in tree.structured_content["tree"]

    text = _mcp_call(server, "samba_read_text", {"path": "docs/a.txt"})
    assert text.structured_content["text"] == "hello smb"

    local = tmp_path / "mcp.txt"
    got = _mcp_call(
        server,
        "samba_download",
        {"remote_path": "docs/a.txt", "local_path": str(local)},
    )
    assert got.structured_content["size"] == 9
    assert local.read_bytes() == b"hello smb"

    put = _mcp_call(
        server,
        "samba_upload",
        {"local_path": str(local), "remote_path": "docs/mcp.txt", "overwrite": True},
    )
    assert put.structured_content["action"] == "upload"

    made = _mcp_call(server, "samba_mkdir", {"path": "docs/deep/x"})
    assert made.structured_content["created"] == 2

    renamed = _mcp_call(
        server,
        "samba_rename",
        {"source": "docs/mcp.txt", "target": "docs/deep/x/mcp.txt"},
    )
    assert renamed.structured_content["target"] == "docs/deep/x/mcp.txt"

    removed = _mcp_call(server, "samba_remove", {"path": "docs/deep/x/mcp.txt"})
    assert removed.structured_content["removed"] is True

    sessions = _mcp_call(server, "samba_sessions", {})
    assert sessions.structured_content["count"] == 1

    closed = _mcp_call(server, "samba_close", {})
    assert closed.structured_content["closed"] is True


def test_mcp_error_mapping(backend: FakeBackend) -> None:
    from cs.samba.mcp_server import create_server

    registry = SessionRegistry(
        client_factory=lambda options: SMBClient(options, backend=backend)
    )
    server = create_server(registry=registry)
    _mcp_call(server, "samba_open", {"server": "nas", "share": "share"})
    missing = _mcp_call(server, "samba_stat", {"path": "docs/none.txt"})
    assert isinstance(missing, Exception)
    assert "SambaNotFoundError" in str(missing)


def test_mcp_create_server_is_independent() -> None:
    from cs.samba.mcp_server import create_server

    first = create_server("a", registry=SessionRegistry())
    second = create_server("b", registry=SessionRegistry())
    assert first.name == "a" and second.name == "b"


# ================================================================= CLI
def test_cli_options_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMB_HOST", "nas")
    monkeypatch.setenv("SMB_SHARE", "share")
    parsed = _build_parser().parse_args(["ls", "docs"])
    options = _options(parsed)
    assert options.server == "nas"
    assert options.share == "share"


def test_cli_options_prefer_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMB_SHARE", "env-share")
    parsed = _build_parser().parse_args(
        ["ls", "--server", "host2", "--share", "cli-share"]
    )
    options = _options(parsed)
    assert options.server == "host2"
    assert options.share == "cli-share"


def test_cli_options_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SMB_HOST", raising=False)
    monkeypatch.delenv("SMB_SERVER", raising=False)
    parsed = _build_parser().parse_args(["ls"])
    with pytest.raises(SambaConfigError):
        _options(parsed)


def test_cli_no_command_shows_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SMB_HOST", raising=False)
    monkeypatch.delenv("SMB_SERVER", raising=False)
    assert main([]) == 1  # 缺连接参数 -> 打印错误并返回 1


def test_cli_help_exits() -> None:
    with pytest.raises(SystemExit):
        main(["--help"])


def test_cli_module_is_runnable() -> None:
    """``python -m cs.samba`` 必须真的执行 main（漏写 __main__ 守卫会静默退出 0）。"""
    out = subprocess.run(
        [sys.executable, "-m", "cs.samba", "--help"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0
    assert "SMB/Samba" in out.stdout
    assert "serve" in out.stdout


# ================================================================= 传输层细节
class _StubHandle:
    """冒充 ``smbprotocol.open.Open``：只需 read/write/close/file_name。"""

    def __init__(self, data: bytes = b"", name: str = "docs/a.txt") -> None:
        self.buffer = bytearray(data)
        self.file_name = name.replace("/", "\\")
        self.closed = False
        self.calls: list[tuple[str, int, int]] = []

    def read(self, offset: int, length: int) -> bytes:
        self.calls.append(("read", offset, length))
        return bytes(self.buffer[offset : offset + length])

    def write(self, data: bytes, offset: int) -> int:
        self.calls.append(("write", offset, len(data)))
        end = offset + len(data)
        if end > len(self.buffer):
            self.buffer.extend(b"\0" * (end - len(self.buffer)))
        self.buffer[offset:end] = data
        return len(data)

    def close(self) -> None:
        self.closed = True


class _StubBackend:
    """冒充 :class:`SmbprotocolBackend`：``invoke`` 直接执行，不连接。"""

    closing = False

    @staticmethod
    def invoke(action: str, path: str, func: Any) -> Any:
        return func()


def _remote_file(handle: _StubHandle, **kwargs: Any) -> Any:
    from cs.samba.transport import RemoteFile

    params: dict[str, Any] = {
        "readable": True,
        "writable": False,
        "size": len(handle.buffer),
    }
    params.update(kwargs)
    return RemoteFile(_StubBackend(), handle, **params)  # type: ignore[arg-type]


def test_remote_file_read_path() -> None:
    """真实传输层的读取路径：``readinto`` 必须能跑通（dummy 句柄即可覆盖）。"""
    handle = _StubHandle(b"hello world")
    stream = _remote_file(handle)
    assert stream.readable() is True and stream.writable() is False
    assert stream.read(5) == b"hello"
    assert stream.tell() == 5
    assert stream.read() == b" world"
    assert stream.tell() == 11
    assert stream.read() == b""  # 末尾返回空
    stream.close()
    assert handle.closed is True


def test_remote_file_seek_and_size() -> None:
    handle = _StubHandle(b"0123456789")
    stream = _remote_file(handle)
    assert stream.seekable() is True
    assert stream.size == 10
    assert stream.seek(-3, io.SEEK_END) == 7
    assert stream.read(2) == b"78"
    assert stream.seek(0) == 0
    assert stream.read(3) == b"012"
    with pytest.raises(ValueError, match="负的偏移量"):
        stream.seek(-1)


def test_remote_file_write_path() -> None:
    handle = _StubHandle(b"")
    stream = _remote_file(handle, readable=False, writable=True, size=0)
    assert stream.write(b"abc") == 3
    assert stream.write(b"def") == 3
    assert bytes(handle.buffer) == b"abcdef"
    assert stream.size == 6
    with pytest.raises(io.UnsupportedOperation):
        stream.read(1)


def test_remote_file_rejects_wrong_direction() -> None:
    read_only = _remote_file(_StubHandle(b"x"))
    with pytest.raises(io.UnsupportedOperation):
        read_only.write(b"y")

    write_only = _remote_file(_StubHandle(b""), readable=False, writable=True)
    with pytest.raises(io.UnsupportedOperation):
        write_only.read(1)


def test_remote_file_closed_raises() -> None:
    stream = _remote_file(_StubHandle(b"data"))
    stream.close()
    with pytest.raises(ValueError, match="closed file"):
        stream.read(1)


def test_remote_file_path_property() -> None:
    stream = _remote_file(_StubHandle(b"", name="docs/sub/a.txt"))
    assert stream._path == "docs/sub/a.txt"  # 契约：错误信息里要显示远程路径


def test_translate_privilege_not_held_is_access_error() -> None:
    from smbprotocol.exceptions import PrivilegeNotHeld

    from cs.samba.transport import _translate

    options = SMBOptions(server="nas", share="share")
    mapped = _translate(PrivilegeNotHeld(), options, path="docs", action="创建目录")
    assert isinstance(mapped, SambaAccessError)
    assert isinstance(mapped, PermissionError)


def test_session_error_prefers_auth_failure() -> None:
    """认证阶段掉线应报认证错误（Windows 常直接掐 TCP 而不回状态码）。"""
    from cs.samba.transport import _session_error

    options = SMBOptions(server="nas", share="share", username="bob")
    mapped = _session_error(OSError("socket aborted by peer"), options)
    assert isinstance(mapped, SambaAuthError)
    assert mapped.username == "bob"

    timed_out = _session_error(TimeoutError("timed out"), options)
    assert not isinstance(timed_out, SambaAuthError)


# ================================================================= 结构性
def test_backend_protocol_conformance() -> None:
    from cs.samba.transport import SmbprotocolBackend

    assert isinstance(FakeBackend(SMBOptions(server="nas", share="share")), SmbBackend)
    assert isinstance(
        SmbprotocolBackend(SMBOptions(server="nas", share="share")), SmbBackend
    )


def test_lazy_import_does_not_pull_fastmcp() -> None:
    code = "import sys, cs.samba; print('fastmcp' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"


# ================================================================= 真实服务端
def _live_options(**kwargs: Any) -> SMBOptions:
    base = {
        "server": "127.0.0.1",
        "share": "IPC$",
        "allow_guest": True,
        "retries": 0,
        "timeout": 3.0,
    }
    base.update(kwargs)
    return SMBOptions(**base)


def _live_available() -> bool:
    try:
        SMBClient(_live_options()).close()
    except SambaError:
        return False
    return True


LIVE = _live_available()


def test_live_negotiates_modern_dialect() -> None:
    if not LIVE:
        pytest.skip("本机没有可匿名访问的 SMB 服务")
    client = SMBClient(_live_options())
    info = client.info()
    assert info.dialect.startswith("SMB 3.")
    assert info.port == 445
    assert client.list_dir("")  # IPC$ 里至少有一个命名管道
    client.close()


def test_live_auth_failure_maps_to_permission_error() -> None:
    if not LIVE:
        pytest.skip("本机没有可匿名访问的 SMB 服务")
    options = _live_options(share="IPC$")
    options = options.replace(allow_guest=False)
    with pytest.raises(SambaAuthError):
        SMBClient(
            options.replace(username="aigcs", password="definitely-wrong-password")
        )

    with pytest.raises(PermissionError):  # 兼容内建语义
        SMBClient(
            options.replace(username="aigcs", password="definitely-wrong-password")
        )


def test_live_missing_share() -> None:
    if not LIVE:
        pytest.skip("本机没有可匿名访问的 SMB 服务")
    with pytest.raises(SambaShareError):
        SMBClient(_live_options(share="no-such-share-cs-samba-test"))
