import hashlib
import io
import shutil
import threading
import time
import zipfile
from os import utime
from pathlib import Path
from typing import BinaryIO, Callable

import platform
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# 常用后缀集合（按用途分组）
# ---------------------------------------------------------------------------

# Ultralytics 平台常用图片格式：JPEG、PNG、WebP、BMP、TIFF、HEIC、AVIF、JP2、DNG、MPO
IMG_SUFFIXS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".avif",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
}
COMMON_IMG_SUFFIX = ".png"
# Ultralytics 平台常用视频格式：MP4、WebM、MOV、AVI、MKV、M4V
VIDEO_SUFFIXS: set[str] = {
    ".mp4",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
}


DOCS_SUFFIXS: set[str] = {
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".doc",
    ".xlsx",
    ".xls",
    ".pptx",
    ".ppt",
    ".csv",
}
ARCHIVE_SUFFIX_SET: set[str] = {
    ".iso",
    ".tar",
    ".gz",
    ".7z",
    ".dmg",
    ".rar",
    ".rar",
    ".zip",
}


# ---------------------------------------------------------------------------
# Windows：调用 shell32 的 SHFileOperationW（FO_DELETE + FOF_ALLOWUNDO）
# ---------------------------------------------------------------------------


def _delete_to_trash_windows(path: Path) -> None:
    """把 *path* 通过 shell32 接口移入 Windows 回收站。"""
    import ctypes
    from ctypes import wintypes

    FO_DELETE = 0x0003
    FOF_ALLOWUNDO = 0x0040
    FOF_NOCONFIRMATION = 0x0010
    FOF_SILENT = 0x0004
    FOF_NOERRORUI = 0x0400

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND),
            ("wFunc", wintypes.UINT),
            ("pFrom", ctypes.c_wchar_p),
            ("pTo", ctypes.c_wchar_p),
            ("fFlags", wintypes.USHORT),
            ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", ctypes.c_void_p),
            ("lpszProgressTitle", ctypes.c_wchar_p),
        ]

    # pFrom 要求字符串以双 NUL 结尾
    pfrom_buf = str(path.resolve()) + '\0\0'

    op = SHFILEOPSTRUCTW()
    op.hwnd = 0
    op.wFunc = FO_DELETE
    op.pFrom = pfrom_buf
    op.pTo = None
    op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI
    op.fAnyOperationsAborted = False
    op.hNameMappings = None
    op.lpszProgressTitle = None

    ret = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if ret != 0:
        raise OSError(f"SHFileOperationW returned {ret}")
    if op.fAnyOperationsAborted:
        raise OSError("File operation was aborted by the user")


# ---------------------------------------------------------------------------
# Linux：按 FreeDesktop.org 回收站规范实现
# ---------------------------------------------------------------------------


def _get_mount_point(p: Path) -> Path:
    """向上回溯，找到 *p* 所在的挂载点。"""
    p = p.resolve()
    while not p.is_mount():
        parent = p.parent
        if parent == p:  # 已到达根目录
            break
        p = parent
    return p


def _find_trash_dir(path: Path) -> Path:
    """返回 *path* 应当使用的回收站目录。"""
    home_trash = Path.home() / '.local' / 'share' / 'Trash'
    path_mount = _get_mount_point(path)
    home_mount = _get_mount_point(Path.home())

    # 同一分区 → 用家目录回收站
    if path_mount == home_mount:
        return home_trash

    # 跨分区 → 先试 .Trash-<uid>，再试 .Trash/<uid>
    uid = str(path.stat().st_uid)
    alt_trash = path_mount / f'.Trash-{uid}'

    if alt_trash.is_dir():
        return alt_trash

    alt_trash2 = path_mount / '.Trash' / uid
    if alt_trash2.is_dir():
        return alt_trash2

    # 两者都不存在 → 在挂载点根目录创建 .Trash-<uid>
    alt_trash.mkdir(parents=True, exist_ok=True)
    return alt_trash


def _unique_trash_name(files_dir: Path, name: str) -> Path:
    """在 *files_dir* 内返回一个尚不存在的目标路径。"""
    dest = files_dir / name
    if not dest.exists():
        return dest

    stem = Path(name).stem
    suffix = Path(name).suffix
    counter = 1
    while (dest := files_dir / f"{stem}.{counter}{suffix}").exists():
        counter += 1
    return dest


def _delete_to_trash_linux(path: Path) -> None:
    """把 *path* 移入 Linux 回收站（FreeDesktop 规范）。"""
    path = path.resolve()
    trash_dir = _find_trash_dir(path)
    files_dir = trash_dir / 'files'
    info_dir = trash_dir / 'info'

    files_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)

    dest = _unique_trash_name(files_dir, path.name)

    # 先写 .trashinfo 再移动，保证元数据不丢失
    info_path = info_dir / f"{dest.name}.trashinfo"
    deletion_date = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
    info_path.write_text(
        f"[Trash Info]\nPath={path}\nDeletionDate={deletion_date}\n", encoding='utf-8'
    )

    try:
        path.rename(dest)
    except Exception:
        info_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# macOS：通过 AppleScript 交给 Finder 处理
# ---------------------------------------------------------------------------


def _delete_to_trash_macos(path: Path) -> None:
    """把 *path* 通过 Finder 移入 macOS 废纸篓。"""
    import subprocess

    script = f'tell application "Finder" to delete POSIX file "{path.resolve()}"'
    result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
    if result.returncode != 0:
        raise OSError(f"osascript failed: {result.stderr.strip()}")


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------


def delete(path: str | Path) -> None:
    """
    把 *path* 移入系统回收站（废纸篓）。

    等价于图形文件管理器里的“删除”，之后仍可还原。

    Args:
        path: 待删除的文件或目录。

    Raises:
        FileNotFoundError: *path* 不存在。
        NotImplementedError: 当前操作系统不受支持。
        OSError: 底层系统调用失败。
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    system = platform.system()

    match system:
        case 'Windows':
            _delete_to_trash_windows(p)
        case 'Linux':
            _delete_to_trash_linux(p)
        case 'Darwin':
            _delete_to_trash_macos(p)
        case _:
            raise NotImplementedError(f"Unsupported platform: {system!r}")


def _read_chunks(file: BinaryIO, size: int = io.DEFAULT_BUFFER_SIZE):
    """按块迭代读取文件内容。"""
    while chunk := file.read(size):
        yield chunk


def hash_file(filename: str | Path) -> str:
    """计算文件的 MD5 哈希值。"""
    filename = Path(filename)
    h = hashlib.md5()
    with filename.open("rb") as f:
        for chunk in _read_chunks(f, 8192):
            h.update(chunk)
    return h.hexdigest()


def find_duplicates(folder: str | Path) -> list[tuple[str, str]]:
    """按 MD5 哈希查找目录内的重复文件。"""
    hashes: dict[str, str] = {}
    duplicates: list[tuple[str, str]] = []
    for file_path in Path(folder).rglob("*"):
        if file_path.is_file():
            try:
                file_hash = hash_file(file_path)
                if file_hash in hashes:
                    duplicates.append((str(file_path), hashes[file_hash]))
                else:
                    hashes[file_hash] = str(file_path)
            except PermissionError:
                print(f"Permission denied: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return duplicates


class File:
    img_suffixs = IMG_SUFFIXS

    @staticmethod
    def delete(path: Path) -> None:
        """删除文件，文件不存在时忽略。"""
        path.unlink(missing_ok=True)

    @staticmethod
    def delete_dir(path: Path) -> None:
        """递归删除目录，忽略异常。"""
        shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def delete_to_trash(path: Path) -> None:
        """把文件移入系统回收站。"""
        delete(path)

    @staticmethod
    def get_file_hash(
        file_path: Path, blocksize: int = 1 << 20, hash_type: str = 'sha256'
    ) -> str:
        """计算文件哈希（sha256 或 md5）。"""
        if hash_type == 'sha256':
            file_hash = hashlib.sha256()
        else:
            file_hash = hashlib.md5()

        with file_path.open("rb") as f:
            for block in _read_chunks(f, size=blocksize):
                file_hash.update(block)
        return file_hash.hexdigest()

    @staticmethod
    def copy(file_path: Path, new_file_path: Path, keep_state: bool = True) -> None:
        """复制文件；keep_state=True 时一并保留元数据。"""
        if keep_state:
            shutil.copy2(file_path, new_file_path)
        else:
            shutil.copy(file_path, new_file_path)

    @staticmethod
    def cp_dir(src_dir: Path, dst_dir: Path) -> None:
        """复制整个目录树（保留元数据）。"""
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    @staticmethod
    def get_size(file_path: Path) -> float:
        """获取文件大小（字节），保留两位小数。"""
        return round(file_path.stat().st_size, 2)

    @staticmethod
    def get_size_mb(file_path: Path) -> float:
        """获取文件大小（MB），保留两位小数。"""
        return round(file_path.stat().st_size / 1024 / 1024, 2)

    @staticmethod
    def set_file_time(path: Path, atime: float, mtime: float) -> None:
        """设置文件的访问时间与修改时间。"""
        path.touch()
        utime(path, (atime, mtime))  # pathlib 没有 utime，仍需用 os.utime

    @staticmethod
    def is_img_file(file_path: Path) -> bool:
        """按后缀判断是否为图片文件。"""
        return file_path.suffix in IMG_SUFFIXS

    @staticmethod
    def get_txt_path(file_path: Path) -> Path:
        """返回同名的 .txt 路径。"""
        return file_path.with_suffix(".txt")

    # ---- 扫描与收集 ----

    @staticmethod
    def get_all_suffix_file(
        file_dir: Path, suffix_set: set[str] | None = None
    ) -> list[Path]:
        """获取 file_dir 下所有后缀命中 suffix_set 的文件。"""
        suffix_set = suffix_set or {".jpg"}
        file_list: list[Path] = []
        # 用 Path.walk()（3.12+）单次遍历，替代每个后缀一次 rglob
        for _dirpath, _dirnames, filenames in file_dir.walk():
            for name in filenames:
                fp = _dirpath / name
                if fp.suffix in suffix_set:
                    file_list.append(fp)
        return file_list

    @staticmethod
    def get_file_name_dict(
        file_dir: Path, suffix_set: set[str] | None = None
    ) -> dict[str, list[Path]]:
        """按后缀分组返回文件：{'.ext': [Path, ...]}。"""
        suffix_file_dict: dict[str, list[Path]] = {}
        # Path.walk() 直接给出文件名，无需再过滤 is_file()
        for _dirpath, _dirnames, filenames in file_dir.walk():
            for name in filenames:
                fp = _dirpath / name
                if suffix_set is None or fp.suffix in suffix_set:
                    suffix_file_dict.setdefault(fp.suffix, []).append(fp)
        return suffix_file_dict

    @staticmethod
    def get_img_file_list(file_path: Path) -> list[Path]:
        """获取目录下所有图片文件。"""
        return File.get_all_suffix_file(file_path, suffix_set=File.img_suffixs)

    @staticmethod
    def get_file_suffix_set(file_dir: str | Path) -> set[str]:
        """获取文件后缀集合（仅顶层目录）。"""
        file_dir = Path(file_dir)
        return {f.suffix for f in file_dir.iterdir() if f.is_file()}

    @staticmethod
    def get_file_stat(path: Path, relative_dir: Path | None = None) -> dict[str, dict]:
        """返回单个文件的 {key: {size, atime, mtime, ctime, hash}}。"""
        stat = path.stat()
        key = (
            str(path.absolute())
            if relative_dir is None
            else str(path.relative_to(relative_dir))
        )
        return {
            key: {
                "size": stat.st_size,
                "atime": stat.st_atime,
                "mtime": stat.st_mtime,
                "ctime": stat.st_ctime,
                "hash": File.get_file_hash(path),
            }
        }

    @staticmethod
    def get_dir_stat(path: Path) -> dict[str, dict]:
        """递归收集文件或目录下所有文件的统计信息。"""
        stat_dict: dict[str, dict] = {}
        if path.is_file():
            stat_dict.update(File.get_file_stat(path))
        elif path.is_dir():
            # Path.walk() 比手写递归更高效
            for dirpath, _dirnames, filenames in path.walk():
                for filename in filenames:
                    file_path = dirpath / filename
                    stat_dict.update(File.get_file_stat(file_path, relative_dir=path))
        return stat_dict

    @staticmethod
    def rename_by_dir(dir_path: Path) -> None:
        """把所有条目重命名为“目录名_原文件名”。"""
        for file in dir_path.iterdir():
            new_name = f"{dir_path.name}_{file.name}"
            file.rename(dir_path / new_name)

    @staticmethod
    def mv_file_type_to_dir(old_dir: Path, new_dir: Path) -> None:
        """把 old_dir 中的文件按后缀分类移动到 new_dir。"""
        file_list = list(old_dir.rglob("*hh*/*"))
        file_list = [f for f in file_list if f.is_file()]
        file_type_set: set[str] = set()
        for idx, file in enumerate(file_list):
            suffix = file.suffix
            if suffix not in file_type_set:
                (new_dir / suffix).mkdir(parents=True, exist_ok=True)
                file_type_set.add(suffix)
            dst = new_dir / suffix / file.name
            try:
                file.rename(dst)
            except FileExistsError:
                file.rename(new_dir / suffix / f"{idx}_{file.name}")
        print(f"File types: {file_type_set}")
        print(f"Files moved: {len(file_list)}")

    @staticmethod
    def cp_file_type_dir(
        old_dir: Path,
        new_dir: Path,
        dir_name: bool = False,
        show_msg: Callable[..., None] | None = None,
    ) -> None:
        """把 old_dir 中的文件按后缀分类复制到 new_dir。"""
        show_msg = show_msg or print
        # Path.walk() 直接给出文件名，无需再过滤 is_file()
        file_list: list[Path] = []
        for _dirpath, _dirnames, filenames in old_dir.walk():
            for name in filenames:
                file_list.append(_dirpath / name)
        file_type_set: set[str] = set()

        if dir_name:
            for file in file_list:
                shutil.copyfile(file, new_dir / file.name)
        else:
            for idx, file in enumerate(file_list):
                suffix = file.suffix
                if suffix not in file_type_set:
                    (new_dir / suffix).mkdir(parents=True, exist_ok=True)
                    file_type_set.add(suffix)
                dst = new_dir / suffix / file.name
                try:
                    shutil.copyfile(file, dst)
                except shutil.SameFileError:
                    shutil.copyfile(file, new_dir / suffix / f"{idx}_{file.name}")
        show_msg(f"File types: {file_type_set}")
        show_msg(f"Files copied: {len(file_list)}")

    # ---- 重复文件检测 ----

    @staticmethod
    def repeat_check(
        file_list: list[Path], show_msg: Callable[..., None] | None = None
    ) -> dict[Path, list[Path]]:
        """按 MD5 哈希检测重复文件。"""
        show_msg = show_msg or print
        hash_to_file: dict[str, Path] = {}
        duplicate_dict: dict[Path, list[Path]] = {}
        for idx, file in enumerate(file_list, 1):
            # 分块读取，避免把整个文件载入内存
            file_hash = File.get_file_hash(file, hash_type='md5')
            if idx % 50 == 0:
                show_msg(f"Checked {idx} files")
            if file_hash not in hash_to_file:
                hash_to_file[file_hash] = file
            else:
                original = hash_to_file[file_hash]
                duplicate_dict.setdefault(original, []).append(file)
                show_msg(f"{file} duplicates {original}")
        return duplicate_dict

    # ---- 目录比对 ----

    @staticmethod
    def compare_file_dir(
        file_dir_1: Path,
        file_dir_2: Path,
        suffix: str = ".txt",
        show_msg: Callable[..., None] | None = None,
    ) -> None:
        """按内容检查 file_dir_1 是否覆盖 file_dir_2 的全部文件。"""
        show_msg = show_msg or print
        # iterdir() 只取顶层文件（rglob 会深入到子目录）
        file_set_1 = {f.name for f in file_dir_1.iterdir() if f.is_file()}
        file_not_in_set: set[Path] = set()
        is_covered = True
        for file in file_dir_2.iterdir():
            if not file.is_file():
                continue
            if file.name in file_set_1:
                hash_1 = File.get_file_hash(file_dir_1 / file.name)
                hash_2 = File.get_file_hash(file)
                if hash_1 != hash_2:
                    show_msg(f"Hash mismatch: {file.name} | {hash_1} | {hash_2}")
            else:
                file_not_in_set.add(file)
                is_covered = False
                show_msg(file)
        if is_covered:
            show_msg("--- Fully covered ---")
        else:
            show_msg(f"Uncovered files: {len(file_not_in_set)}")

    # ---- 打包归档 ----

    @staticmethod
    def make_archive_for_dir(src_dir: str | Path, dst_path: str | Path) -> None:
        """把目录打包为 zip。"""
        shutil.make_archive(
            str(Path(dst_path).with_suffix("")), "zip", str(Path(src_dir))
        )

    @staticmethod
    def make_archive(file_list: list[Path], dst_path: Path) -> None:
        """按文件列表打包 zip，保留目录结构。"""
        dst_path = Path(dst_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.unlink(missing_ok=True)
        with zipfile.ZipFile(
            dst_path, mode="a", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            for file in file_list:
                file = Path(file)
                if file.is_file():
                    zf.write(file, arcname=file.name)
                elif file.is_dir():
                    for _dirpath, _dirnames, filenames in file.walk():
                        for name in filenames:
                            item = _dirpath / name
                            zf.write(item, arcname=str(item.relative_to(file.parent)))

    # ---- 异步 IO ----

    @staticmethod
    async def save(file_obj, file_path: str) -> None:
        """把上传的文件异步写入磁盘。"""
        import aiofiles

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(await file_obj.read())

    @staticmethod
    def file_dir_repeat(file_dir: Path, copy_nums: int) -> None:
        """数据增广：把目录内每个文件复制 N 份。"""
        files = list(file_dir.iterdir())
        for f in files:
            if not f.is_file():
                raise ValueError(f"Not a file: {f}")
        for f in files:
            for i in range(1, copy_nums + 1):
                new_file = f.parent / f"{f.stem}_copy_{i}{f.suffix}"
                shutil.copyfile(f, new_file)

    @staticmethod
    def rename_file_by_dir(file_dir: Path, level: int = 1) -> None:
        """按父目录名批量重命名文件。

        level=1：重命名为“父目录名_原文件名”
        level=2：递归一层子目录，重命名为“子目录名_原文件名”
        """
        file_dir = Path(file_dir)
        if level == 1:
            for item in file_dir.iterdir():
                item.rename(item.parent / f"{file_dir.name}_{item.name}")
        elif level == 2:
            for item in file_dir.iterdir():
                if item.is_dir():
                    for sub_file in item.iterdir():
                        sub_file.rename(
                            sub_file.parent / f"{item.name}_{sub_file.name}"
                        )
