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
# Windows – use shell32 SHFileOperationW (FO_DELETE + FOF_ALLOWUNDO)
# ---------------------------------------------------------------------------

# Ultralytics Platform: JPEG、PNG、WebP、BMP、TIFF、HEIC、AVIF、JP2、DNG、MPO
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
# Ultralytics Platform: MP4、WebM、MOV、AVI、MKV、M4V
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


def _delete_to_trash_windows(path: Path) -> None:
    """Send *path* to the Windows recycle bin via the shell32 API."""
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

    # pFrom requires a double-null-terminated string
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
# Linux – implement the FreeDesktop.org Trash specification
# ---------------------------------------------------------------------------


def _get_mount_point(p: Path) -> Path:
    """Walk upward to find the mount point of *p*."""
    p = p.resolve()
    while not p.is_mount():
        parent = p.parent
        if parent == p:  # reached root
            break
        p = parent
    return p


def _find_trash_dir(path: Path) -> Path:
    """Return the trash directory that should be used for *path*."""
    home_trash = Path.home() / '.local' / 'share' / 'Trash'
    path_mount = _get_mount_point(path)
    home_mount = _get_mount_point(Path.home())

    # Same partition → home trash
    if path_mount == home_mount:
        return home_trash

    # Different partition → try .Trash-<uid> or .Trash/<uid>
    uid = str(path.stat().st_uid)
    alt_trash = path_mount / f'.Trash-{uid}'

    if alt_trash.is_dir():
        return alt_trash

    alt_trash2 = path_mount / '.Trash' / uid
    if alt_trash2.is_dir():
        return alt_trash2

    # Neither exists → create .Trash-<uid> at the mount root
    alt_trash.mkdir(parents=True, exist_ok=True)
    return alt_trash


def _unique_trash_name(files_dir: Path, name: str) -> Path:
    """Return a non-existing destination path inside *files_dir*."""
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
    """Send *path* to the Linux Trash (FreeDesktop spec)."""
    path = path.resolve()
    trash_dir = _find_trash_dir(path)
    files_dir = trash_dir / 'files'
    info_dir = trash_dir / 'info'

    files_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)

    dest = _unique_trash_name(files_dir, path.name)

    # Write .trashinfo *before* moving (atomicity in spirit)
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
# macOS – use AppleScript to talk to the Finder
# ---------------------------------------------------------------------------


def _delete_to_trash_macos(path: Path) -> None:
    """Send *path* to the macOS Trash via Finder."""
    import subprocess

    script = f'tell application "Finder" to delete POSIX file "{path.resolve()}"'
    result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
    if result.returncode != 0:
        raise OSError(f"osascript failed: {result.stderr.strip()}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def delete(path: str | Path) -> None:
    """
    Move *path* to the system recycle bin / trash.

    This is the moral equivalent of "Delete" in a GUI file manager – the
    file or folder can be restored later.

    Args:
        path: A file or directory to delete.

    Raises:
        FileNotFoundError: *path* does not exist.
        NotImplementedError: Running on an unsupported operating system.
        OSError: The underlying OS call failed.
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
    """Yield file content chunk by chunk."""
    while chunk := file.read(size):
        yield chunk


def hash_file(filename: str | Path) -> str:
    """Compute MD5 hash of a file."""
    filename = Path(filename)
    h = hashlib.md5()
    with filename.open("rb") as f:
        for chunk in _read_chunks(f, 8192):
            h.update(chunk)
    return h.hexdigest()


def find_duplicates(folder: str | Path) -> list[tuple[str, str]]:
    """Find duplicate files in a folder by MD5 hash."""
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
        """Delete file, ignore if missing."""
        path.unlink(missing_ok=True)

    @staticmethod
    def delete_dir(path: Path) -> None:
        """Recursively delete directory, ignore errors."""
        shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def delete_to_trash(path: Path) -> None:
        """Move file to system trash / recycle bin."""
        delete(path)

    @staticmethod
    def get_file_hash(
        file_path: Path, blocksize: int = 1 << 20, hash_type: str = 'sha256'
    ) -> str:
        """Compute file hash (sha256 or md5)."""
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
        """Copy file; keep_state=True preserves metadata."""
        if keep_state:
            shutil.copy2(file_path, new_file_path)
        else:
            shutil.copy(file_path, new_file_path)

    @staticmethod
    def cp_dir(src_dir: Path, dst_dir: Path) -> None:
        """Copy entire directory tree (preserves metadata)."""
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    @staticmethod
    def get_size(file_path: Path) -> float:
        """Get file size in bytes, rounded to 2 decimal places."""
        return round(file_path.stat().st_size, 2)

    @staticmethod
    def get_size_mb(file_path: Path) -> float:
        """Get file size in MB, rounded to 2 decimal places."""
        return round(file_path.stat().st_size / 1024 / 1024, 2)

    @staticmethod
    def set_file_time(path: Path, atime: float, mtime: float) -> None:
        """Set file access and modification times."""
        path.touch()
        utime(path, (atime, mtime))  # pathlib has no utime; keep os.utime

    @staticmethod
    def is_img_file(file_path: Path) -> bool:
        """Check if file is an image by extension."""
        return file_path.suffix in IMG_SUFFIXS

    @staticmethod
    def get_txt_path(file_path: Path) -> Path:
        """Return the corresponding .txt path for a file."""
        return file_path.with_suffix(".txt")

    # ---- scanning / collection ----

    @staticmethod
    def get_all_suffix_file(
        file_dir: Path, suffix_set: set[str] | None = None
    ) -> list[Path]:
        """Get all files under file_dir matching any suffix in suffix_set."""
        suffix_set = suffix_set or {".jpg"}
        file_list: list[Path] = []
        # single pass with Path.walk() (3.12+) instead of one rglob per suffix
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
        """Group files by suffix: {'.ext': [Path, ...]}."""
        suffix_file_dict: dict[str, list[Path]] = {}
        # Path.walk() provides files directly, no is_file() filter needed
        for _dirpath, _dirnames, filenames in file_dir.walk():
            for name in filenames:
                fp = _dirpath / name
                if suffix_set is None or fp.suffix in suffix_set:
                    suffix_file_dict.setdefault(fp.suffix, []).append(fp)
        return suffix_file_dict

    @staticmethod
    def get_img_file_list(file_path: Path) -> list[Path]:
        """Get all image files under a directory."""
        return File.get_all_suffix_file(file_path, suffix_set=File.img_suffixs)

    @staticmethod
    def get_file_suffix_set(file_dir: str | Path) -> set[str]:
        """Get the set of file extensions (top-level only)."""
        file_dir = Path(file_dir)
        return {f.suffix for f in file_dir.iterdir() if f.is_file()}

    @staticmethod
    def get_file_stat(path: Path, relative_dir: Path | None = None) -> dict[str, dict]:
        """Return {key: {size, atime, mtime, ctime, hash}} for a single file."""
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
        """Recursively collect file stats for a file or directory."""
        stat_dict: dict[str, dict] = {}
        if path.is_file():
            stat_dict.update(File.get_file_stat(path))
        elif path.is_dir():
            # Path.walk() is more efficient than manual recursion
            for dirpath, _dirnames, filenames in path.walk():
                for filename in filenames:
                    file_path = dirpath / filename
                    stat_dict.update(File.get_file_stat(file_path, relative_dir=path))
        return stat_dict

    @staticmethod
    def rename_by_dir(dir_path: Path) -> None:
        """Rename all items to 'dirname_originalname'."""
        for file in dir_path.iterdir():
            new_name = f"{dir_path.name}_{file.name}"
            file.rename(dir_path / new_name)

    @staticmethod
    def mv_file_type_to_dir(old_dir: Path, new_dir: Path) -> None:
        """Move files from old_dir into new_dir, grouped by extension."""
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
        """Copy files from old_dir into new_dir, grouped by extension."""
        show_msg = show_msg or print
        # Path.walk() provides files directly, no is_file() filter needed
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

    # ---- duplicate detection ----

    @staticmethod
    def repeat_check(
        file_list: list[Path], show_msg: Callable[..., None] | None = None
    ) -> dict[Path, list[Path]]:
        """Detect duplicate files by MD5 hash."""
        show_msg = show_msg or print
        hash_to_file: dict[str, Path] = {}
        duplicate_dict: dict[Path, list[Path]] = {}
        for idx, file in enumerate(file_list, 1):
            # chunked read avoids loading whole file into memory
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

    # ---- directory comparison ----

    @staticmethod
    def compare_file_dir(
        file_dir_1: Path,
        file_dir_2: Path,
        suffix: str = ".txt",
        show_msg: Callable[..., None] | None = None,
    ) -> None:
        """Check whether file_dir_1 covers all files in file_dir_2 by content."""
        show_msg = show_msg or print
        # iterdir() limits to top-level files (rglob would descend into subdirs)
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

    # ---- archiving ----

    @staticmethod
    def make_archive_for_dir(src_dir: str | Path, dst_path: str | Path) -> None:
        """Create a zip archive from a directory."""
        shutil.make_archive(
            str(Path(dst_path).with_suffix("")), "zip", str(Path(src_dir))
        )

    @staticmethod
    def make_archive(file_list: list[Path], dst_path: Path) -> None:
        """Create a zip from a file list, preserving directory structure."""
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

    # ---- async I/O ----

    @staticmethod
    async def save_file(file_obj, file_path: str) -> None:
        """Save an uploaded file to disk asynchronously."""
        import aiofiles

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(await file_obj.read())

    @staticmethod
    def file_dir_repeat(file_dir: Path, copy_nums: int) -> None:
        """Data augmentation: copy every file in a folder N times."""
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
        """Batch-rename files by parent directory name.

        level=1: rename to 'parentdir_originalname'
        level=2: recurse into subdirs, rename to 'subdir_originalname'
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
