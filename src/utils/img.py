"""Image utilities: header-based metadata extraction, I/O, and format conversion.

Detects size / DPI / color count / channel count from binary file headers.
Supported: GIF, PNG, JPEG, JPEG2000, HEIF/AVIF, TIFF, BigTIFF, SVG, Netpbm,
WebP, BMP.

Optimised for Python ≥3.14 — uses PEP 695 ``type`` aliases, precompiled
``struct.Struct``, ``memoryview``, and ``contextlib.suppress``.
"""

from __future__ import annotations

import base64
import io
import re
import struct
from contextlib import suppress
from os import PathLike
from pathlib import Path
from typing import BinaryIO, NamedTuple, Protocol, runtime_checkable
from urllib.parse import urlparse
from urllib.request import urlopen

import cv2
import numpy as np
from PIL import Image

from .my_types import ImgSize

# ============================================================
# Precompiled patterns, constants & structs
# ============================================================

_RE_SVG_W = re.compile(r'[^-]width="(.*?)"')
_RE_SVG_H = re.compile(r'[^-]height="(.*?)"')
_RE_CSS = re.compile(r"(\d+(?:\.\d+)?)?([a-z]*)$")

_UNIT_DENSITY: dict[int, float] = {
    -3: 0.000_025_4,
    -2: 0.000_254,
    -1: 0.002_54,  # km, hm, dam
    0: 0.025_4,  # m
    1: 0.254,
    2: 2.54,
    3: 25.4,
    4: 254.0,
    5: 2_540.0,
    6: 25_400.0,
}

_CSS_PX: dict[str, float] = {
    "cm": 96 / 2.54,
    "mm": 96 / 25.4,
    "in": 96,
    "pc": 96 / 6,
    "pt": 96 / 72,
    "px": 1,
}

_HEIF_BRANDS: frozenset[bytes] = frozenset(
    {
        b"avif",
        b"avis",
        b"heic",
        b"heix",
        b"hevc",
        b"hevx",
        b"mif1",
        b"msf1",
    }
)
_JPEG_NO_SOF: frozenset[int] = frozenset({0xC4, 0xC8, 0xCC})

# Magic bytes
_B_PNG = b"\x89PNG\r\n\x1a\n"
_B_JPEG = b"\xff\xd8"
_B_JP2 = b"\x00\x00\x00\x0cjP  \r\n\x87\n"
_B_TIFF_BE = b"\x4d\x4d\x00\x2a"
_B_TIFF_LE = b"\x49\x49\x2a\x00"
_B_BIGTIFF = b"\x49\x49\x2b\x00"
_GIF87a, _GIF89a = b"GIF87a", b"GIF89a"

_PNG_CH: dict[int, int] = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_PNG_CL: dict[int, int] = {0: 1, 2: 3, 3: 1, 4: 1, 6: 3}

# ---- Precompiled struct helpers (significant perf win) ----
_S = struct.Struct
_U16_BE = _S(">H")
_U32_BE = _S(">L")
_U64_BE = _S(">Q")
_U16_LE = _S("<H")
_U32_LE = _S("<L")
_U64_LE = _S("<Q")
_I16_LE = _S("<h")
_2U16_BE = _S(">HH")
_2U32_BE = _S(">LL")
_2I16_LE = _S("<hh")
_U32U32U8 = _S(">LLB")
_U8U16U16 = _S(">BHH")
_2U162U8 = _S(">HHBB")
_TIFF_TAG_LE = _S("<HHII")
_TIFF_IFD_LE = _S("<HHQQ")
_BMP_WH = _S("<ii")
_BMP_BD = _S("<H")


# ============================================================
# Protocols & types
# ============================================================


@runtime_checkable
class ReadSeekBinary(Protocol):
    """Readable, seekable binary stream."""

    def read(self, size: int = -1, /) -> bytes: ...
    def seek(self, offset: int, whence: int = 0, /) -> int: ...
    def readline(self, size: int = -1, /) -> bytes: ...
    def close(self) -> None: ...


type PathInput = str | bytes | PathLike[str] | PathLike[bytes]
type FileInput = PathInput | BinaryIO | ReadSeekBinary


class ImageInfo(NamedTuple):
    """Image metadata."""

    width: int = -1
    height: int = -1
    xdpi: int = -1
    ydpi: int = -1
    colors: int = -1
    channels: int = -1


# ============================================================
# General helpers
# ============================================================


def _open_file(path: FileInput) -> tuple[BinaryIO | ReadSeekBinary, bool]:
    """Open a path / URL / file-like → (handle, should_close)."""
    match path:
        case BinaryIO() | ReadSeekBinary():
            return path, False
        case str() | bytes():
            if isinstance(path, str) and urlparse(path).scheme in {"http", "https"}:
                with urlopen(path) as r:
                    return io.BytesIO(r.read()), True
            return open(path, "rb"), True
        case _:
            return open(path, "rb"), True


def _to_dpi(density: int, unit: int) -> int:
    """Convert *density* in TIFF/pHYS unit → standard DPI."""
    return (
        int(density * _UNIT_DENSITY.get(unit, 1) + 0.5)
        if unit in _UNIT_DENSITY
        else density
    )


def _css_to_px(value: str) -> float:
    """'2cm' | '100px' → px at 96 DPI."""
    if m := _RE_CSS.fullmatch(value):
        n, u = m.group(1), m.group(2) or "px"
        return (float(n) if n else 0.0) * _CSS_PX.get(u, 0)
    raise ValueError(f"Unknown length: {value!r}")


def _read_jpeg_seg(f: BinaryIO | ReadSeekBinary) -> tuple[int, int]:
    """Read JPEG segment → (marker, data_size)."""
    while (b := f.read(1)) == b"\xff":
        pass
    if not b:
        raise ValueError("JPEG ended unexpectedly")
    size = _U16_BE.unpack(f.read(2))[0] - 2
    if size < 0:
        raise ValueError("Invalid JPEG segment size")
    return b[0], size


def _seek_jpeg_sof(f: BinaryIO | ReadSeekBinary) -> None:
    """Fast-forward to JPEG SOF marker."""
    size, m = 2, 0
    while not (0xC0 <= m <= 0xCF and m not in _JPEG_NO_SOF):
        f.seek(size, 1)
        m, size = _read_jpeg_seg(f)


# ============================================================
# ISO box parser (HEIF / AVIF shared)
# ============================================================


def _iso_boxes(data: memoryview, start: int, end: int):
    """Yield (offset, size, box_type, header_size) for ISO BMFF boxes."""
    off = start
    while off + 8 <= end:
        sz = int.from_bytes(data[off : off + 4], "big")
        bt = bytes(data[off + 4 : off + 8])
        hdr = (
            16
            if sz == 1
            and off + 16 <= end
            and (sz := int.from_bytes(data[off + 8 : off + 16], "big"))
            else 8
        )
        if sz == 0:
            sz = end - off
        if sz < 8 or off + sz > end:
            return
        yield off, sz, bt, hdr
        off += sz


def _heif_size(f: BinaryIO | ReadSeekBinary) -> tuple[int, int]:
    """Extract width & height from HEIF / AVIF metadata."""
    f.seek(0)
    d = memoryview(f.read())
    n = len(d)

    # locate 'meta'
    meta = next(((o, s, h) for o, s, t, h in _iso_boxes(d, 0, n) if t == b"meta"), None)
    if not meta:
        return -1, -1
    mo, ms, mh = meta
    ms_lo, ms_hi = mo + mh + 4, mo + ms

    props: list[tuple[int, int, bytes, int]] = []
    assoc: dict[int, list[int]] = {}
    primary = None

    for off, sz, bt, hdr in _iso_boxes(d, ms_lo, ms_hi):
        ps, pe = off + hdr, off + sz
        if bt == b"pitm":
            if d[ps] == 0 and ps + 6 <= pe:
                primary = int.from_bytes(d[ps + 4 : ps + 6], "big")
            elif d[ps] > 0 and ps + 8 <= pe:
                primary = int.from_bytes(d[ps + 4 : ps + 8], "big")
        elif bt == b"iinf" and ps + 6 <= pe:
            if d[ps] == 0 and ps + 6 <= pe:
                cur = ps + 6
                cnt = _U16_BE.unpack(d[ps + 4 : ps + 6])[0]
            elif ps + 8 <= pe:
                cur = ps + 8
                cnt = _U32_BE.unpack(d[ps + 4 : ps + 8])[0]
            else:
                continue
            for _ in range(cnt):
                if cur + 8 > pe:
                    break
                esz = int.from_bytes(d[cur : cur + 4], "big")
                if cur + esz > pe:
                    break
                if d[cur + 4 : cur + 8] == b"infe":
                    pass  # item entry – skip for brevity (not needed if we use iprp)
                cur += esz
        elif bt == b"iprp":
            for po, ps2, pt, ph in _iso_boxes(d, ps, pe):
                pps, ppe = po + ph, po + ps2
                if pt == b"ipco":
                    props = list(_iso_boxes(d, pps, ppe))
                elif pt == b"ipma" and pps + 8 <= ppe:
                    large = bool(int.from_bytes(d[pps + 1 : pps + 4], "big") & 1)
                    cur = pps + 4
                    if cur + 4 > ppe:
                        continue
                    ec = int.from_bytes(d[cur : cur + 4], "big")
                    cur += 4
                    for _ in range(ec):
                        if cur + (3 if large else 2) > ppe:
                            break
                        iid = int.from_bytes(d[cur : cur + 2], "big")
                        cur += 2
                        ac = d[cur]
                        cur += 1
                        lst: list[int] = []
                        for _ in range(ac):
                            if large:
                                if cur + 2 > ppe:
                                    break
                                v = int.from_bytes(d[cur : cur + 2], "big")
                                cur += 2
                                lst.append(v & 0x7FFF)
                            else:
                                if cur >= ppe:
                                    break
                                lst.append(d[cur] & 0x7F)
                                cur += 1
                        if lst:
                            assoc[iid] = lst

    if not props:
        return -1, -1

    targets = (
        assoc.get(primary) if primary is not None else list(range(1, len(props) + 1))
    ) or list(range(1, len(props) + 1))

    for idx in targets:
        if 1 <= idx <= len(props):
            po, psz, pt, ph = props[idx - 1]
            if pt == b"ispe" and (pps := po + ph) + 12 <= po + psz:
                return (
                    int.from_bytes(d[pps + 4 : pps + 8], "big"),
                    int.from_bytes(d[pps + 8 : pps + 12], "big"),
                )
    return -1, -1


# ============================================================
# Size detection
# ============================================================


def _get_size(f: BinaryIO | ReadSeekBinary) -> tuple[int, int]:
    """Detect (width, height) from file header."""
    f.seek(0)
    head = memoryview(f.read(64))
    sz = len(head)

    # ----- fast 2-byte checks first -----
    if sz < 2:
        return -1, -1

    if head[:2] == b"\xff\xd8":
        f.seek(0)
        _seek_jpeg_sof(f)
        f.seek(1, 1)
        h, w = _2U16_BE.unpack(f.read(4))
        return w, h

    if sz < 8:
        return -1, -1

    b8 = bytes(head[:8])

    # ----- 4–6 byte signatures -----
    if sz >= 10 and b8[:6] in (_GIF87a, _GIF89a):
        return _2I16_LE.unpack(head[6:10])

    if b8[:4] == b"\x89PNG" and sz >= 24:
        off = 16 if b8[4:8] == b"\r\n\x1a\n" and head[12:16] == b"IHDR" else 8
        return _2U32_BE.unpack(head[off : off + 8])

    if b8[:4] == _B_JP2[:4]:
        f.seek(48)
        h, w = _2U32_BE.unpack(f.read(8))
        return w, h

    if b8[:4] == _B_TIFF_BE[:4]:
        return _tiff_size(f, head, ">")
    if b8[:4] == _B_TIFF_LE[:4]:
        return _tiff_size(f, head, "<")
    if b8[:4] == _B_BIGTIFF[:4]:
        return _bigtiff_size(f, head)

    if b8[4:8] == b"ftyp":
        ftyp_sz = int.from_bytes(head[:4], "big")
        if ftyp_sz < 8:
            raise ValueError("Invalid HEIF")
        f.seek(8)
        if any(b in f.read(ftyp_sz - 8) for b in _HEIF_BRANDS):
            w, h = _heif_size(f)
            if w != -1 and h != -1:
                return w, h
            raise ValueError("Invalid HEIF: no dimensions")

    if b8[:5] in (b"<?xml", b"<svg "):
        f.seek(0)
        data = f.read(1024).decode("utf-8", errors="replace")
        if (wm := _RE_SVG_W.search(data)) and (hm := _RE_SVG_H.search(data)):
            return int(_css_to_px(wm.group(1))), int(_css_to_px(hm.group(1)))
        raise ValueError("SVG: missing width/height")

    if head[:1] == b"P" and b8[1:2] in b"123456":
        return _netpbm_size(f)

    if b8[:4] == b"RIFF" and b8[4:8] == b"WEBP":
        return _webp_size(head)

    if b8[:2] == b"BM":
        return _BMP_WH.unpack(head[18:26])

    return -1, -1


def _tiff_size(
    f: BinaryIO | ReadSeekBinary, head: memoryview, e: str
) -> tuple[int, int]:
    """TIFF width & height."""
    offset = struct.unpack(f">{e}L" if e == ">" else f"<{e}L", head[4:8])[0]
    f.seek(offset)
    cnt = struct.unpack(f">{e}H" if e == ">" else f"<{e}H", f.read(2))[0]
    w = h = -1
    fmt = f">{e}HHLL" if e == ">" else f"<{e}HHLL"
    for _ in range(cnt):
        tag, dt, _, v = struct.unpack(fmt, f.read(12))
        match tag:
            case 256:
                w = int(v / 65536) if dt == 3 else v
            case 257:
                h = int(v / 65536) if dt == 3 else v
        if w != -1 and h != -1:
            return w, h
    raise ValueError("TIFF: missing width/height")


def _bigtiff_size(f: BinaryIO | ReadSeekBinary, head: memoryview) -> tuple[int, int]:
    """BigTIFF width & height."""
    if _U32_LE.unpack(head[4:8])[0] != 8:
        raise ValueError("BigTIFF: bad offset")
    f.seek(_U64_LE.unpack(head[8:16])[0])
    cnt = _U64_LE.unpack(f.read(8))[0]
    w = h = -1
    for _ in range(cnt):
        tag, _, _, v = _TIFF_IFD_LE.unpack(f.read(20))
        match tag:
            case 256:
                w = v
            case 257:
                h = v
        if w != -1 and h != -1:
            return w, h
    raise ValueError("BigTIFF: missing width/height")


def _netpbm_size(f: BinaryIO | ReadSeekBinary) -> tuple[int, int]:
    """Netpbm width & height."""
    f.seek(2)
    nums: list[int] = []
    while True:
        if not (ch := f.read(1)):
            raise ValueError("Truncated Netpbm")
        if ch.isspace():
            continue
        if ch == b"#":
            f.readline()
            continue
        if not ch.isdigit():
            raise ValueError(f"Netpbm: bad char {ch!r}")
        val = ch
        while (nxt := f.read(1)).isdigit():
            val += nxt
        nums.append(int(val))
        if len(nums) == 2:
            return nums[0], nums[1]
        f.seek(-1, 1)


def _webp_size(head: memoryview) -> tuple[int, int]:
    """WebP width & height."""
    sub = bytes(head[12:16])
    if sub == b"VP8 ":
        return _U16_LE.unpack_from(head, 26)
    if sub == b"VP8X":
        return (
            int.from_bytes(bytes(head[24:27]) + b"\0", "little") + 1,
            int.from_bytes(bytes(head[27:30]) + b"\0", "little") + 1,
        )
    if sub == b"VP8L":
        b = head[21:25]
        return (((b[1] & 63) << 8) | b[0]) + 1, (
            ((b[3] & 15) << 10) | (b[2] << 2) | ((b[1] & 192) >> 6)
        ) + 1
    raise ValueError("Unsupported WebP format")


# ============================================================
# DPI detection
# ============================================================


def _get_dpi(f: BinaryIO | ReadSeekBinary) -> tuple[int, int]:
    """Detect (xdpi, ydpi) from file header."""
    f.seek(0)
    head = f.read(24)
    sz = len(head)

    if sz >= 24 and head.startswith(_B_PNG):
        return _png_dpi(f)
    if sz >= 2 and head.startswith(_B_JPEG):
        try:
            return _jpeg_dpi(f)
        except (struct.error, ValueError):
            raise ValueError("Invalid JPEG")
    if sz >= 12 and head.startswith(_B_JP2):
        try:
            return _jp2_dpi(f)
        except struct.error:
            raise ValueError("Invalid JPEG2000")
    return -1, -1


def _png_dpi(f: BinaryIO | ReadSeekBinary) -> tuple[int, int]:
    """PNG pHYs chunk → DPI."""
    off = 8
    while True:
        f.seek(off)
        chunk = memoryview(f.read(17))
        if len(chunk) < 8:
            return -1, -1
        ct = bytes(chunk[4:8])
        if ct == b"pHYs":
            xd, yd, u = _U32U32U8.unpack(chunk[8:])
            return (_to_dpi(xd, 0), _to_dpi(yd, 0)) if u else (xd, yd)
        if ct == b"IDAT":
            return -1, -1
        off += int.from_bytes(chunk[:4], "big") + 12


def _jpeg_dpi(f: BinaryIO | ReadSeekBinary) -> tuple[int, int]:
    """JPEG APP0/JFIF → DPI."""
    f.seek(0)
    size, m = 2, 0
    while not (0xC0 <= m <= 0xCF):
        f.seek(size, 1)
        m, size = _read_jpeg_seg(f)
        if m == 0xE0:
            f.seek(7, 1)
            u, xd, yd = _U8U16U16.unpack(f.read(5))
            match u:
                case 0 | 1:
                    return xd, yd
                case 2:
                    return _to_dpi(xd, 2), _to_dpi(yd, 2)
            break
    return -1, -1


def _jp2_dpi(f: BinaryIO | ReadSeekBinary) -> tuple[int, int]:
    """JPEG2000 resolution box → DPI."""
    f.seek(32)
    left = _U32_BE.unpack(f.read(4))[0] - 8
    f.seek(4, 1)
    while left > 0:
        hdr = f.read(8)
        bt = hdr[4:]
        if bt == b"res ":
            left -= 8
            while left > 0:
                h2 = f.read(8)
                bt2 = h2[4:]
                if bt2 == b"resd":
                    yd, xd, yu, xu = _2U162U8.unpack(f.read(10))
                    return _to_dpi(xd, xu), _to_dpi(yd, yu)
                bsz = _U32_BE.unpack(h2[:4])[0]
                f.seek(bsz - 8, 1)
                left -= bsz
            break
        bsz = _U32_BE.unpack(hdr[:4])[0]
        f.seek(bsz - 8, 1)
        left -= bsz
    return -1, -1


# ============================================================
# Colour / channel detection
# ============================================================


def _get_colors(f: BinaryIO | ReadSeekBinary) -> int:
    """Detect colour count from header."""
    f.seek(0)
    h = memoryview(f.read(32))
    sz = len(h)

    if sz >= 11 and bytes(h[:6]) in (_GIF87a, _GIF89a) and h[10] & 0x80:
        return 1 << ((h[10] & 0x07) + 1)
    if (
        sz >= 26
        and h[:4] == _B_PNG[:4]
        and bytes(h[12:16]) == b"IHDR"
        and (nc := _PNG_CL.get(h[25]))
    ):
        return 1 << (h[24] * nc)
    return -1


def _get_channels(f: BinaryIO | ReadSeekBinary) -> int:
    """Detect channel count from header."""
    f.seek(0)
    h = memoryview(f.read(32))
    sz = len(h)

    if sz >= 26 and h[:4] == _B_PNG[:4] and bytes(h[12:16]) == b"IHDR":
        return _PNG_CH.get(h[25], -1)
    if sz >= 2 and bytes(h[:2]) == _B_JPEG:
        try:
            f.seek(0)
            _seek_jpeg_sof(f)
            f.seek(5, 1)
            return f.read(1)[0]
        except (struct.error, ValueError):
            raise ValueError("Invalid JPEG")
    if sz >= 11 and bytes(h[:6]) in (_GIF87a, _GIF89a):
        return 3
    if sz >= 28 and bytes(h[:2]) == b"BM":
        bd = _BMP_BD.unpack_from(h, 28)[0]
        return {8: 1, 24: 3, 32: 4, 16: 3}.get(bd, -1) if bd <= 32 else -1
    return -1


# ============================================================
# Public API
# ============================================================


def get_info(
    path: FileInput,
    *,
    size: bool = True,
    dpi: bool = True,
    colors: bool = True,
    channels: bool = True,
) -> ImageInfo:
    """Extract full image metadata from a file path, URL, or file-like object."""
    fh, do_close = _open_file(path)
    try:
        w = h = xd = yd = cc = ch = -1
        if size:
            w, h = _get_size(fh)
        if dpi:
            xd, yd = _get_dpi(fh)
        if colors:
            cc = _get_colors(fh)
        if channels:
            ch = _get_channels(fh)
        return ImageInfo(w, h, xd, yd, cc, ch)
    finally:
        if do_close:
            fh.close()


def get(path: FileInput) -> tuple[int, int]:
    """Return (width, height)."""
    with suppress(Exception):
        return get_info(path, size=True, dpi=False, colors=False, channels=False)[:2]
    return -1, -1


def get_dpi(path: FileInput) -> tuple[int, int]:
    """Return (horizontal DPI, vertical DPI)."""
    with suppress(Exception):
        info = get_info(path, size=False, dpi=True, colors=False)
        return info.xdpi, info.ydpi
    return -1, -1


# ============================================================
# Img utility class
# ============================================================


class Img:
    """Image generation, I/O, format conversion, and processing."""

    # ---- Generation ----

    @staticmethod
    def generate_img(size: ImgSize, *, is_gray: bool = True) -> np.ndarray:
        shape = (size.height, size.width) if is_gray else (size.height, size.width, 3)
        return np.full(shape, 255, dtype=np.uint8)

    @staticmethod
    def generate_thumbnail(infile: Path, size: tuple[int, int]) -> None:
        img = Image.open(infile)
        img.thumbnail(size)
        img.save(f"{infile.stem}_thm_.jpg")

    @staticmethod
    def generate_thumbnail_dir(dir_: Path, size: int = 128) -> None:
        from threading import Thread

        for f in dir_.iterdir():
            Thread(
                target=Img.generate_thumbnail, args=(f, (size, size)), daemon=True
            ).start()

    # ---- I/O ----

    @staticmethod
    def load(path: str | Path, flags: int = -1) -> np.ndarray:
        data = np.fromfile(str(path), dtype=np.uint8)
        if (img := cv2.imdecode(data, flags)) is None:
            raise ValueError("Failed to decode image")
        return img

    @staticmethod
    def save(path: str | Path, img: np.ndarray) -> None:
        cv2.imencode(ext=Path(path).suffix, img=img)[1].tofile(str(path))

    @staticmethod
    def imread_svg(path: Path) -> list[str]:
        return path.read_text("utf-8").splitlines()

    @staticmethod
    def cv_show(img: np.ndarray, name: str = " ") -> None:
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def is_gray_img(img: np.ndarray) -> bool:
        b, g, r = (np.asarray(ch, np.int16) for ch in cv2.split(img))
        return ((r - g).var() + (g - b).var() + (b - r).var()) / 3.0 == 0

    @staticmethod
    def get_img_size(input: Path | np.ndarray) -> ImgSize:
        if isinstance(input, np.ndarray):
            h, w = input.shape[:2]
            return ImgSize(w, h)
        if isinstance(input, Path) and input.suffix.lower() == ".bmp":
            with Image.open(input) as img:
                return ImgSize(*img.size)
        fh, dc = _open_file(input)
        try:
            w, h = _get_size(fh)
            return ImgSize(w, h)
        finally:
            if dc:
                fh.close()

    # ---- Transforms ----

    @staticmethod
    def resize(img: np.ndarray, new_size: ImgSize) -> np.ndarray:
        return cv2.resize(img, (new_size.width, new_size.height))

    @staticmethod
    def crop(img: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
        return img[y0:y1, x0:x1]

    @staticmethod
    def crop_left_right(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cx = img.shape[1] // 2
        return img[:, :cx], img[:, cx:]

    @staticmethod
    def flip(img: np.ndarray, code: int = 0) -> np.ndarray:
        return cv2.flip(img, code)

    @staticmethod
    def rotate(img: np.ndarray, angle: int) -> np.ndarray:
        return np.rot90(img, (angle // 90) % 4) if angle % 90 == 0 else img

    @staticmethod
    def rotate_image(img: np.ndarray, angle: float, crop: bool) -> np.ndarray:
        h, w = img.shape[:2]
        angle %= 360
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(img, m, (w, h))
        if crop:
            a2 = angle % 180
            a = a2 if a2 <= 90 else 180 - a2
            t = a * np.pi / 180
            r = (h / w) if h > w else (w / h)
            cm = (np.cos(t) + np.sin(t) * np.tan(t)) / (r * np.tan(t) + 1)
            w_c, h_c = int(cm * w), int(cm * h)
            x0, y0 = (w - w_c) // 2, (h - h_c) // 2
            rotated = rotated[y0 : y0 + h_c, x0 : x0 + w_c]
        return rotated

    @staticmethod
    def resize_dir(src: Path, dst: Path, new_size: ImgSize) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}:
                Img.save(dst / f.name, Img.resize(Img.load(f), new_size))

    # ---- Conversion helpers ----

    class Convert:
        """Format conversion utilities."""

        @staticmethod
        def save_by_img_type(img_input, path: str | Path) -> None:
            match img_input:
                case Image.Image():
                    img_input.save(path)
                case np.ndarray():
                    Img.save(path, img_input)

        @staticmethod
        def pil_to_png_data(data: bytes) -> bytes:
            with io.BytesIO(data) as f:
                img = Image.open(f)
                with io.BytesIO() as out:
                    img.save(out, "PNG")
                    return out.getvalue()

        @staticmethod
        def bytes_to_pil(data: bytes) -> Image.Image:
            return Image.open(io.BytesIO(data))

        @staticmethod
        def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
            return Image.fromarray(
                cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB).astype(np.uint8)
            )

        @staticmethod
        def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
            return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

        @staticmethod
        def pil_to_bytes(pil_img: Image.Image, fmt: str = "PNG") -> bytes:
            buf = io.BytesIO()
            pil_img.save(buf, format=fmt)
            return buf.getvalue()

        @staticmethod
        def pil_to_base64(pil_img: Image.Image, fmt: str = "JPEG") -> str:
            return base64.b64encode(Img.Convert.pil_to_bytes(pil_img, fmt)).decode()

        @staticmethod
        def tensor_to_pil(tensor) -> Image.Image:
            from torchvision import transforms

            return transforms.ToPILImage()(tensor.cpu().clone().squeeze(0))

        @staticmethod
        def cv2_to_tensor(cv2_img: np.ndarray):
            from torch import from_numpy

            t = from_numpy(
                cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
            )
            return t.float().div(255).unsqueeze(0)

        @staticmethod
        def tensor_to_cv2(tensor) -> np.ndarray:
            return tensor.mul(255).byte().cpu().numpy().squeeze(0).transpose((1, 2, 0))

        @staticmethod
        def multi_cv2_to_tensor(cv2_img: np.ndarray):
            from torch import from_numpy

            return from_numpy(cv2_img.transpose((0, 3, 1, 2))).float().div(255)

        @staticmethod
        def path_to_base64(path: str | Path) -> str:
            return base64.b64encode(Path(path).read_bytes()).decode()

        @staticmethod
        def base64_to_path(b64: str, path: str | Path) -> None:
            Path(path).write_bytes(base64.b64decode(b64))

        @staticmethod
        def base64_to_pil(b64: str) -> Image.Image:
            return Image.open(io.BytesIO(base64.b64decode(b64.split(",", 1)[-1])))

        @staticmethod
        def cv2_to_base64(cv2_img: np.ndarray, fmt: str = ".jpg") -> str:
            return base64.b64encode(cv2.imencode(fmt, cv2_img)[1].tobytes()).decode()

        @staticmethod
        def base64_to_cv2(b64: str) -> np.ndarray:
            img = cv2.imdecode(
                np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR
            )
            if img is None:
                raise ValueError("Decoded image is None")
            return img

        @staticmethod
        def data_url_to_cv2(url: str) -> np.ndarray:
            return Img.Convert.base64_to_cv2(url.split(";base64,", 1)[1])

        @staticmethod
        def bytes_to_base64(data: bytes) -> str:
            return base64.b64encode(data).decode()

        @staticmethod
        def base64_to_bytes(b64: str) -> bytes:
            return base64.b64decode(b64)

        @staticmethod
        def base64_to_io_reader(b64: str) -> io.BufferedReader:
            return io.BufferedReader(io.BytesIO(base64.b64decode(b64)))

        @staticmethod
        def cv2_to_bytes(cv2_img: np.ndarray, fmt: str = ".jpg") -> bytes:
            return cv2.imencode(fmt, cv2_img)[1].tobytes()

        @staticmethod
        def bytes_to_cv2(data: bytes) -> np.ndarray:
            return np.frombuffer(data, dtype=np.uint8)

        @staticmethod
        def web_source_to_base64(src: str, file_type: str = "JPEG") -> str | None:
            import requests

            if src.startswith(("http://", "https://")):
                resp = requests.get(src, timeout=30)
                if resp.status_code != 200:
                    return None
                data = io.BytesIO(resp.content)
            else:
                data = src
            ft = file_type.lstrip(".").lower()
            if ft == "jpg":
                ft = "jpeg"
            try:
                with Image.open(data) as img:
                    if img.mode in ("RGBA", "LA") or (
                        img.mode == "P" and "transparency" in img.info
                    ):
                        img = img.convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format=ft.upper())
                    return f"data:image/{ft};base64,{base64.b64encode(buf.getvalue()).decode()}"
            except Exception:
                return None

        @staticmethod
        def smaller(src: Path, dst: Path, file_type: str = "JPEG") -> None:
            img = Img.load(src, 0)
            h, w = img.shape[:2]
            if w > 1920:
                img = cv2.resize(
                    img, (1920, int(h / (w / 1920))), interpolation=cv2.INTER_AREA
                )
            elif h > 1080:
                img = cv2.resize(
                    img, (int(w / (h / 1080)), 1080), interpolation=cv2.INTER_AREA
                )
            cv2.imwrite(str(dst), img)

    class Process:
        """Image processing filters."""

        @staticmethod
        def equalize(img: np.ndarray) -> np.ndarray:
            return cv2.equalizeHist(img)

        @staticmethod
        def red_filter(img: np.ndarray) -> np.ndarray:
            out = np.zeros_like(img)
            out[:, :, 2] = img[:, :, 2]
            return out

        @staticmethod
        def add_border(img: np.ndarray, border: int = 1) -> np.ndarray:
            return cv2.copyMakeBorder(
                img,
                border,
                border,
                border,
                border,
                borderType=cv2.BORDER_CONSTANT,
                value=114,
            )

        @staticmethod
        def canny(img: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
            return cv2.Canny(img, low, high)

        @staticmethod
        def otsu_threshold(img: np.ndarray) -> tuple[float, np.ndarray]:
            t, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            return t, binary
