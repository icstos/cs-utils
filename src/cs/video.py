from __future__ import annotations

import cv2
from pathlib import Path
from typing import Any, Self
from collections.abc import Iterator

VideoSource = str | Path | int
OutputPath = str | Path

DEFAULT_FRAME_FORMAT = "jpg"
DEFAULT_JPEG_QUALITY = 95


class Video:
    """Video helper for OpenCV-based common operations."""

    __slots__ = ("source", "backend", "_capture")

    def __init__(self, source: VideoSource, *, backend: int | None = None):
        self.source = Path(source) if isinstance(source, (str, Path)) else source
        self.backend = backend
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self, exc_type: type | None, exc: BaseException | None, tb: Any | None
    ) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()

    def open(self) -> cv2.VideoCapture:
        """Open the video source and validate the capture."""
        if self._capture is not None:
            self.release()

        if isinstance(self.source, Path) and not self.source.exists():
            raise FileNotFoundError(f"Video file does not exist: {self.source}")

        capture_source = (
            self.source if isinstance(self.source, int) else str(self.source)
        )
        self._capture = (
            cv2.VideoCapture(capture_source, self.backend)
            if self.backend is not None
            else cv2.VideoCapture(capture_source)
        )
        if not self._capture.isOpened():
            self.release()
            raise RuntimeError(f"Cannot open video source: {self.source}")
        return self._capture

    def release(self) -> None:
        """Release the OpenCV capture handle."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _capture_or_open(self) -> cv2.VideoCapture:
        if self._capture is None:
            return self.open()
        if not self._capture.isOpened():
            raise RuntimeError("Video capture is not opened")
        return self._capture

    def _get_property(self, prop: int) -> float:
        return self._capture_or_open().get(prop)

    @property
    def width(self) -> int:
        return int(self._get_property(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._get_property(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        return float(self._get_property(cv2.CAP_PROP_FPS)) or 0.0

    @property
    def frame_count(self) -> int:
        return int(self._get_property(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    @property
    def is_opened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def info(self) -> dict[str, float | int]:
        return {
            "source": str(self.source),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration": self.duration,
        }

    def set_position(self, index: int) -> bool:
        return self._capture_or_open().set(cv2.CAP_PROP_POS_FRAMES, float(index))

    def read_frame(self, index: int | None = None) -> Any | None:
        """Read a single frame by index or from current position."""
        capture = self._capture_or_open()
        if index is not None:
            self.set_position(index)
        success, frame = capture.read()
        return frame if success else None

    def _output_path(self, dest: OutputPath, fmt: str) -> Path:
        dest_path = Path(dest)
        return (
            dest_path.with_suffix(f".{fmt.lstrip('.')}")
            if dest_path.suffix == ""
            else dest_path
        )

    def _imwrite(self, path: Path, frame: Any, fmt: str, quality: int | None) -> None:
        params: list[int] = []
        if fmt.lower() in {"jpg", "jpeg"} and quality is not None:
            params = [cv2.IMWRITE_JPEG_QUALITY, max(0, min(100, quality))]
        if not cv2.imwrite(str(path), frame, params):
            raise RuntimeError(f"Failed to write frame to {path}")

    def frames(
        self,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> Iterator[Any]:
        """Yield frames from the video between start and stop indexes."""
        if step <= 0:
            raise ValueError("step must be a positive integer")

        capture = self._capture_or_open()
        self.set_position(start)
        index = start

        while True:
            if stop is not None and index >= stop:
                break
            success, frame = capture.read()
            if not success:
                break
            if (index - start) % step == 0:
                yield frame
            index += 1

    def save_frame(
        self,
        dest: OutputPath,
        index: int | None = None,
        fmt: str = DEFAULT_FRAME_FORMAT,
        quality: int | None = DEFAULT_JPEG_QUALITY,
    ) -> Path:
        frame = self.read_frame(index)
        if frame is None:
            raise RuntimeError("Unable to read frame from video")

        dest_path = self._output_path(dest, fmt)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        self._imwrite(dest_path, frame, fmt, quality)
        return dest_path

    def extract_frames(
        self,
        dest_dir: OutputPath,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
        prefix: str = "frame",
        fmt: str = DEFAULT_FRAME_FORMAT,
        quality: int | None = DEFAULT_JPEG_QUALITY,
        overwrite: bool = False,
    ) -> list[Path]:
        dest_path = Path(dest_dir)
        dest_path.mkdir(parents=True, exist_ok=True)

        saved_files: list[Path] = []
        capture = self._capture_or_open()
        self.set_position(start)
        index = start

        while True:
            if stop is not None and index >= stop:
                break
            success, frame = capture.read()
            if not success:
                break
            if (index - start) % step == 0:
                out_file = dest_path / f"{prefix}_{index:06d}.{fmt.lstrip('.')}"
                if out_file.exists() and not overwrite:
                    raise FileExistsError(
                        f"Destination file already exists: {out_file}"
                    )
                self._imwrite(out_file, frame, fmt, quality)
                saved_files.append(out_file)
            index += 1

        return saved_files

    def snapshot(
        self,
        dest: Source,
        time_sec: float | None = None,
        index: int | None = None,
        fmt: str = DEFAULT_FRAME_FORMAT,
        quality: int | None = DEFAULT_JPEG_QUALITY,
    ) -> Path:
        if index is None:
            if time_sec is None:
                raise ValueError("Either time_sec or index must be provided")
            if self.fps <= 0:
                raise RuntimeError("Cannot compute snapshot index from video fps")
            index = int(time_sec * self.fps)
        return self.save_frame(dest, index=index, fmt=fmt, quality=quality)


__all__ = ["Video"]
