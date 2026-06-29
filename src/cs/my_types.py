"""Enums & data classes.

Optimised for Python ≥3.11 — uses ``StrEnum``, ``@unique`` on every enum,
``@dataclass(eq=True)`` with explicit ``__hash__`` for hashable value objects.
"""

from dataclasses import dataclass
from enum import Enum, StrEnum, unique


class AIMode(StrEnum):
    """AI model backend identifier (name = value, usable as string)."""

    ASSISTANT = "智能管家"
    TEACHER = "师长模式"
    GIRL_FRIEND = "女友模式"


@unique
class PlatformType(Enum):
    """Operating system / platform identifiers."""

    WINDOWS = "win"
    LINUX = "linux"
    ANDROID = "android"
    DARWIN = "darwin"
    OTHER = "other"


@unique
class Response(Enum):
    """HTTP-semantic response codes."""

    SUCCESS = 200
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    PAYLOAD_TOO_LARGE = 413
    UNKNOWN = -1


@unique
class Layout(Enum):
    """Layout orientation."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


@unique
class AnnotatorMode(Enum):
    """Annotator interaction modes."""

    POINT = 1  # point annotation
    BOX = 2  # axis-aligned bounding box
    POLYGON = 3  # polygon annotation
    ROTATE_BOX = 4  # rotated bounding box
    BRUSH = 5  # brush / freehand mask
    AI_POINT = 7  # AI-assisted point
    FREEDRAW = 9  # free-draw mode
    EDIT = 10  # edit existing annotation


@unique
class UserType(Enum):
    """User / account roles."""

    ROOT = 1  # system administrator
    LEADER = 2  # team leader
    PROJECT_MANAGER = 3  # project manager
    PRODUCT_MANAGER = 4  # product manager
    DEVELOPER = 5  # developer / engineer
    CUSTOMER = 6  # end customer
    VISITOR = 7  # read-only visitor


@unique
class Status(Enum):
    """Record activation status."""

    ACTIVE = 1  # in use
    DEPRECATED = 2  # deprecated — avoid further usage


@unique
class TaskType(Enum):
    """Computer vision task categories."""

    CLASSIFY = 1  # image classification
    DETECT = 2  # object detection
    SEGMENT = 3  # image segmentation
    POSE = 4  # pose estimation
    OBB = 5  # oriented bounding-box detection
    OCR = 6  # optical character recognition
    SAM = 7


@unique
class ModeType(Enum):
    """Model pipeline execution stages."""

    TRAIN = 1  # training
    VAL = 2  # validation
    PREDICT = 3  # inference / prediction
    EXPORT = 4  # model export
    TRACK = 5  # object tracking
    BENCHMARK = 6  # performance benchmarking


@unique
class ImageStatistic(Enum):
    """Per-image / per-region grayscale statistics."""

    MAX = 1  # maximum grayscale value
    MIN = 2  # minimum grayscale value
    AVG = 3  # average grayscale value
    DIFF = 4  # max grayscale difference (max − min)


@unique
class Extension(Enum):
    """Common file extensions."""

    XLSX = ".xlsx"
    CSV = ".csv"
    JSON = ".json"
    PDF = ".pdf"
    DOCX = ".docx"


@unique
class DetectionStrategy(Enum):
    """Inference strategy for object detectors."""

    ACCURACY = 1
    SPEED = 2


@dataclass(eq=True)
class ImgSize:
    """Immutable image size — hashed by (width, height)."""

    width: int = -1
    height: int = -1

    def __hash__(self) -> int:
        return hash((self.width, self.height))

    def __repr__(self) -> str:
        return f"ImgSize(width={self.width}, height={self.height})"
