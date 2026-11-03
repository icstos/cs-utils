"""枚举与数据类。

面向 Python ≥3.11 优化：使用 ``StrEnum``，每个枚举都加 ``@unique``，
可哈希值对象用 ``@dataclass(eq=True)`` 并显式实现 ``__hash__``。
"""

from dataclasses import dataclass
from enum import Enum, StrEnum, unique


class AIMode(StrEnum):
    """AI 模型后端标识（名称即取值，可直接当字符串用）。"""

    ASSISTANT = "智能管家"
    TEACHER = "师长模式"
    GIRL_FRIEND = "女友模式"


@unique
class PlatformType(Enum):
    """操作系统 / 平台标识。"""

    WINDOWS = "win"
    LINUX = "linux"
    ANDROID = "android"
    DARWIN = "darwin"
    OTHER = "other"


@unique
class Response(Enum):
    """HTTP 语义的响应码。"""

    SUCCESS = 200
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    PAYLOAD_TOO_LARGE = 413
    UNKNOWN = -1


@unique
class Layout(Enum):
    """布局方向。"""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


@unique
class AnnotatorMode(Enum):
    """标注器交互模式。"""

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
    """用户 / 账号角色。"""

    ROOT = 1  # system administrator
    LEADER = 2  # team leader
    PROJECT_MANAGER = 3  # project manager
    PRODUCT_MANAGER = 4  # product manager
    DEVELOPER = 5  # developer / engineer
    CUSTOMER = 6  # end customer
    VISITOR = 7  # read-only visitor


@unique
class Status(Enum):
    """记录启用状态。"""

    ACTIVE = 1  # 正在使用
    DEPRECATED = 2  # 已弃用，应避免继续使用


@unique
class TaskType(Enum):
    """计算机视觉任务类型。"""

    CLASSIFY = 1  # 图像分类
    DETECT = 2  # 目标检测
    SEGMENT = 3  # 图像分割
    POSE = 4  # 姿态估计
    OBB = 5  # 旋转框检测
    OCR = 6  # 光学字符识别
    SAM = 7  # 可提示分割（SAM）


@unique
class ModeType(Enum):
    """模型流水线执行阶段。"""

    TRAIN = 1  # 训练
    VAL = 2  # 验证
    PREDICT = 3  # 推理 / 预测
    EXPORT = 4  # 模型导出
    TRACK = 5  # 目标跟踪
    BENCHMARK = 6  # 性能基准测试


@unique
class ImageStatistic(Enum):
    """单张图像 / 单个区域的灰度统计。"""

    MAX = 1  # 最大灰度值
    MIN = 2  # 最小灰度值
    AVG = 3  # 平均灰度值
    DIFF = 4  # 最大灰度差（max − min）


@unique
class Extension(Enum):
    """常用文件后缀。"""

    XLSX = ".xlsx"
    CSV = ".csv"
    JSON = ".json"
    PDF = ".pdf"
    DOCX = ".docx"


@unique
class DetectionStrategy(Enum):
    """目标检测器的推理策略。"""

    ACCURACY = 1
    SPEED = 2


@dataclass(eq=True)
class ImgSize:
    """不可变图像尺寸——以 (width, height) 作为哈希键。"""

    width: int = -1
    height: int = -1

    def __hash__(self) -> int:
        return hash((self.width, self.height))

    def __repr__(self) -> str:
        return f"ImgSize(width={self.width}, height={self.height})"
