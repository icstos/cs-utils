import datetime
import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
except (ImportError, OverflowError):  # pragma: no cover - 可选依赖 / 平台差异
    np = None


class EnhancedJSONEncoder(json.JSONEncoder):
    """支持若干额外 Python 类型的 JSON 编码器。

    - datetime.datetime -> 格式化字符串
    - datetime.date -> ISO 日期
    - bytes -> UTF-8 字符串
    - numpy 标量 / 数组 -> Python 原生类型（需已安装 numpy）
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime.datetime):
            # 使用可读的时间格式
            return obj.strftime("%Y-%m-%d %H:%M:%S")

        if isinstance(obj, datetime.date):
            return obj.isoformat()

        if isinstance(obj, bytes):
            return obj.decode("utf-8")

        if np is not None:
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()

        return super().default(obj)


class Json:
    """JSON 文件读写与字符串序列化工具。

    方法名尽量与标准库保持一致（`load`、`dump`、`loads`、`dumps`），
    同时补充一些小便利（支持 Path、自动创建上级目录）。
    """

    @staticmethod
    def load(file: str | Path) -> Any:
        path = Path(file)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def loads(s: str) -> Any:
        return json.loads(s)

    @staticmethod
    def save(
        data: Any, file: str | Path, *, ensure_ascii: bool = False, indent: int = 4
    ) -> None:
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=indent,
                ensure_ascii=ensure_ascii,
                cls=EnhancedJSONEncoder,
            )

    @staticmethod
    def dumps(obj: Any, *, ensure_ascii: bool = False, indent: int = 4) -> str:
        # 支持的原生类型映射（Python -> JSON）：
        # int, float: number
        # True: true
        # False: false
        # None: null
        # str: string
        # list, tuple: array
        # dict: object

        return json.dumps(
            obj, cls=EnhancedJSONEncoder, indent=indent, ensure_ascii=ensure_ascii
        )

    # 向后兼容的别名
    dump = save
    loads = loads
    to_str = dumps
