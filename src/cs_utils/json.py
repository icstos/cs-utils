import datetime
import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None


class EnhancedJSONEncoder(json.JSONEncoder):
    """A JSON encoder that supports a few extra Python types.

    - datetime.datetime -> formatted string
    - datetime.date -> ISO date
    - bytes -> UTF-8 string
    - numpy scalars/arrays -> native Python types (if numpy is available)
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime.datetime):
            # Use a readable timestamp format
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
    """Helper for JSON file I/O and string serialization.

    Methods mirror the stdlib API where convenient (`load`, `dump`, `loads`, `dumps`),
    while adding small conveniences (Path support, parent directory creation).
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
        # 支持的原生类型：
        # Python -> JSON
        # int, float: number
        # True: true
        # False: false
        # None: null
        # str: string
        # list,tuple: array
        # dict: object

        return json.dumps(
            obj, cls=EnhancedJSONEncoder, indent=indent, ensure_ascii=ensure_ascii
        )

    # backward-compatible aliases
    dump = save
    loads = loads
    to_str = dumps
