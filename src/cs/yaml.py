"""YAML 工具：面向 Python 3.14 现代化。

提供安全读写，支持 Path 与可选的文件名注解；解析前会剔除
不可打印字符，让脏文件的加载更稳健。
"""

import re
from pathlib import Path
from typing import Any

import yaml

# 清洗用正则在模块导入时编译一次，避免重复开销。
_NON_PRINTABLE_RE = re.compile(
    r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+",
    flags=re.UNICODE,
)


class Yaml:
    """简洁的 YAML 读写工具，支持 Path。

    方法：
    - `load(file, append_filename=False) -> dict[str, Any]`
    - `save(obj, file, allow_unicode=True) -> None`
    """

    @staticmethod
    def load(
        file: str | Path = Path("data.yaml"), append_filename: bool = False
    ) -> dict[str, Any]:
        path = Path(file)
        if path.suffix not in (".yaml", ".yml"):
            raise ValueError(f"Attempting to load a non-yaml file: {path}")

        text = path.read_text(encoding="utf-8", errors="ignore")

        # 解析前先剔除异常字符（不可打印字符）
        if not text.isprintable():
            text = _NON_PRINTABLE_RE.sub("", text)

        data = yaml.safe_load(text) or {}
        if append_filename:
            data["yaml_file"] = str(path)
        return data

    @staticmethod
    def save(obj: Any, file: str | Path, allow_unicode: bool = True) -> None:
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = yaml.safe_dump(obj, allow_unicode=allow_unicode)
        path.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    example = Yaml.load(Path("test.yaml"))
    print(example)
