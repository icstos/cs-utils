"""YAML utilities: modernized for Python 3.14.

Provides safe loading and saving with Path support and optional
filename annotation. Non-printable characters are removed before
parsing to make loading robust for messy files.
"""

import re
from pathlib import Path
from typing import Any

import yaml

# Compile the sanitizer regex once at module import.
_NON_PRINTABLE_RE = re.compile(
    r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+",
    flags=re.UNICODE,
)


class Yaml:
    """Simple YAML helper with Path support.

    Methods:
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

        # Remove pathological non-printable characters before parsing.
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
