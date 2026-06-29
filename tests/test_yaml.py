from pathlib import Path
import sys
from textwrap import dedent

import pytest

# Ensure src is importable when tests run from repo root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from cs.yaml import Yaml


def test_save_and_load_roundtrip(tmp_path: Path):
    data = {"name": "Bob", "age": 30}
    p = tmp_path / "person.yaml"

    Yaml.save(data, p)
    loaded = Yaml.load(p)

    assert loaded == data


def test_load_non_yaml_raises(tmp_path: Path):
    p = tmp_path / "not_yaml.txt"
    p.write_text("key: value")

    with pytest.raises(ValueError):
        Yaml.load(p)


def test_append_filename_flag(tmp_path: Path):
    p = tmp_path / "cfg.yaml"
    p.write_text("a: 1\n")

    loaded = Yaml.load(p, append_filename=True)
    assert loaded["yaml_file"] == str(p)


def test_sanitizer_removes_nonprintable(tmp_path: Path):
    # Insert some non-printable bytes into a YAML file but keep valid YAML content
    raw = "a: 1\n"
    noisy = "\x00\x01" + raw + "\x02"
    p = tmp_path / "noisy.yaml"
    p.write_bytes(noisy.encode("utf-8", errors="ignore"))

    loaded = Yaml.load(p)
    assert loaded == {"a": 1}
