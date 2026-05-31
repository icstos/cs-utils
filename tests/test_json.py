import os
import sys
from datetime import date, datetime
from pathlib import Path
import tempfile

import pytest

# Ensure src is importable when tests run from repo root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from utils.json import Json


def test_dumps_datetime_and_date_and_bytes():
    now = datetime(2020, 1, 2, 3, 4, 5)
    d = date(2020, 1, 2)
    b = b"hello"

    s = Json.dumps({"ts": now, "d": d, "b": b})

    assert "2020-01-02 03:04:05" in s
    assert "2020-01-02" in s
    assert "hello" in s


def test_save_and_load_file(tmp_path: Path):
    data = {"name": "Alice", "ts": datetime(2021, 12, 31, 23, 59, 59), "raw": b"x"}
    p = tmp_path / "data.json"

    Json.save(data, p)
    loaded = Json.load(p)

    assert loaded["name"] == "Alice"
    assert isinstance(loaded["ts"], str) and loaded["ts"].startswith("2021-12-31")
    assert loaded["raw"] == "x"


def test_numpy_support_or_skip():
    try:
        np = pytest.importorskip("numpy")
    except OverflowError:
        pytest.skip("numpy is not compatible with this Python version")

    arr = np.array([1, 2, 3])
    obj = {"arr": arr, "n": np.int64(5), "f": np.float64(3.14)}
    s = Json.dumps(obj)

    # Parse back the JSON and assert semantic equality (avoid format-dependent checks)
    parsed = Json.loads(s)
    assert parsed["arr"] == [1, 2, 3]
    assert parsed["n"] == 5
    assert parsed["f"] == 3.14
