"""Shared fixtures and helpers.

Most of these tests read the split parquets, which are deliberately not in
git -- they are rebuilt by split_data.py. Tests that need a missing
artifact skip instead of failing, so the suite still runs on a fresh clone
and CI can tell "not built yet" apart from "broken".
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

SPLITS = ["train", "validation", "test"]


def require(path):
    """Skip the test unless `path` has been built."""
    path = Path(path)
    if not path.exists():
        pytest.skip(f"{path.name} not built yet -- run split_data.py, "
                    f"then `python code/pipeline.py train`")
    return path
