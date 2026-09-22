"""Whatever the container needs at boot has to actually be in the repo.

app.py resolves the model and loads the category vocabulary at import time,
so a deploy missing either one does not fail on the first request -- it
fails to start at all, and the build logs say nothing about why.

Both live under paths that are gitignored on purpose, with narrow exceptions
for the run being shipped. That leaves one easy way to break production:
retrain, commit the new latest.json, and ship a pointer to a run that never
left the laptop. These tests are the tripwire on that.
"""

import json
import subprocess
from pathlib import Path

import pytest

from config import CATEGORIES_PATH, MODEL_DIR, ROOT


def tracked(path):
    """Is this path committed, rather than merely present on this machine?"""
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(Path(path).resolve())],
        cwd=ROOT, capture_output=True, text=True)
    return result.returncode == 0


@pytest.fixture(scope="module", autouse=True)
def needs_git():
    try:
        inside = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is not available")
    if inside.returncode != 0:
        pytest.skip("not a git repository")


def test_category_vocabulary_ships():
    if not CATEGORIES_PATH.exists():
        pytest.skip("categories.json not built yet -- train first")
    assert tracked(CATEGORIES_PATH), (
        f"{CATEGORIES_PATH.name} is not committed; app.py reads it at import "
        f"and a deploy without it cannot start")


def test_latest_pointer_ships():
    latest = MODEL_DIR / "latest.json"
    if not latest.exists():
        pytest.skip("no model trained yet")
    assert tracked(latest), "models/latest.json is not committed"


def test_the_run_latest_points_at_ships():
    """The trap: a new run id in latest.json, with the run itself gitignored."""
    latest = MODEL_DIR / "latest.json"
    if not latest.exists():
        pytest.skip("no model trained yet")

    run_id = json.loads(latest.read_text())["run_id"]
    for name in ("model.txt", "meta.json"):
        path = MODEL_DIR / run_id / name
        assert path.exists(), f"run {run_id} is missing {name} locally"
        assert tracked(path), (
            f"latest.json points at run {run_id}, but {name} is not committed "
            f"-- add an exception for models/{run_id}/ to .gitignore, or point "
            f"latest.json back at a run that ships")
