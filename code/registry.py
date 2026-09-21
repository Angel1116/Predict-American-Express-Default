"""Keep every trained model instead of overwriting one file.

    models/
      runs.jsonl          one line per run, newest last
      latest.json         which run predict.py reaches for by default
      <run_id>/
        model.txt         the booster
        meta.json         everything needed to reproduce and to score with it

A fixed filename made the question "what produced this 0.7939?" unanswerable
the moment a second run happened. A run id per fit makes the history append
only, and runs.jsonl is one grep away from a comparison table.

This is deliberately a directory and two json files rather than MLflow: it
needs no server, survives being copied around, and diffs in git. Swap it for
a real tracking server when runs start outliving the laptop.
"""

import json
import time
from pathlib import Path

from config import MODEL_DIR

RUNS_LOG = MODEL_DIR / "runs.jsonl"
LATEST = MODEL_DIR / "latest.json"

# fields lifted out of meta.json into the one-line run log
SUMMARY_FIELDS = ["split_id", "feature_version", "best_iteration",
                  "trained_on", "validated_on"]


def new_run_id(seed):
    return f"{time.strftime('%Y%m%d-%H%M%S')}-s{seed}"


def run_dir(run_id):
    return MODEL_DIR / run_id


def save_run(run_id, model, meta):
    """Write the booster and its meta, then append to the log."""
    directory = run_dir(run_id)
    directory.mkdir(parents=True, exist_ok=True)

    model_path = directory / "model.txt"
    model.save_model(str(model_path), num_iteration=model.best_iteration)
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    summary = {"run_id": run_id,
               "score": meta.get("valid_amex", {}).get("score"),
               **{field: meta.get(field) for field in SUMMARY_FIELDS},
               "git": meta.get("git")}
    with RUNS_LOG.open("a", encoding="utf-8") as log:
        log.write(json.dumps(summary) + "\n")
    LATEST.write_text(json.dumps({"run_id": run_id}, indent=2))

    return model_path


def list_runs():
    """Every run recorded so far, oldest first."""
    if not RUNS_LOG.exists():
        return []
    return [json.loads(line) for line in
            RUNS_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_run_id():
    if not LATEST.exists():
        return None
    return json.loads(LATEST.read_text()).get("run_id")


def resolve_model(run_id=None, model_path=None):
    """Find the booster to score with: an explicit path, a run id, or latest."""
    if model_path:
        return Path(model_path)

    if run_id is None:
        run_id = latest_run_id()
        if run_id is None:
            raise FileNotFoundError(
                f"no runs recorded in {RUNS_LOG.name} -- train a model first, "
                f"or point --model at a booster file")

    path = run_dir(run_id) / "model.txt"
    if not path.exists():
        raise FileNotFoundError(f"run {run_id} has no model at {path}")
    return path


def load_meta(model_path):
    """The meta beside a booster, whether it is a run dir or a loose file."""
    for candidate in (Path(model_path).with_name("meta.json"),
                      Path(model_path).with_suffix(".json")):
        if candidate.exists():
            return json.loads(candidate.read_text())
    return None
