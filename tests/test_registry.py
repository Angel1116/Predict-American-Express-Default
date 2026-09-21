"""Runs must accumulate rather than overwrite.

The whole point of the registry is that a second fit cannot destroy the
first, so these tests write two runs and check both are still reachable.
Everything happens in tmp_path -- a test that appended to the real
models/runs.jsonl would corrupt the record it is meant to protect.
"""

import json

import lightgbm as lgb
import numpy as np
import pytest

import registry
from registry import (latest_run_id, list_runs, load_meta, new_run_id,
                      resolve_model, save_run)


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(registry, "RUNS_LOG", tmp_path / "runs.jsonl")
    monkeypatch.setattr(registry, "LATEST", tmp_path / "latest.json")
    return tmp_path


@pytest.fixture
def tiny_model():
    rng = np.random.default_rng(0)
    x = rng.random((200, 4))
    y = (x[:, 0] > 0.5).astype(int)
    model = lgb.train({"objective": "binary", "verbosity": -1},
                      lgb.Dataset(x, label=y), num_boost_round=3)
    model.best_iteration = 3
    return model


def test_run_id_carries_the_seed():
    assert new_run_id(42).endswith("-s42")


def test_saving_two_runs_keeps_both(isolated_registry, tiny_model):
    first = save_run("run-one", tiny_model,
                     {"valid_amex": {"score": 0.70}, "split_id": "abc"})
    second = save_run("run-two", tiny_model,
                      {"valid_amex": {"score": 0.80}, "split_id": "abc"})

    assert first.exists() and second.exists()
    assert first != second

    runs = list_runs()
    assert [r["run_id"] for r in runs] == ["run-one", "run-two"]
    assert [r["score"] for r in runs] == [0.70, 0.80]


def test_latest_points_at_the_newest_run(isolated_registry, tiny_model):
    save_run("run-one", tiny_model, {"valid_amex": {"score": 0.70}})
    save_run("run-two", tiny_model, {"valid_amex": {"score": 0.80}})

    assert latest_run_id() == "run-two"
    assert resolve_model() == resolve_model("run-two")


def test_an_older_run_stays_reachable(isolated_registry, tiny_model):
    save_run("run-one", tiny_model, {"valid_amex": {"score": 0.70}, "note": "first"})
    save_run("run-two", tiny_model, {"valid_amex": {"score": 0.80}})

    assert load_meta(resolve_model("run-one"))["note"] == "first"


def test_explicit_model_path_wins(isolated_registry, tiny_model, tmp_path):
    save_run("run-one", tiny_model, {"valid_amex": {"score": 0.70}})
    elsewhere = tmp_path / "somewhere" / "other.txt"
    assert resolve_model(model_path=elsewhere) == elsewhere


def test_resolving_without_any_run_is_an_error(isolated_registry):
    with pytest.raises(FileNotFoundError, match="no runs recorded"):
        resolve_model()


def test_unknown_run_id_is_an_error(isolated_registry, tiny_model):
    save_run("run-one", tiny_model, {"valid_amex": {"score": 0.70}})
    with pytest.raises(FileNotFoundError, match="no model"):
        resolve_model("run-that-never-was")


def test_run_log_stays_one_json_object_per_line(isolated_registry, tiny_model):
    for i in range(3):
        save_run(f"run-{i}", tiny_model, {"valid_amex": {"score": 0.5 + i / 10}})
    lines = (isolated_registry / "runs.jsonl").read_text().splitlines()
    assert len(lines) == 3
    assert all(json.loads(line)["run_id"] for line in lines)
