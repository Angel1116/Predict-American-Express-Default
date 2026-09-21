"""Lineage is only worth keeping if it would actually catch a mismatch.

The failure it exists for is subtle: someone re-runs the splitter with a
different seed, every filename stays the same, every shape stays the same,
and every score computed afterwards is quietly meaningless. These tests
check that the chain is recorded and that mixing two chains raises.
"""

import json
from pathlib import Path

import pytest

from config import CATEGORIES_PATH, DATA_DIR
from lineage import (digest_of, fingerprint, get, make_split_id,
                     require_same_split, split_id_of)

SPLITS = ["train", "validation", "test"]
SPLIT_ARTIFACTS = ([f"{s}_labels.parquet" for s in SPLITS]
                   + [f"{s}_data.parquet" for s in SPLITS])


def require(path):
    """Skip unless the artifact has been built; it is not in git."""
    path = Path(path)
    if not path.exists():
        pytest.skip(f"{path.name} not built yet -- run split_data.py, "
                    f"then `python code/pipeline.py train`")
    return path


def test_fingerprint_separates_different_content(tmp_path):
    a, b, c = (tmp_path / n for n in ("a.bin", "b.bin", "c.bin"))
    a.write_bytes(b"hello world")
    b.write_bytes(b"hello world")
    c.write_bytes(b"hello worlds")

    assert fingerprint(a)["digest"] == fingerprint(b)["digest"]
    assert fingerprint(a)["digest"] != fingerprint(c)["digest"]
    assert fingerprint(a)["digest_kind"] == "sha256"


def test_digest_ignores_key_order():
    assert digest_of({"a": 1, "b": 2}) == digest_of({"b": 2, "a": 1})
    assert digest_of({"a": 1}) != digest_of({"a": 2})


def test_split_id_changes_with_the_seed(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(b"customer_ID,target\nx,0\n")
    ratios = {"train": 0.7, "validation": 0.2, "test": 0.1}

    assert make_split_id(source, 42, ratios) == make_split_id(source, 42, ratios)
    assert make_split_id(source, 42, ratios) != make_split_id(source, 43, ratios)
    assert make_split_id(source, 42, ratios) != make_split_id(
        source, 42, {"train": 0.8, "validation": 0.1, "test": 0.1})


@pytest.mark.parametrize("artifact", SPLIT_ARTIFACTS)
def test_every_split_artifact_has_a_recorded_origin(artifact):
    require(DATA_DIR / artifact)
    entry = get(artifact)
    assert entry is not None, f"{artifact} has no lineage entry"
    assert entry["split_id"] and entry["seed"] is not None
    assert entry["source"]["digest"]


def test_all_split_artifacts_share_one_split_id():
    for artifact in SPLIT_ARTIFACTS:
        require(DATA_DIR / artifact)
    assert len({split_id_of(a) for a in SPLIT_ARTIFACTS}) == 1


def test_preprocessed_files_inherit_their_source_split():
    for split in SPLITS:
        source = f"{split}_data.parquet"
        derived = f"{split}_data_preprocessed.parquet"
        require(DATA_DIR / derived)
        assert split_id_of(derived) == split_id_of(source), derived


def test_categories_records_the_file_it_was_scanned_from():
    require(CATEGORIES_PATH)
    entry = get(CATEGORIES_PATH)
    assert entry["source"]["name"] == "train_data.parquet", (
        "the vocabulary must come from train, never from a held-out split")


def test_require_same_split_rejects_a_mismatch(monkeypatch):
    import lineage
    monkeypatch.setattr(lineage, "load", lambda: {
        "one.parquet": {"split_id": "aaaaaaaaaaaa"},
        "two.parquet": {"split_id": "bbbbbbbbbbbb"},
    })
    with pytest.raises(ValueError, match="different splits"):
        require_same_split("one.parquet", "two.parquet")


def test_require_same_split_accepts_a_match(monkeypatch):
    import lineage
    monkeypatch.setattr(lineage, "load", lambda: {
        "one.parquet": {"split_id": "aaaaaaaaaaaa"},
        "two.parquet": {"split_id": "aaaaaaaaaaaa"},
    })
    assert require_same_split("one.parquet", "two.parquet") == "aaaaaaaaaaaa"


def test_predict_refuses_a_model_from_another_split(monkeypatch, tmp_path):
    """The guard has to fire in predict too, not just at training time."""
    import lineage
    import predict as predict_module
    from registry import resolve_model

    features = require(DATA_DIR / "test_data_preprocessed.parquet")
    try:
        model_path = resolve_model()
    except FileNotFoundError:
        pytest.skip("no run recorded yet -- train a model first")

    monkeypatch.setattr(lineage, "load",
                        lambda: {features.name: {"split_id": "ffffffffffff"}})
    monkeypatch.setattr(predict_module, "load_meta",
                        lambda _: {"feature_version": __import__("config").FEATURE_VERSION,
                                   "split_id": "000000000000"})

    with pytest.raises(ValueError, match="different cuts"):
        predict_module.predict(features, model_path, None, None)


def test_lineage_file_is_valid_json():
    from lineage import LINEAGE_PATH
    require(LINEAGE_PATH)
    json.loads(LINEAGE_PATH.read_text())
