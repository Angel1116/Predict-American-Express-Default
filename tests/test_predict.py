"""Scoring fails loudly or not at all.

Booster.predict lines features up by position, so a frame with the right
columns in the wrong order scores perfectly happily and returns nonsense.
Both guards against that live in predict(), and both are tested here.
"""

import json

import pandas as pd
import pytest

from conftest import require
from config import FEATURE_VERSION, ID_COL, MODEL_DIR, SEED
from predict import predict

MODEL = MODEL_DIR / f"lgbm_seed{SEED}.txt"
FEATURES = "test_data_preprocessed.parquet"


def _meta():
    return json.loads(require(MODEL.with_suffix(".json")).read_text())


def test_saved_model_matches_the_current_feature_version():
    """A model trained on older features must not be scored with newer ones."""
    assert _meta()["feature_version"] == FEATURE_VERSION


def test_meta_records_the_feature_order():
    meta = _meta()
    assert meta["feature_name"], "feature order is what predict() reindexes by"
    assert all(col in meta["feature_name"] for col in meta["cat_features"])


def test_predict_rejects_a_frame_with_a_feature_missing(tmp_path):
    from config import DATA_DIR
    df = pd.read_parquet(require(DATA_DIR / FEATURES)).head(50)
    dropped = next(c for c in df.columns if c.endswith("_diff"))

    src = tmp_path / "incomplete.parquet"
    df.drop(columns=[dropped]).to_parquet(src, index=False)

    with pytest.raises(KeyError, match="missing"):
        predict(src, require(MODEL), None, None)


def test_predict_is_order_independent(tmp_path):
    """Shuffling the columns must not change a single prediction."""
    from config import DATA_DIR
    df = pd.read_parquet(require(DATA_DIR / FEATURES)).head(200)

    straight = tmp_path / "straight.parquet"
    df.to_parquet(straight, index=False)

    shuffled_cols = [ID_COL] + list(reversed([c for c in df.columns if c != ID_COL]))
    shuffled = tmp_path / "shuffled.parquet"
    df[shuffled_cols].to_parquet(shuffled, index=False)

    a = predict(straight, require(MODEL), None, None)
    b = predict(shuffled, require(MODEL), None, None)
    pd.testing.assert_frame_equal(a, b)
