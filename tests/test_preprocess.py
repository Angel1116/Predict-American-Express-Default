"""Preprocessing has one property the whole design rests on: it is per-customer.

If a feature ever starts depending on other rows in the file, preprocessing
train and validation together silently leaks -- and nothing downstream can
detect it. test_aggregation_is_per_customer is the guard on that.
"""

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from conftest import SPLITS, require
from config import CAT_COLS, DATA_DIR, ID_COL, MISSING_LEVEL, load_categories
from preprocess import (_aggregate_categorical, _labels_of, _level_order,
                        preprocess, scan_categories)


def test_labels_of_merges_every_flavour_of_missing():
    """Floats, empty strings and nulls all have to land on the same level."""
    s = pd.Series([0.0, -1.0, 2.0, "CR", "", None, float("nan")], dtype=object)
    assert list(_labels_of(s)) == ["0", "-1", "2", "CR",
                                   MISSING_LEVEL, MISSING_LEVEL, MISSING_LEVEL]


def test_numeric_levels_sort_numerically():
    """Lexicographic order would put '10' before '2' and scramble the codes."""
    assert sorted(["10", "2", "-1", "CL"], key=_level_order) == ["-1", "2", "10", "CL"]


def test_scanned_vocabulary_always_offers_a_missing_level():
    """Without it an unseen value at inference has nowhere to go."""
    categories = scan_categories(require(DATA_DIR / "train_data.parquet"))
    for col in CAT_COLS:
        assert categories[col][-1] == MISSING_LEVEL, col


def test_unknown_level_folds_into_missing_and_warns(tmp_path, capsys):
    """Production will meet a level train never saw; it must not crash."""
    categories = {col: ["0", "1", MISSING_LEVEL] for col in CAT_COLS}
    frame = {ID_COL: ["a", "a", "a", "b", "b", "b"]}
    for col in CAT_COLS:
        frame[col] = ["0", "1", "0", "1", "0", "1"]
    frame[CAT_COLS[0]] = ["0", "1", "SURPRISE", "1", "0", "1"]

    src = tmp_path / "statements.parquet"
    pd.DataFrame(frame).to_parquet(src, index=False)

    out = _aggregate_categorical(src, np.array([0, 0, 0, 1, 1, 1]), categories)

    assert "not in the training vocabulary" in capsys.readouterr().out
    # customer 'a' spent one of three statements at the unknown level
    assert out[f"{CAT_COLS[0]}_{MISSING_LEVEL}"].iloc[0] == pytest.approx(1 / 3)
    assert out[f"{CAT_COLS[0]}_{MISSING_LEVEL}"].iloc[1] == pytest.approx(0.0)


def test_aggregation_is_per_customer(tmp_path):
    """Dropping other customers must not move a customer's own features.

    This is what makes it safe to preprocess train and validation in one pass,
    and what would break the moment a cross-customer statistic crept in.
    """
    table = pq.read_table(require(DATA_DIR / "test_data.parquet"))
    ids = pc.unique(table[ID_COL].combine_chunks())
    categories = load_categories()

    outputs = []
    for n in (20, 8):
        subset = table.filter(pc.is_in(table[ID_COL], value_set=ids.slice(0, n)))
        src = tmp_path / f"src_{n}.parquet"
        pq.write_table(subset, src)
        outputs.append(pd.read_parquet(
            preprocess(src, tmp_path / f"out_{n}.parquet", categories)))

    many, few = (df.set_index(ID_COL).sort_index() for df in outputs)
    pd.testing.assert_frame_equal(many.loc[few.index], few)


@pytest.mark.parametrize("split", SPLITS)
def test_preprocessed_splits_share_one_column_layout(split):
    """LightGBM matches features by position, so order is part of the contract."""
    reference = pq.ParquetFile(
        require(DATA_DIR / "train_data_preprocessed.parquet")).schema_arrow.names
    names = pq.ParquetFile(
        require(DATA_DIR / f"{split}_data_preprocessed.parquet")).schema_arrow.names
    assert names == reference


def test_preprocessing_yields_one_row_per_customer():
    for split in SPLITS:
        features = pd.read_parquet(
            require(DATA_DIR / f"{split}_data_preprocessed.parquet"), columns=[ID_COL])
        labels = pd.read_parquet(require(DATA_DIR / f"{split}_labels.parquet"),
                                 columns=[ID_COL])
        assert len(features) == features[ID_COL].nunique() == len(labels)
