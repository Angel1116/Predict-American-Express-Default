"""The split has to hold three properties that nothing downstream re-checks.

If any of them breaks, training still runs to completion and still reports a
score -- just a score that means nothing. That is exactly the kind of bug
worth spending a test on.
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from conftest import SPLITS, require
from config import DATA_DIR, ID_COL, TARGET_COL


def _labels(split):
    return pd.read_parquet(require(DATA_DIR / f"{split}_labels.parquet"))


def test_label_splits_are_disjoint():
    """A customer in two splits would leak training data into the score."""
    ids = {s: set(_labels(s)[ID_COL]) for s in SPLITS}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not (ids[a] & ids[b]), f"{len(ids[a] & ids[b])} customers in both {a} and {b}"


def test_label_splits_cover_every_customer_exactly_once():
    total = sum(len(_labels(s)) for s in SPLITS)
    union = set().union(*(set(_labels(s)[ID_COL]) for s in SPLITS))
    assert total == len(union), "a customer_ID appears twice across the splits"


def test_label_splits_are_stratified():
    """Within 0.5pp of each other; a drifting base rate makes splits incomparable."""
    rates = {s: _labels(s)[TARGET_COL].astype("float64").mean() for s in SPLITS}
    assert max(rates.values()) - min(rates.values()) < 0.005, rates


def test_split_ratio_is_seven_two_one():
    sizes = {s: len(_labels(s)) for s in SPLITS}
    total = sum(sizes.values())
    for split, expected in [("train", 0.7), ("validation", 0.2), ("test", 0.1)]:
        assert sizes[split] / total == pytest.approx(expected, abs=0.005)


@pytest.mark.parametrize("split", SPLITS)
def test_data_split_matches_its_labels(split):
    data = pq.read_table(require(DATA_DIR / f"{split}_data.parquet"), columns=[ID_COL])
    data_ids = set(data[ID_COL].to_pylist())
    assert data_ids == set(_labels(split)[ID_COL])


@pytest.mark.parametrize("split", SPLITS)
def test_customer_rows_are_contiguous(split):
    """`last` only means the latest statement while a customer's rows sit together.

    preprocess raises if this breaks, but by then the file has been written;
    catching it here keeps a resorted parquet from reaching the pipeline.
    """
    col = pq.read_table(require(DATA_DIR / f"{split}_data.parquet"),
                        columns=[ID_COL])[ID_COL].combine_chunks()
    codes = col.dictionary_encode().indices.to_numpy()
    blocks = int((np.diff(codes) != 0).sum()) + 1
    assert blocks == len(set(codes.tolist()))
