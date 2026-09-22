"""The batch path and the serving path must agree, feature for feature.

Training reads a parquet off disk; an API request arrives as a frame in
memory. If those two ever compute a feature differently the model is scored
on inputs it was not trained on, and nothing downstream can detect it --
predictions stay plausible and are quietly wrong. That failure has a name,
training/serving skew, and this file is the guard against it.
"""

from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from config import DATA_DIR, ID_COL, load_categories
from preprocess import aggregate_frame, preprocess

SOURCE = DATA_DIR / "test_data.parquet"
N_CUSTOMERS = 30


def require(path):
    path = Path(path)
    if not path.exists():
        pytest.skip(f"{path.name} not built yet -- run split_data.py first")
    return path


@pytest.fixture
def sample(tmp_path):
    """A handful of whole customers, as both a parquet file and a frame."""
    table = pq.read_table(require(SOURCE))
    ids = pc.unique(table[ID_COL].combine_chunks()).slice(0, N_CUSTOMERS)
    subset = table.filter(pc.is_in(table[ID_COL], value_set=ids))

    path = tmp_path / "statements.parquet"
    pq.write_table(subset, path)
    return path, subset.to_pandas()


def test_serving_path_matches_the_batch_path(sample, tmp_path):
    path, frame = sample
    categories = load_categories()

    from_file = pd.read_parquet(preprocess(path, tmp_path / "out.parquet", categories))
    from_frame = aggregate_frame(frame, categories)

    pd.testing.assert_frame_equal(
        from_file.set_index(ID_COL).sort_index(),
        from_frame.set_index(ID_COL).sort_index())


def test_row_order_does_not_change_the_result(sample):
    """A caller posting months out of order must still get the right `last`."""
    _, frame = sample
    categories = load_categories()

    straight = aggregate_frame(frame, categories)
    shuffled = aggregate_frame(frame.sample(frac=1, random_state=0), categories)

    pd.testing.assert_frame_equal(
        straight.set_index(ID_COL).sort_index(),
        shuffled.set_index(ID_COL).sort_index())


def test_one_row_per_customer(sample):
    _, frame = sample
    out = aggregate_frame(frame, load_categories())
    assert len(out) == frame[ID_COL].nunique() == N_CUSTOMERS
    assert out[ID_COL].is_unique


def test_missing_column_is_reported_by_name(sample):
    _, frame = sample
    with pytest.raises(KeyError, match="B_30"):
        aggregate_frame(frame.drop(columns=["B_30"]), load_categories())


def test_missing_id_column_is_reported(sample):
    _, frame = sample
    with pytest.raises(KeyError, match=ID_COL):
        aggregate_frame(frame.drop(columns=[ID_COL]), load_categories())
