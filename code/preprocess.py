"""Collapse statement-level rows into one row per customer.

Input rows are (customer, statement) pairs -- one customer has up to 13
monthly statements. This module reduces that history to a single feature
vector per customer_ID:

  numeric (177 cols)  -> mean, max, last, diff      = 708 cols
  categorical (11)    -> one-hot share + _last code

'diff' is last - mean: how far the final statement sits from the customer's
own average. Once a column is standardised the mean sits at 0, so this is
the last statement's distance from zero.

The one-hot share is the fraction of a customer's statements that fell in
each level, so a customer who moved between levels is not flattened down to
whatever the final month happened to say. Missing is kept as a level of its
own -- missingness in this dataset carries signal -- and no reference level
is dropped, since a neural net does not care about collinearity the way a
linear model does.

The level vocabulary is scanned from the *training* split (scan_categories)
and reused verbatim for validation and test, so no information about the
held-out splits reaches the feature definitions. Everything else here is
computed strictly within one customer, which makes the module stateless
apart from that one fitted artifact. Anything else that has to be fitted on
train and applied elsewhere (imputation, scaling) belongs in train.py,
saved alongside the model -- not here.

Numeric columns are aggregated in batches so peak memory stays near 1 GB
regardless of how wide the input is.

    python code/preprocess.py --input validation_data.parquet
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (CAT_COLS, DATA_DIR, DATE_COL, FEATURE_VERSION, ID_COL,
                    MISSING_LEVEL, NON_FEATURE_COLS, load_categories,
                    save_categories)
from lineage import digest_of, fingerprint, record, split_id_of

COL_BATCH = 40


def _label(v):
    """0.0 -> '0', -1.0 -> '-1', 'CR' -> 'CR' so dummy names stay readable."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _labels_of(series):
    """One categorical column as strings, with every flavour of missing merged."""
    labels = series.map(_label, na_action="ignore").fillna(MISSING_LEVEL)
    # pyarrow's csv reader keeps an empty string field as "" rather than null
    # (strings_can_be_null defaults to False), so D_64's missing values arrive
    # as "" and have to be folded in here.
    return labels.replace("", MISSING_LEVEL)


def _level_order(v):
    """Sort numeric-looking levels numerically, the rest alphabetically."""
    try:
        return (0, float(v), "")
    except ValueError:
        return (1, 0.0, v)


def scan_categories(path):
    """Derive the level vocabulary from one statement-level parquet."""
    df = pq.read_table(path, columns=CAT_COLS).to_pandas()
    categories = {}
    for col in CAT_COLS:
        levels = sorted(set(_labels_of(df[col]).unique()) - {MISSING_LEVEL},
                        key=_level_order)
        # always present, even when this split has no missing values: it is
        # where an unseen level lands at inference time
        categories[col] = levels + [MISSING_LEVEL]
    return categories


def ensure_categories(path, force=False):
    """Scan `path` for the vocabulary, or reuse the saved one."""
    try:
        if not force:
            categories = load_categories()
            print(f"using the saved category vocabulary "
                  f"({sum(len(v) for v in categories.values())} levels)")
            return categories
    except FileNotFoundError:
        pass

    print(f"Scanning {Path(path).name} for the category vocabulary...")
    categories = scan_categories(path)
    dst = save_categories(categories)
    record(dst, source=fingerprint(path), split_id=split_id_of(path),
           digest=digest_of(categories), produced_by="preprocess.scan_categories",
           levels=sum(len(v) for v in categories.values()))
    for col in CAT_COLS:
        print(f"  {col:6s} {len(categories[col]):>2} levels  {categories[col]}")
    print(f"  wrote {dst.name}")
    return categories


def _read_ids(path):
    """Customer ids as integer group codes, cheaper than 5M python strings."""
    col = pq.read_table(path, columns=[ID_COL])[ID_COL].combine_chunks()
    enc = col.dictionary_encode()
    codes = enc.indices.to_numpy()
    uniques = enc.dictionary.to_pandas()

    # 'last' is only the latest statement if each customer's rows sit together
    # in file order; the splitter preserves that, but a resorted file would
    # silently produce garbage here.
    blocks = int((np.diff(codes) != 0).sum()) + 1
    if blocks != len(uniques):
        raise ValueError(f"{ID_COL} rows are not contiguous ({blocks} blocks "
                         f"for {len(uniques)} customers) -- sort the file first")
    return codes, uniques


def _aggregate_numeric(read_batch, num_cols, codes, verbose=True):
    """Aggregate the numeric columns a batch at a time.

    `read_batch(cols)` hands back those columns as a frame -- off disk for a
    file, or a slice of one already in memory for a request. Batching keeps
    peak memory flat regardless of how wide the input is, and routing both
    callers through here is what stops the file path and the serving path
    from drifting into two different definitions of the same feature.
    """
    parts = []
    for i in range(0, len(num_cols), COL_BATCH):
        batch = num_cols[i:i + COL_BATCH]
        df = read_batch(batch).copy()
        df.insert(0, "_g", codes)

        agg = df.groupby("_g", sort=True)[batch].agg(["mean", "max", "last"])
        agg.columns = [f"{col}_{how}" for col, how in agg.columns]

        # built as one block and concatenated once: assigning the diffs in a
        # loop grows the frame a column at a time and leaves it fragmented
        diff = pd.DataFrame(
            agg[[f"{col}_last" for col in batch]].to_numpy()
            - agg[[f"{col}_mean" for col in batch]].to_numpy(),
            index=agg.index, columns=[f"{col}_diff" for col in batch])

        parts.append(pd.concat([agg, diff], axis="columns").astype("float32"))
        del df, agg, diff
        if verbose:
            print(f"  numeric {min(i + COL_BATCH, len(num_cols))}/{len(num_cols)} columns",
                  flush=True)
    return pd.concat(parts, axis="columns")


def _aggregate_categorical(df, codes, categories, verbose=True):
    shares, lasts = [], []

    for col in CAT_COLS:
        levels = categories[col]
        labels = _labels_of(df[col])
        cat = pd.Categorical(labels, categories=levels)

        # A level absent from the training split has nowhere of its own to go.
        # Folding it into the missing bucket keeps the schema stable and the
        # run alive; the warning is what stops that being silent.
        unknown = pd.isna(cat)
        if unknown.any():
            seen = sorted(set(labels[unknown]), key=_level_order)
            if verbose:
                print(f"  warning: {col} has {unknown.sum():,} row(s) at level(s) "
                      f"{seen[:5]} not in the training vocabulary, folded into "
                      f"'{MISSING_LEVEL}'")
            cat = pd.Categorical(labels.mask(unknown, MISSING_LEVEL),
                                 categories=levels)

        # share of the customer's statements spent in each level
        dummies = pd.get_dummies(cat, prefix=col, dtype="float32")
        dummies.insert(0, "_g", codes)
        shares.append(dummies.groupby("_g", sort=True).mean().astype("float32"))

        # the level of the final statement, as an int code into categories[col]
        last = (pd.Series(cat.codes, name=f"{col}_last")
                .groupby(codes, sort=True).last().astype("int8"))
        lasts.append(last)
        del dummies

    return pd.concat(shares + lasts, axis="columns")


def aggregate_frame(df, categories=None):
    """Collapse statement rows already in memory to one row per customer.

    The serving counterpart of preprocess(): same aggregation, same feature
    names, same order, but fed a frame rather than a parquet path. A request
    carries a handful of customers, so there is nothing to stream.

    Unlike the file path this sorts defensively instead of demanding sorted
    input. `last` means the latest statement, and a caller posting a
    customer's months in whatever order their database returned them should
    get the right answer rather than an error.
    """
    categories = categories if categories is not None else load_categories()

    missing = [c for c in [ID_COL] + CAT_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"missing required column(s): {missing}")

    by = [ID_COL, DATE_COL] if DATE_COL in df.columns else [ID_COL]
    df = df.sort_values(by, kind="stable").reset_index(drop=True)

    codes, uniques = pd.factorize(df[ID_COL])
    num_cols = [c for c in df.columns if c not in NON_FEATURE_COLS + CAT_COLS]

    num_agg = _aggregate_numeric(lambda cols: df[cols], num_cols, codes,
                                 verbose=False)
    cat_agg = _aggregate_categorical(df[CAT_COLS], codes, categories,
                                     verbose=False)

    out = pd.concat([num_agg, cat_agg], axis="columns")
    out.insert(0, ID_COL, uniques)
    return out.reset_index(drop=True)


def preprocess(src, dst=None, categories=None):
    """Collapse statement-level rows in `src` to one row per customer."""
    src = Path(src)
    dst = Path(dst) if dst else src.with_name(f"{src.stem}_preprocessed.parquet")
    categories = categories if categories is not None else load_categories()

    n_rows_in = pq.ParquetFile(src).metadata.num_rows
    print(f"Preprocessing {src.name} ({n_rows_in:,} rows)...")

    codes, uniques = _read_ids(src)
    all_cols = pq.ParquetFile(src).schema_arrow.names
    num_cols = [c for c in all_cols if c not in NON_FEATURE_COLS + CAT_COLS]
    print(f"  {len(num_cols)} numeric columns, {len(CAT_COLS)} categorical columns")

    num_agg = _aggregate_numeric(
        lambda cols: pq.read_table(src, columns=cols).to_pandas(), num_cols, codes)
    cat_agg = _aggregate_categorical(
        pq.read_table(src, columns=CAT_COLS).to_pandas(), codes, categories)

    out = pd.concat([num_agg, cat_agg], axis="columns")
    out.insert(0, ID_COL, uniques.values)
    out.reset_index(drop=True, inplace=True)

    table = pa.Table.from_pandas(out, preserve_index=False)
    pq.write_table(table, dst, compression="zstd", compression_level=9)

    # inherits the split it came from, so train.py can refuse to pair files
    # that were cut from the data in two different ways
    record(dst, source=fingerprint(src), split_id=split_id_of(src),
           categories_digest=digest_of(categories),
           feature_version=FEATURE_VERSION, produced_by="preprocess.py",
           customers=len(out), features=len(out.columns) - 1)

    print(f"  categorical _last columns: {[f'{c}_last' for c in CAT_COLS]}")
    print(f"  {n_rows_in:,} rows -> {len(out):,} rows, "
          f"{out[ID_COL].nunique():,} unique {ID_COL}")
    print(f"  {len(out.columns) - 1} features "
          f"({num_agg.shape[1]} numeric + {cat_agg.shape[1]} categorical)")
    print(f"  wrote {dst.name} ({os.path.getsize(dst) / 1e9:.2f} GB)")
    return dst


def preprocess_if_missing(src, dst=None, force=False, categories=None):
    """Reuse an existing output unless `force`, and say how old it is.

    The mtime is printed rather than swallowed because a cached file from
    before an aggregation change is otherwise indistinguishable from a fresh
    one, and silently training on stale features is very hard to notice.
    """
    src = Path(src)
    dst = Path(dst) if dst else src.with_name(f"{src.stem}_preprocessed.parquet")

    if dst.exists() and not force:
        age = time.strftime("%Y-%m-%d %H:%M", time.localtime(dst.stat().st_mtime))
        print(f"{dst.name} already exists (built {age}), skipping preprocessing "
              f"-- pass --force to rebuild")
        return dst
    return preprocess(src, dst, categories)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default="train_data.parquet",
                    help="statement-level parquet inside the data directory")
    ap.add_argument("--output", default=None,
                    help="defaults to <input>_preprocessed.parquet")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the output exists")
    ap.add_argument("--rescan-categories", action="store_true",
                    help="derive the vocabulary from --input instead of reusing "
                         "the saved one (only correct on the training split)")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.is_absolute():
        src = DATA_DIR / src
    dst = Path(args.output) if args.output else None
    if dst is not None and not dst.is_absolute():
        dst = DATA_DIR / dst

    categories = ensure_categories(src, force=args.rescan_categories)
    preprocess_if_missing(src, dst, force=args.force, categories=categories)


if __name__ == "__main__":
    main()
