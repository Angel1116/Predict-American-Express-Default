"""Split the labels 7:2:1, then route the statement rows to match.

Two passes over two files:

  all_labels.csv -> train_labels.parquet / validation_labels.parquet
                    / test_labels.parquet     (stratified on target)

  all_data.csv   -> train_data.parquet / validation_data.parquet
                    / test_data.parquet       (routed by customer_ID)

The split is decided on the labels alone and the statement rows simply
follow their customer, so a customer's 13 monthly statements always land in
exactly one file -- splitting statements directly would put the same
customer on both sides of the validation boundary.

all_data.csv is ~16 GB, so it is streamed through pyarrow's incremental CSV
reader and written out row group by row group; peak memory stays at a couple
of hundred MB regardless of file size.

Float columns are stored as float16 with BYTE_STREAM_SPLIT + zstd. That
holds ~3 decimal digits (max error 2.4e-4 on this data, an order of
magnitude below the noise Amex added during anonymisation) and halves the
file against float32. Values above 65504 would overflow to infinity, so
every batch is checked and the affected columns reported at the end.

    python code/split_data.py
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, ID_COL, SEED, TARGET_COL
from lineage import fingerprint, make_split_id, record

SPLIT_NAMES = ["train", "validation", "test"]
TEST_FRACTION = 0.1
VALID_FRACTION = 0.2

BLOCK_SIZE = 1 << 26        # 64 MB of CSV per read block
ROWS_PER_GROUP = 250_000    # parquet row group size
REPORT_EVERY = 1_000_000
FLOAT16_MAX = 65504


def split_labels(labels_csv, outdir, seed):
    """Stratified 7:2:1 split, written as <name>_labels.parquet."""
    df = pd.read_csv(labels_csv)
    print(f"{Path(labels_csv).name}: {len(df):,} rows, "
          f"{df[TARGET_COL].mean():.4f} positive")

    # two cuts: hold out validation+test together, then divide that remainder
    # in the 2:1 ratio they have between themselves
    held_out = VALID_FRACTION + TEST_FRACTION
    train, rest = train_test_split(
        df, test_size=held_out, stratify=df[TARGET_COL], random_state=seed)
    valid, test = train_test_split(
        rest, test_size=TEST_FRACTION / held_out, stratify=rest[TARGET_COL],
        random_state=seed)

    parts = {"train": train, "validation": valid, "test": test}
    print()
    for name in SPLIT_NAMES:
        part = parts[name]
        table = pa.table({
            ID_COL: pa.array(part[ID_COL].to_numpy(), type=pa.string()),
            TARGET_COL: pa.array(part[TARGET_COL].to_numpy()).cast(
                pa.float32()).cast(pa.float16()),
        })
        dst = os.path.join(outdir, f"{name}_labels.parquet")
        pq.write_table(table, dst, compression="zstd", compression_level=9)
        print(f"  {name + '_labels.parquet':28s} {len(part):>7,} rows "
              f"({len(part) / len(df):.1%})   target {part[TARGET_COL].mean():.4f}")

    return {name: parts[name][ID_COL] for name in SPLIT_NAMES}


def probe_schema(path):
    """Infer column types from the first block, so every batch parses the same way."""
    with pacsv.open_csv(path, read_options=pacsv.ReadOptions(block_size=BLOCK_SIZE)) as reader:
        schema = reader.schema
    # a column that is entirely null in the first block would be inferred as
    # null type and then clash with later blocks; treat those as numeric
    return pa.schema([
        pa.field(f.name, pa.float64() if pa.types.is_null(f.type) else f.type)
        for f in schema
    ])


def to_float16(table, overflowed):
    """Cast float columns to half precision, recording any that overflow."""
    cols = []
    for f in table.schema:
        col = table[f.name]
        if pa.types.is_floating(f.type):
            col32 = col.cast(pa.float32(), safe=False)
            # is_inf has no half-precision kernel, so check the magnitude
            # before the cast rather than hunting for infinities after it
            peak = pc.max(pc.abs(col32)).as_py()
            if peak is not None and peak > FLOAT16_MAX:
                overflowed.add(f.name)
            col = col32.cast(pa.float16(), safe=False)
        cols.append(col)
    return pa.Table.from_arrays(cols, names=table.schema.names)


class GroupedWriter:
    """Buffers batches and flushes them as fixed-size parquet row groups."""

    def __init__(self, path, schema, compression, level, float_cols):
        self.path = path
        self.writer = pq.ParquetWriter(
            path, schema,
            compression=compression,
            compression_level=level,
            use_dictionary=[ID_COL],
            column_encoding={c: "BYTE_STREAM_SPLIT" for c in float_cols},
        )
        self.buf = []
        self.buffered = 0
        self.rows = 0
        self.ids = set()

    def add(self, table):
        if table.num_rows == 0:
            return
        self.rows += table.num_rows
        self.ids.update(pc.unique(table[ID_COL].combine_chunks()).to_pylist())
        self.buf.append(table)
        self.buffered += table.num_rows
        if self.buffered >= ROWS_PER_GROUP:
            self.flush()

    def flush(self):
        if self.buf:
            self.writer.write_table(pa.concat_tables(self.buf))
            self.buf, self.buffered = [], 0

    def close(self):
        self.flush()
        self.writer.close()


def split_data(data_csv, id_groups, outdir, compression, level):
    """Route every statement row to the file its customer was assigned to."""
    csv_schema = probe_schema(data_csv)
    out_schema = pa.schema([
        pa.field(f.name, pa.float16() if pa.types.is_floating(f.type) else f.type)
        for f in csv_schema
    ])
    float_cols = [f.name for f in out_schema if pa.types.is_floating(f.type)]
    print(f"\n{Path(data_csv).name}: {len(csv_schema.names)} columns, "
          f"{len(float_cols)} cast to float16")

    value_sets = {name: pa.array(ids.to_numpy(), type=pa.string())
                  for name, ids in id_groups.items() if name != "train"}
    writers = {
        name: GroupedWriter(os.path.join(outdir, f"{name}_data.parquet"),
                            out_schema, compression, level, float_cols)
        for name in SPLIT_NAMES
    }

    overflowed = set()
    n_rows = 0
    next_report = REPORT_EVERY
    read = pacsv.ReadOptions(block_size=BLOCK_SIZE)
    convert = pacsv.ConvertOptions(column_types=csv_schema)
    try:
        with pacsv.open_csv(data_csv, read_options=read, convert_options=convert) as reader:
            for batch in reader:
                table = to_float16(
                    pa.Table.from_batches([batch], schema=csv_schema), overflowed)

                remaining = pa.array([True] * table.num_rows)
                for name, ids in value_sets.items():
                    mask = pc.is_in(table[ID_COL], value_set=ids)
                    writers[name].add(table.filter(mask))
                    remaining = pc.and_(remaining, pc.invert(mask))
                writers["train"].add(table.filter(remaining))

                n_rows += table.num_rows
                if n_rows >= next_report:
                    print(f"  ...{n_rows:,} rows processed", flush=True)
                    next_report += REPORT_EVERY
    finally:
        for w in writers.values():
            w.close()

    print(f"\ndata rows processed: {n_rows:,}")
    for name in SPLIT_NAMES:
        w = writers[name]
        print(f"  {name + '_data.parquet':28s} {len(w.ids):>7,} unique {ID_COL}   "
              f"{w.rows:>9,} rows   {os.path.getsize(w.path) / 1e9:.2f} GB")

    for name in SPLIT_NAMES:
        expected = len(id_groups[name])
        got = len(writers[name].ids)
        if got != expected:
            print(f"warning: {name} has {got} customers but its labels list "
                  f"{expected} -- {expected - got} never appeared in the csv")
    pairs = [(a, b) for i, a in enumerate(SPLIT_NAMES) for b in SPLIT_NAMES[i + 1:]]
    for a, b in pairs:
        overlap = writers[a].ids & writers[b].ids
        if overlap:
            print(f"warning: {len(overlap)} {ID_COL} landed in both {a} and {b}")
    if overflowed:
        print(f"warning: {len(overflowed)} column(s) exceeded the float16 range "
              f"and became infinite: {sorted(overflowed)[:10]}")

    return {name: {"rows": writers[name].rows,
                   "customers": len(writers[name].ids)} for name in SPLIT_NAMES}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="all_data.csv")
    ap.add_argument("--labels", default="all_labels.csv")
    ap.add_argument("--outdir", default=str(DATA_DIR))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--compression", default="zstd")
    ap.add_argument("--compression-level", type=int, default=9)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data_csv = Path(args.data)
    labels_csv = Path(args.labels)
    if not data_csv.is_absolute():
        data_csv = DATA_DIR / data_csv
    if not labels_csv.is_absolute():
        labels_csv = DATA_DIR / labels_csv

    # names this particular cut of the data: source content + seed + ratios.
    # Every artifact below inherits it, so a later run with a different seed
    # cannot be silently mixed with these files.
    ratios = {"train": round(1 - VALID_FRACTION - TEST_FRACTION, 4),
              "validation": VALID_FRACTION, "test": TEST_FRACTION}
    split_id = make_split_id(data_csv, args.seed, ratios)
    data_source = fingerprint(data_csv)
    labels_source = fingerprint(labels_csv)
    print(f"split_id {split_id}  (seed {args.seed}, "
          f"{data_source['name']} {data_source['digest']})\n")

    id_groups = split_labels(labels_csv, args.outdir, args.seed)
    stats = split_data(data_csv, id_groups, args.outdir,
                       args.compression, args.compression_level)

    for name in SPLIT_NAMES:
        common = {"split_id": split_id, "seed": args.seed, "ratios": ratios,
                  "produced_by": "split_data.py"}
        record(f"{name}_labels.parquet", source=labels_source,
               customers=len(id_groups[name]), **common)
        record(f"{name}_data.parquet", source=data_source, **stats[name], **common)
    print(f"\nlineage written for 6 artifacts under split_id {split_id}")


if __name__ == "__main__":
    main()
