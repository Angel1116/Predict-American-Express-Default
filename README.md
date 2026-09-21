# Amex default prediction

A LightGBM pipeline for the American Express default-prediction task: collapse
each customer's monthly statement history into one feature vector, train a
booster, and score held-out customers with the competition metric.

Current test-set result, on customers the model never saw during training or
early stopping:

```
amex_metric: {'score': 0.793863, 'gini': 0.926109, 'top4': 0.661617, 'auc': 0.963055}
```

## Setup

```bash
python -m pip install -r requirements.txt
```

Versions are pinned exactly. LightGBM 4.x moved early stopping out of
`train()` into a callback, and pyarrow only grew usable float16 parquet
support recently, so a floating resolve is a broken build waiting to happen.

## Data

Not in git -- `all_data.csv` alone is 16.4 GB. Put the two source files in
`data/`:

```
data/
  all_data.csv      one row per (customer, monthly statement)
  all_labels.csv    customer_ID, target
```

## Running it

```bash
python code/split_data.py                  # 1. cut 70/20/10 by customer
python code/pipeline.py train              # 2. aggregate, then fit
python code/pipeline.py predict            # 3. score and evaluate the test split
```

Each step caches its output, so re-running skips work that is already done.
Pass `--force-preprocess` to rebuild the aggregated features, or `--force` to
`preprocess.py` directly.

What those produce:

| Step | Output |
|---|---|
| `split_data.py` | `{train,validation,test}_{data,labels}.parquet`, `lineage.json` |
| `pipeline.py train` | `categories.json`, `*_preprocessed.parquet`, `models/<run_id>/` |
| `pipeline.py predict` | `predictions.parquet`, and the metric on stdout |

To score a different file, or with an older model:

```bash
python code/pipeline.py predict --input other_data.parquet --no-eval
python code/pipeline.py predict --run 20260921-205141-s42
```

## Layout

```
code/
  config.py       paths, column roles, the derived category vocabulary
  split_data.py   csv -> three stratified parquet splits, float16
  preprocess.py   statement rows -> one row per customer (708 + 67 features)
  train.py        LightGBM, early stopping on the validation split
  predict.py      score a preprocessed table, evaluate if labels exist
  pipeline.py     entry point: `train` or `predict`
  evaluation.py   the competition metric
  lineage.py      what every artifact was built from
  registry.py     one directory per training run
tests/            invariants that fail silently if they break
```

## Design notes

**Preprocessing is stateless.** Every feature is computed inside one
customer's own rows, and the category vocabulary is frozen on disk, so a
customer's features do not change when other customers are added or removed.
That is what makes it safe to aggregate the splits independently, and
`test_aggregation_is_per_customer` is the guard on it. Anything that has to be
fitted on train and applied elsewhere -- imputation, scaling -- belongs in
`train.py` next to the model, not in `preprocess.py`.

**The vocabulary comes from train alone.** `categories.json` is scanned from
the training split, never from a held-out one, and reused verbatim everywhere
else. A level that only appears at inference falls into the missing bucket
with a warning rather than crashing.

**Runs accumulate.** Each fit gets a run id and its own directory, plus a
summary line in `models/runs.jsonl`. `predict` uses the newest by default and
`--run` reaches an older one, which is also how a rollback works.

**Artifacts know where they came from.** `data/lineage.json` records the
inputs, seed and commit behind every produced file, and a `split_id` derived
from the source content plus the seed. Training refuses to pair files from
different splits, and scoring refuses a model against a holdout it was not cut
with -- the failure that would otherwise leave every filename identical and
every number quietly incomparable.

## Tests

```bash
python -m pytest
```

Tests that need the built parquets skip when those are absent, so the suite
runs on a fresh clone and CI can tell "not built yet" apart from "broken".
