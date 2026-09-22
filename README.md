# AMEX Default Prediction

![Uploading image.png…]()

A LightGBM pipeline for the American Express default prediction task. The pipeline aggregates each customer's monthly statement history into a single feature vector, trains a LightGBM model, and evaluates predictions using the competition metric.

The current test-set results, evaluated on customers that were not used for training or early stopping:

```text
amex_metric: {
    'score': 0.793863,
    'gini': 0.926109,
    'top4': 0.661617,
    'auc': 0.963055
}
```

## Setup

```bash
python -m pip install -r requirements.txt
```

The package versions are pinned to keep the environment reproducible. This is especially important because LightGBM 4.x moved early stopping to callbacks, while newer versions of PyArrow added better support for float16 Parquet files.

## Data

The raw data is not included in the repository because `all_data.csv` alone is 16.4 GB.

Place the two source files in the `data/` directory:

```text
data/
├── all_data.csv      # one row per customer per monthly statement
└── all_labels.csv    # customer_ID and target
```

## Running the Pipeline

Run the following commands in order:

```bash
python code/split_data.py          # 1. Split customers into train/validation/test sets
python code/pipeline.py train      # 2. Aggregate features and train the model
python code/pipeline.py predict    # 3. Generate predictions and evaluate the test set
```

The pipeline caches intermediate results, so re-running a step will skip work that has already been completed.

Use `--force-preprocess` to rebuild the aggregated features, or `--force` when running `preprocess.py` directly.

### Outputs

| Step                  | Output                                                          |
| --------------------- | --------------------------------------------------------------- |
| `split_data.py`       | `{train,validation,test}_{data,labels}.parquet`, `lineage.json` |
| `pipeline.py train`   | `categories.json`, `*_preprocessed.parquet`, `models/<run_id>/` |
| `pipeline.py predict` | `predictions.parquet` and the evaluation metric                 |

To score a different input file:

```bash
python code/pipeline.py predict --input other_data.parquet --no-eval
```

To make predictions using an older model:

```bash
python code/pipeline.py predict --run 20260921-205141-s42
```

## Project Structure

```text
code/
├── config.py        # Paths, column definitions, and category vocabulary
├── split_data.py    # Split the CSV data into three stratified Parquet files
├── preprocess.py    # Aggregate monthly statement data into one row per customer
├── train.py         # Train LightGBM with early stopping on the validation set
├── predict.py       # Generate predictions and evaluate them when labels are available
├── pipeline.py      # Main entry point for training and prediction
├── evaluation.py    # Competition evaluation metric
└── lineage.py       # Tracks where each generated artifact came from
tests/               # Tests for key data and pipeline assumptions
```

## Design Notes

### Stateless preprocessing

The preprocessing step only uses data from each customer's own monthly records. The category vocabulary is also saved and reused, so a customer's features do not change when other customers are added or removed.

This makes it safe to preprocess the train, validation, and test splits separately.

The `test_aggregation_is_per_customer` test checks this behavior.

Any transformation that needs to be **fitted on the training data and then applied to other splits**, such as imputation or scaling, belongs in `train.py` alongside the model rather than in `preprocess.py`.

### Category vocabulary

The category vocabulary is built using the training data only.

The resulting `categories.json` is then reused for validation, test, and inference data.

If a new category appears during inference, it is mapped to the missing-value bucket and a warning is shown instead of causing the pipeline to fail.

### Training runs

Each training run gets its own run ID and directory.

A summary of each run is also stored in:

```text
models/runs.jsonl
```

By default, `predict` uses the most recent model. You can use `--run` to select an older model, which also makes it possible to roll back to a previous version.

### Data lineage

The pipeline keeps track of where each generated artifact came from.

`data/lineage.json` records the source files, random seed, and Git commit associated with each output file. It also stores a `split_id` derived from the source data and random seed.

This helps prevent accidentally mixing artifacts from different data splits.

For example, training will stop if the data and labels come from different splits, and evaluation will stop if a model is used with a holdout set that was not created from the same split.

Without these checks, files can look identical by name while actually coming from different experiments, making the resulting metrics difficult to compare.

## Tests

Run the test suite with:

```bash
python -m pytest
```

Tests that depend on generated Parquet files are skipped when those files are not available. This allows the test suite to run on a fresh clone while still letting CI distinguish between **"not built yet"** and **"broken."**
