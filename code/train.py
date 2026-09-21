"""Train LightGBM on the one-row-per-customer table, scored by amex_metric.

The training and validation customers were separated back in
split_data.py, so this module trains on one file and early-stops on
another rather than carving a validation set out of the training split. That
keeps the 70/20/10 ratio the splitter produced intact instead of shrinking
train to 56% of the data.

Order matters here. The category vocabulary is scanned from the *training*
split first, then reused when the validation split is aggregated, so nothing
about the held-out customers reaches the feature definitions.

Alongside the booster this writes a small json describing the fitted
artifact: the feature order, which columns were treated as categorical, the
vocabulary, and the feature version. predict.py reads that file rather than
assuming anything about the parquet it is handed -- a model and the feature
layout it expects have to travel together.

    python code/train.py
    python code/train.py --force-preprocess
"""

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (CAT_COLS, DATA_DIR, FEATURE_VERSION, ID_COL, MODEL_DIR,
                    SEED, TARGET_COL, cat_feature_names)
from evaluation import amex_metric
from lineage import digest_of, get, git_commit, require_same_split
from preprocess import ensure_categories, preprocess_if_missing
from registry import list_runs, new_run_id, save_run

NUM_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 200
PRINT_EVERY = 50            # set to 1 to print the metric on every iteration

LGB_PARAMS = {
    "objective": "binary",
    "metric": "None",       # amex score is the only thing we steer on
    "boosting": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 100,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.20,
    "bagging_fraction": 0.50,
    "bagging_freq": 10,
    "lambda_l2": 2,
    "n_jobs": -1,
    "seed": SEED,
    "verbosity": -1,
}

# the metric dict from the most recent evaluation, so the printing callback
# can report gini/top4/auc without recomputing the whole thing
_LATEST = {}


def _amex_feval(y_pred, dataset):
    y_true = dataset.get_label().astype(int)
    m = amex_metric(pd.DataFrame({TARGET_COL: y_true}),
                    pd.DataFrame({"prediction": y_pred}))
    _LATEST.clear()
    _LATEST.update(m)
    return "amex", m["score"], True


def _print_metric(period):
    def callback(env):
        it = env.iteration + 1
        if it == 1 or it % period == 0:
            print(f"  [{it:>5}] " + str({k: round(v, 6) for k, v in _LATEST.items()}),
                  flush=True)
    callback.order = 40
    return callback


def _load_xy(features_path, labels_path, what):
    df = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)

    n_before = len(df)
    df = df.merge(labels, on=ID_COL, how="inner")
    if len(df) != n_before:
        print(f"  warning: {n_before - len(df)} {what} customers had no label "
              f"and were dropped")

    X = df.drop(columns=[ID_COL, TARGET_COL])
    # the splitter stores target as float16; LightGBM wants a plain numeric label
    y = df[TARGET_COL].astype("int8")
    print(f"  {what:10s} {len(X):>7,} customers, {X.shape[1]} features, "
          f"{y.mean():.1%} positive")
    return X, y


def train_model(train_features, valid_features, train_labels=None,
                valid_labels=None, categories=None):
    """Fit a booster, early-stopping on the validation split, and save it."""
    train_labels = Path(train_labels) if train_labels else DATA_DIR / "train_labels.parquet"
    valid_labels = Path(valid_labels) if valid_labels else DATA_DIR / "validation_labels.parquet"

    print(f"\nTraining on {Path(train_features).name}, "
          f"validating on {Path(valid_features).name}...")
    x_train, y_train = _load_xy(train_features, train_labels, "train")
    x_valid, y_valid = _load_xy(valid_features, valid_labels, "validation")

    # same column names is not enough: two files can line up perfectly and
    # still come from different cuts of the data
    split_id = require_same_split(train_features, valid_features,
                                  train_labels, valid_labels)
    print(f"  split_id: {split_id}")

    if list(x_train.columns) != list(x_valid.columns):
        raise ValueError("train and validation features do not line up -- "
                         "rebuild both with the same category vocabulary "
                         "(--force-preprocess)")

    cat_features = cat_feature_names()
    print(f"  categorical_feature: {cat_features}")

    train_data = lgb.Dataset(x_train, label=y_train, categorical_feature=cat_features)
    valid_data = lgb.Dataset(x_valid, label=y_valid, categorical_feature=cat_features,
                             reference=train_data)

    print(f"\n  amex_metric on the validation set every {PRINT_EVERY} rounds:")
    model = lgb.train(
        LGB_PARAMS,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[valid_data],
        valid_names=["valid"],
        feval=_amex_feval,
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            _print_metric(PRINT_EVERY),
        ],
    )

    best = amex_metric(pd.DataFrame({TARGET_COL: y_valid.values}),
                       pd.DataFrame({"prediction": model.predict(
                           x_valid, num_iteration=model.best_iteration)}))
    print(f"\n  best iteration: {model.best_iteration}")
    print(f"  final validation amex_metric: {best}")

    run_id = new_run_id(SEED)
    meta = {
        "run_id": run_id,
        "git": git_commit(),
        "split_id": split_id,
        "feature_version": FEATURE_VERSION,
        "feature_name": model.feature_name(),
        "cat_features": cat_features,
        "cat_cols": CAT_COLS,
        "categories": categories,
        "categories_digest": digest_of(categories) if categories else None,
        "best_iteration": model.best_iteration,
        "valid_amex": best,
        "params": LGB_PARAMS,
        "params_digest": digest_of(LGB_PARAMS),
        "trained_on": Path(train_features).name,
        "validated_on": Path(valid_features).name,
        "inputs": {Path(p).name: get(p) for p in (train_features, valid_features,
                                                  train_labels, valid_labels)},
    }
    model_path = save_run(run_id, model, meta)
    print(f"  run {run_id} -> {model_path.relative_to(MODEL_DIR.parent)}")

    history = list_runs()
    if len(history) > 1:
        best_so_far = max((r for r in history if r.get("score") is not None),
                          key=lambda r: r["score"])
        print(f"  {len(history)} runs recorded; best is {best_so_far['run_id']} "
              f"at {best_so_far['score']:.6f}")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default="train_data.parquet")
    ap.add_argument("--validation", default="validation_data.parquet")
    ap.add_argument("--labels", default="train_labels.parquet")
    ap.add_argument("--validation-labels", default="validation_labels.parquet")
    ap.add_argument("--force-preprocess", action="store_true",
                    help="rebuild the aggregated features even if they exist")
    ap.add_argument("--rescan-categories", action="store_true",
                    help="re-derive the vocabulary from the training split")
    args = ap.parse_args()

    # vocabulary first, from train only, then both splits aggregated with it
    categories = ensure_categories(DATA_DIR / args.input,
                                   force=args.rescan_categories)
    train_features = preprocess_if_missing(DATA_DIR / args.input,
                                           force=args.force_preprocess,
                                           categories=categories)
    valid_features = preprocess_if_missing(DATA_DIR / args.validation,
                                           force=args.force_preprocess,
                                           categories=categories)
    train_model(train_features, valid_features,
                DATA_DIR / args.labels, DATA_DIR / args.validation_labels,
                categories)


if __name__ == "__main__":
    main()
