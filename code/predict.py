"""Score a preprocessed table with a saved booster, and evaluate it if labels exist.

The same preprocess.py that built the training features builds the ones fed
in here, so nothing in this module knows how a feature is computed -- it only
checks that what it was handed matches what the model was fitted on.

    python code/predict.py --input test_data.parquet
    python code/predict.py --input test_data.parquet --labels test_labels.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (DATA_DIR, FEATURE_VERSION, ID_COL, MODEL_DIR, SEED,
                    TARGET_COL, load_categories)
from evaluation import amex_metric
from preprocess import preprocess_if_missing


def predict(features_path, model_path=None, labels_path=None, out_path=None):
    """Return per-customer default probabilities for `features_path`."""
    features_path = Path(features_path)
    model_path = Path(model_path) if model_path else MODEL_DIR / f"lgbm_seed{SEED}.txt"

    print(f"\nScoring {features_path.name} with {model_path.name}...")
    model = lgb.Booster(model_file=str(model_path))
    df = pd.read_parquet(features_path)

    meta_path = model_path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"{model_path.name} was trained on feature_version "
                f"{meta.get('feature_version')}, config.py is now at {FEATURE_VERSION} "
                f"-- retrain, or check out the matching config")

    # Booster.predict matches columns by POSITION, not by name: a reordered
    # frame scores silently and wrongly. Reindexing by the stored feature
    # order turns that into a loud KeyError instead.
    expected = model.feature_name()
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(f"{len(missing)} feature(s) missing from {features_path.name}, "
                       f"first few: {missing[:5]}")
    X = df[expected]

    pred = pd.DataFrame({ID_COL: df[ID_COL],
                         "prediction": model.predict(X)})

    if labels_path:
        labels = pd.read_parquet(labels_path)
        merged = pred.merge(labels, on=ID_COL, how="inner")
        if len(merged) != len(pred):
            print(f"  warning: {len(pred) - len(merged)} customers had no label")
        truth = merged[[TARGET_COL]].astype("int8").reset_index(drop=True)
        score = amex_metric(truth, merged[["prediction"]].reset_index(drop=True))
        print(f"  {len(merged):,} customers evaluated, "
              f"{truth[TARGET_COL].mean():.1%} positive")
        print("  amex_metric: " + str({k: round(v, 6) for k, v in score.items()}))

    if out_path:
        out_path = Path(out_path)
        pred.to_parquet(out_path, index=False)
        print(f"  wrote {out_path.name}")
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default="test_data.parquet",
                    help="statement-level parquet inside the data directory")
    ap.add_argument("--labels", default="test_labels.parquet",
                    help="labels to evaluate against (default: %(default)s)")
    ap.add_argument("--no-eval", action="store_true",
                    help="just score, for data that has no labels")
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", default=None, help="where to write predictions")
    ap.add_argument("--force-preprocess", action="store_true")
    args = ap.parse_args()

    # the vocabulary scanned from train at fit time, never re-derived here
    features = preprocess_if_missing(DATA_DIR / args.input,
                                     force=args.force_preprocess,
                                     categories=load_categories())
    labels = None if args.no_eval else DATA_DIR / args.labels
    out = Path(args.output) if args.output else DATA_DIR / "predictions.parquet"
    predict(features, args.model, labels, out)


if __name__ == "__main__":
    main()
