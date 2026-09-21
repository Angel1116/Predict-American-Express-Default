"""Single entry point: say which stage to run.

    python code/pipeline.py train
    python code/pipeline.py predict --labels test_labels.parquet

This module only routes arguments -- the work still lives in preprocess.py,
train.py and predict.py, and each of those remains runnable on its own. Both
stages call preprocess_if_missing first, so aggregated features are built on
demand and reused afterwards.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, load_categories
from predict import predict
from preprocess import ensure_categories, preprocess_if_missing
from train import train_model


def _in_data_dir(name):
    """Bare filenames are looked up in data/; explicit paths are left alone."""
    if name is None:
        return None
    path = Path(name)
    return path if path.is_absolute() else DATA_DIR / path


def run_train(args):
    # vocabulary first, from the training split only, then both splits
    # aggregated with it
    categories = ensure_categories(_in_data_dir(args.input),
                                   force=args.rescan_categories)
    train_features = preprocess_if_missing(_in_data_dir(args.input),
                                           force=args.force_preprocess,
                                           categories=categories)
    valid_features = preprocess_if_missing(_in_data_dir(args.validation),
                                           force=args.force_preprocess,
                                           categories=categories)
    train_model(train_features, valid_features,
                _in_data_dir(args.labels),
                _in_data_dir(args.validation_labels),
                categories)


def run_predict(args):
    features = preprocess_if_missing(_in_data_dir(args.input),
                                     force=args.force_preprocess,
                                     categories=load_categories())
    # labels are the default rather than an extra, so a bare `predict` scores
    # and evaluates in one go; --no-eval is for data that has no labels yet
    labels = None if args.no_eval else _in_data_dir(args.labels)
    predict(features,
            _in_data_dir(args.model) if args.model else None,
            labels,
            _in_data_dir(args.output) or DATA_DIR / "predictions.parquet",
            args.run)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True, metavar="{train,predict}")

    t = sub.add_parser("train", help="fit a booster and save it to models/")
    t.add_argument("--input", default="train_data.parquet",
                   help="statement-level parquet (default: %(default)s)")
    t.add_argument("--validation", default="validation_data.parquet",
                   help="held-out split used for early stopping (default: %(default)s)")
    t.add_argument("--labels", default="train_labels.parquet")
    t.add_argument("--validation-labels", default="validation_labels.parquet")
    t.add_argument("--force-preprocess", action="store_true",
                   help="rebuild the aggregated features even if they exist")
    t.add_argument("--rescan-categories", action="store_true",
                   help="re-derive the vocabulary from the training split")
    t.set_defaults(func=run_train)

    p = sub.add_parser("predict", help="score a dataset with a saved booster")
    p.add_argument("--input", default="test_data.parquet",
                   help="statement-level parquet (default: %(default)s)")
    p.add_argument("--labels", default="test_labels.parquet",
                   help="labels to evaluate against (default: %(default)s)")
    p.add_argument("--no-eval", action="store_true",
                   help="just score, for data that has no labels")
    p.add_argument("--model", default=None, help="a booster file to use directly")
    p.add_argument("--run", default=None,
                   help="a run id from models/runs.jsonl (default: latest)")
    p.add_argument("--output", default=None,
                   help="where to write predictions (default: data/predictions.parquet)")
    p.add_argument("--force-preprocess", action="store_true")
    p.set_defaults(func=run_predict)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
