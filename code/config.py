"""Shared configuration for the Amex default-prediction pipeline.

Everything here is a fact about the *data*, not about any particular model,
so preprocessing and training can both depend on it without depending on
each other. Adding a categorical column is a one-line change in one place.

The one exception is CATEGORIES, which is *derived* rather than declared:
train.py scans the training split and writes data/categories.json, and this
module loads it. Deriving it from the training data alone -- rather than
from every row in the dataset -- keeps information about the validation and
test splits out of the feature definitions.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

ID_COL = "customer_ID"
TARGET_COL = "target"
DATE_COL = "S_2"
NON_FEATURE_COLS = [ID_COL, DATE_COL]

SEED = 42

CAT_COLS = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120",
            "D_126", "D_63", "D_64", "D_66", "D_68"]

# Every column gets this level, even when the training split has no missing
# values there. It is the landing spot for anything unseen at inference time,
# which is only possible now that the vocabulary comes from train alone.
MISSING_LEVEL = "NaN"

CATEGORIES_PATH = DATA_DIR / "categories.json"

# Bump when CATEGORIES or the aggregation changes, so a model trained on old
# features can be told apart from one trained on new ones.
FEATURE_VERSION = 2


def load_categories(path=None):
    """The level list for each categorical column, as scanned from train."""
    path = Path(path) if path else CATEGORIES_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python code/pipeline.py train`, which "
            f"derives the vocabulary from the training split before it "
            f"preprocesses anything")
    return json.loads(path.read_text())


def save_categories(categories, path=None):
    path = Path(path) if path else CATEGORIES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(categories, indent=2))
    return path


# Convenient for interactive use; modules that run before the first scan
# should call load_categories() themselves rather than read this.
CATEGORIES = json.loads(CATEGORIES_PATH.read_text()) if CATEGORIES_PATH.exists() else {}


def cat_feature_names():
    """The categorical columns as they are named after aggregation."""
    return [f"{col}_last" for col in CAT_COLS]
