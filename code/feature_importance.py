"""Rank the features a trained booster actually used, and chart the top ones.

Two views of the same numbers:

  feature importance-<run_id>.png         the raw features, as the model sees them
  feature importance_merged-<run_id>.png  summed back onto the source column

The merged view exists because each source column becomes four features
(_mean, _max, _last, _diff) that are highly correlated by construction. Gain
gets divided among them, so a column that dominates the model can look
mid-table in the raw ranking. Summing the four back together is the honest
answer to "which measurement matters".

Importance comes out of the saved model.txt, so nothing here retrains or even
needs the training data.

    python code/feature_importance.py
    python code/feature_importance.py --run 20260921-205141-s42 --top 25
"""

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CAT_COLS, ROOT  # noqa: E402
from registry import load_meta, resolve_model  # noqa: E402

OUT_DIR = ROOT / "feature_importance"
SUFFIXES = ("_mean", "_max", "_last", "_diff")

# One series, one colour. Shading bars by their own rank would encode the
# ranking twice and say nothing the ordering does not already say.
BAR = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dcdcd8"


def base_column(feature):
    """Map a derived feature back to the source column it came from."""
    for col in CAT_COLS:
        # one-hot shares (B_30_1) and the last-level code (B_30_last) alike
        if feature.startswith(col + "_"):
            return col
    for suffix in SUFFIXES:
        if feature.endswith(suffix):
            return feature[:-len(suffix)]
    return feature


def importance_table(model):
    """Every feature with its gain, share of total gain, and split count."""
    table = pd.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance("gain"),
        "splits": model.feature_importance("split"),
    })
    table["gain_pct"] = 100 * table["gain"] / table["gain"].sum()
    table["base"] = table["feature"].map(base_column)
    return table.sort_values("gain", ascending=False).reset_index(drop=True)


def merge_by_base(table):
    """Sum the four derived features back onto their source column."""
    merged = (table.groupby("base")
              .agg(gain=("gain", "sum"), gain_pct=("gain_pct", "sum"),
                   splits=("splits", "sum"), parts=("feature", "size"))
              .sort_values("gain", ascending=False)
              .reset_index()
              .rename(columns={"base": "feature"}))
    return merged


def plot(table, title, subtitle, path, top):
    rows = table.head(top).iloc[::-1]          # barh draws bottom-up
    labels, values = rows["feature"], rows["gain_pct"]

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 2.1), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.barh(labels, values, height=0.6, color=BAR, zorder=3)

    # room for the value labels, so the longest bar does not run into the edge
    ax.set_xlim(0, values.max() * 1.14)
    for y, value in enumerate(values):
        ax.text(value + values.max() * 0.012, y, f"{value:.2f}%",
                va="center", ha="left", fontsize=9, color=MUTED, zorder=4)

    ax.set_xlabel("Gain (% of total)", fontsize=10, color=MUTED, labelpad=9)
    ax.tick_params(axis="y", length=0, labelsize=9.5, colors=INK)
    ax.tick_params(axis="x", length=0, labelsize=9, colors=MUTED)

    # recessive chrome: one soft grid direction, no box around the plot
    ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    # the title is padded far enough up to leave the subtitle its own line;
    # both are measured in points so the gap holds at any figure height
    ax.set_title(title, fontsize=13, color=INK, loc="left", pad=32)
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 9), textcoords="offset points",
                fontsize=9.5, color=MUTED, va="bottom", ha="left")

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def report(run_id=None, top=20):
    model_path = resolve_model(run_id)
    run = model_path.parent.name
    meta = load_meta(model_path) or {}

    model = lgb.Booster(model_file=str(model_path))
    table = importance_table(model)
    merged = merge_by_base(table)

    unused = int((table["gain"] == 0).sum())
    print(f"Feature importance for run {run}")
    print(f"  {len(table)} features, {unused} never used by any split")
    if meta.get("valid_amex", {}).get("score"):
        print(f"  validation amex score {meta['valid_amex']['score']:.6f}")

    print(f"\n=== top {top} features by gain ===")
    for i, row in table.head(top).iterrows():
        print(f"  {i + 1:>2}. {row.feature:24s} {row.gain_pct:6.2f}%   "
              f"{int(row.splits):>5,} splits")
    print(f"  these {top} account for {table.head(top).gain_pct.sum():.1f}% of all gain")

    print(f"\n=== top {top} source columns, derived features merged ===")
    for i, row in merged.head(top).iterrows():
        print(f"  {i + 1:>2}. {row.feature:24s} {row.gain_pct:6.2f}%   "
              f"from {int(row.parts)} derived feature(s)")
    print(f"  these {top} account for {merged.head(top).gain_pct.sum():.1f}% of all gain")

    OUT_DIR.mkdir(exist_ok=True)
    outputs = [
        plot(table, f"Feature importance - {run}",
             f"Top {top} of {len(table)} features by gain",
             OUT_DIR / f"feature importance-{run}.png", top),
        plot(merged, f"Feature importance (merged) - {run}",
             f"Top {top} of {len(merged)} source columns, "
             f"derived features summed back together",
             OUT_DIR / f"feature importance_merged-{run}.png", top),
    ]
    # the full ranking, since the charts only show the head of it
    table.to_csv(OUT_DIR / f"gain-{run}.csv", index=False)
    merged.to_csv(OUT_DIR / f"gain_merged-{run}.csv", index=False)

    print()
    for path in outputs:
        print(f"  wrote {path.relative_to(ROOT)}")
    print(f"  wrote {(OUT_DIR / f'gain-{run}.csv').relative_to(ROOT)} "
          f"and gain_merged-{run}.csv")
    return table, merged


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=None,
                    help="a run id from models/runs.jsonl (default: latest)")
    ap.add_argument("--top", type=int, default=20,
                    help="how many bars to chart (default: %(default)s)")
    args = ap.parse_args()
    report(args.run, args.top)


if __name__ == "__main__":
    main()
