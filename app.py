"""Batch scoring API: post statement rows, get one probability per customer.

The endpoint takes the *raw* statement-level file -- the same shape
split_data.py writes -- and runs the aggregation server side, through the
same preprocess code the training run used. Asking callers to send the 775
aggregated features instead would hand them the feature engineering, and the
first time their version drifted from ours the model would be scored on
inputs it was never trained on.

    uvicorn app:app --reload
    open http://127.0.0.1:8000/docs
"""

import io
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Response, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from config import FEATURE_VERSION, ID_COL, load_categories
from preprocess import aggregate_frame
from registry import load_meta, resolve_model

app = FastAPI(title="Amex default prediction")

# Loaded once at boot, not per request. resolve_model() with no argument
# follows models/latest.json; pass a run id to pin a version, which is what
# a deployment should do so that retraining never changes a live endpoint
# without someone deciding it should.
MODEL_PATH = resolve_model()
model = lgb.Booster(model_file=str(MODEL_PATH))
META = load_meta(MODEL_PATH) or {}
RUN_ID = META.get("run_id", MODEL_PATH.parent.name)
CATEGORIES = load_categories()
FEATURES = model.feature_name()

print(f"loaded run {RUN_ID}: {model.num_trees()} trees, {len(FEATURES)} features")


@app.get("/health")
def health():
    """Liveness plus the two things that make a 200 meaningful."""
    stale = META.get("feature_version") != FEATURE_VERSION
    return {
        "status": "degraded" if stale else "ok",
        "run_id": RUN_ID,
        "n_features": len(FEATURES),
        "feature_version": META.get("feature_version"),
        "config_feature_version": FEATURE_VERSION,
        "detail": ("model and config disagree on feature_version -- retrain"
                   if stale else None),
    }


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """Score a statement-level parquet and return it with a probability column."""
    contents = await file.read()
    try:
        statements = pd.read_parquet(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(400, f"could not read {file.filename} as parquet: {exc}")

    try:
        features = aggregate_frame(statements, CATEGORIES)
    except KeyError as exc:
        raise HTTPException(422, f"input is missing columns the model needs: {exc}")

    # Booster.predict lines features up by POSITION, so a frame with the right
    # columns in the wrong order scores happily and wrongly. Reindexing by the
    # order stored with the model turns that into a 422 instead.
    absent = [c for c in FEATURES if c not in features.columns]
    if absent:
        raise HTTPException(
            422,
            f"aggregation produced {len(features.columns) - 1} features but the "
            f"model expects {len(FEATURES)}; {len(absent)} missing, "
            f"first few: {absent[:5]}")

    out = pd.DataFrame({
        ID_COL: features[ID_COL],
        "predicted_probability": model.predict(features[FEATURES]),
        "run_id": RUN_ID,
    })

    buffer = io.BytesIO()
    out.to_parquet(buffer, index=False)
    return Response(
        content=buffer.getvalue(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=predicted_{file.filename}",
            "X-Run-Id": RUN_ID,
            "X-Customers-Scored": str(len(out)),
        },
    )
