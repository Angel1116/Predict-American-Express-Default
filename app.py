"""Two endpoints over one trained model: score new statements, or evaluate them.

Both take the *raw* statement-level file -- the same shape split_data.py
writes -- and run the aggregation server side, through the same preprocess
code the training run used. Asking callers to send the 775 aggregated
features instead would hand them the feature engineering, and the first time
their version drifted from ours the model would be scored on inputs it was
never trained on.

    uvicorn app:app --reload
    open http://127.0.0.1:8000/docs
"""

import io
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.openapi.utils import get_openapi

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from config import FEATURE_VERSION, ID_COL, TARGET_COL, load_categories
from evaluation import amex_metric
from preprocess import aggregate_frame
from registry import load_meta, resolve_model

PREDICT = "PREDICTION NEW DATA"
EVALUATE = "EVALUATION"

app = FastAPI(
    title="Amex default prediction",
    # The models block at the foot of the page only ever lists the generated
    # multipart bodies, which say nothing a file picker does not already show.
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    openapi_tags=[
        {"name": PREDICT,
         "description": "Upload statement rows. Get a csv back, one "
                        "default probability per customer."},
        {"name": EVALUATE,
         "description": "Upload statement rows together with their known "
                        "outcomes, and see how well the model did."},
    ],
)

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


def _read_upload(upload, contents):
    """Accept either of the two formats this project writes."""
    try:
        return pd.read_parquet(io.BytesIO(contents))
    except Exception:
        pass
    try:
        return pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            400, f"could not read {upload.filename} as parquet or csv: {exc}")


def _score(statements):
    """Statement rows in, one probability per customer out."""
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

    return pd.DataFrame({
        ID_COL: features[ID_COL],
        "predicted_probability": model.predict(features[FEATURES]),
    })


@app.post(
    "/predict",
    tags=[PREDICT],
    summary="Upload statement rows, download a csv of probabilities",
    response_class=Response,
    responses={200: {"description": "csv: customer_ID, predicted_probability",
                     "content": {"text/csv": {}}}},
)
async def predict(file: UploadFile = File(..., description="statement-level parquet or csv")):
    scores = _score(_read_upload(file, await file.read()))
    stem = Path(file.filename or "data").stem
    return Response(
        content=scores.to_csv(index=False),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="predicted_{stem}.csv"',
            "X-Run-Id": RUN_ID,
            "X-Customers-Scored": str(len(scores)),
        },
    )


@app.post(
    "/evaluate",
    tags=[EVALUATE],
    summary="Upload statement rows and their labels, see the AUC",
    responses={200: {"description": "the model's score against the labels you supplied",
                     "content": {"application/json": {"example": {
                         "auc": 0.963055,
                         "amex_score": 0.793863,
                         "gini": 0.926109,
                         "top4_capture": 0.661617,
                         "customers_evaluated": 45892,
                         "positive_rate": 0.2589,
                         "run_id": "20260921-205141-s42",
                     }}}}},
)
async def evaluate(
    data: UploadFile = File(..., description="statement-level parquet or csv"),
    labels: UploadFile = File(..., description="customer_ID and target"),
):
    scores = _score(_read_upload(data, await data.read()))
    truth = _read_upload(labels, await labels.read())

    if TARGET_COL not in truth.columns or ID_COL not in truth.columns:
        raise HTTPException(
            422, f"the labels file needs columns '{ID_COL}' and '{TARGET_COL}', "
                 f"found {list(truth.columns)[:6]}")

    merged = scores.merge(truth[[ID_COL, TARGET_COL]], on=ID_COL, how="inner")
    if merged.empty:
        raise HTTPException(
            422, "no customer_ID appears in both files -- they look like "
                 "different datasets")

    metric = amex_metric(
        merged[[TARGET_COL]].astype("int8").reset_index(drop=True),
        merged[["predicted_probability"]].rename(
            columns={"predicted_probability": "prediction"}).reset_index(drop=True))

    return {
        "auc": round(metric["auc"], 6),
        "amex_score": round(metric["score"], 6),
        "gini": round(metric["gini"], 6),
        "top4_capture": round(metric["top4"], 6),
        "customers_evaluated": len(merged),
        "unmatched_customers": len(scores) - len(merged),
        "positive_rate": round(float(merged[TARGET_COL].mean()), 4),
        "run_id": RUN_ID,
    }


def _openapi():
    """Trim the auto-generated validation block out of the docs page.

    FastAPI adds a 422 response and its schema to every route that takes a
    body. Both endpoints here already return 422 with a readable message, so
    the generated version only adds a schema tree to scroll past.
    """
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(title=app.title, version=app.version,
                         routes=app.routes, tags=app.openapi_tags)
    for path in schema.get("paths", {}).values():
        for operation in path.values():
            operation.get("responses", {}).pop("422", None)
    for name in ("HTTPValidationError", "ValidationError"):
        schema.get("components", {}).get("schemas", {}).pop(name, None)

    app.openapi_schema = schema
    return schema


app.openapi = _openapi
