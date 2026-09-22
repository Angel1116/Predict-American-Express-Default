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
import os
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, HTMLResponse

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from config import FEATURE_VERSION, ID_COL, TARGET_COL, load_categories
from evaluation import amex_metric
from preprocess import aggregate_frame
from registry import load_meta, resolve_model

# Sized for a 512 MB container. Raise it with the env var on a bigger one.
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "40"))

PREDICT = "New Data Prediction"
EVALUATE = "Predictive Performance (AUC)"

app = FastAPI(
    title="American Express - Default Prediction",
    description=(
        "Upload monthly statement rows the way the raw data ships them -- one "
        "row per customer per statement. The aggregation into per-customer "
        "features runs here, with the same code that built the training set."
    ),
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,   # hide the generated multipart bodies
        "tryItOutEnabled": True,          # skip the extra click before uploading
    },
    openapi_tags=[
        {"name": PREDICT,
         "description": "Score customers whose outcome is not known yet. "
                        "Returns a csv, one probability per customer."},
        {"name": EVALUATE,
         "description": "Score customers whose outcome *is* known, and compare "
                        "against it. Returns the AUC."},
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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    """The two-step page. Kept out of the schema -- it is a page, not an API."""
    page = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    return (page.replace("{{RUN_ID}}", RUN_ID)
                .replace("{{N_FEATURES}}", str(len(FEATURES))))


SAMPLES = {"data": "demo_data.parquet", "labels": "demo_labels.parquet"}


@app.get("/sample/{which}", include_in_schema=False)
def sample(which: str):
    """Hand out the bundled sample so the page is usable on its own."""
    name = SAMPLES.get(which)
    path = Path(__file__).parent / "data" / name if name else None
    if path is None or not path.exists():
        raise HTTPException(404, f"no sample named {which!r}")
    return FileResponse(path, filename=name,
                        media_type="application/octet-stream")


async def _read_upload(upload):
    """Accept either of the two formats this project writes, within budget."""
    contents = await upload.read()
    size_mb = len(contents) / 1e6

    # A compressed statement file expands roughly six-fold once pandas has it
    # -- float16 unpacks and every customer_ID becomes a python string. Past
    # this the container runs out of memory and dies mid-request, which the
    # caller sees as a 502 with nothing to act on. Better to say so.
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            413,
            f"{upload.filename} is {size_mb:.0f} MB; this instance accepts up "
            f"to {MAX_UPLOAD_MB:.0f} MB. Score it in smaller batches, or run "
            f"the pipeline locally with `python code/pipeline.py predict`.")

    try:
        frame = pd.read_parquet(io.BytesIO(contents))
    except Exception:
        try:
            frame = pd.read_csv(io.BytesIO(contents))
        except Exception as exc:
            raise HTTPException(
                400, f"could not read {upload.filename} as parquet or csv: {exc}")

    del contents        # the parsed frame is the only copy worth keeping
    return frame


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
    summary="Upload statements, download predictions as csv",
    response_class=Response,
    responses={200: {"description": "A csv with two columns: customer_ID and "
                                    "predicted_probability.",
                     "content": {"text/csv": {}}}},
)
async def predict(
    file: UploadFile = File(
        ..., description="Statement rows, parquet or csv. e.g. test_data.parquet"),
):
    scores = _score(await _read_upload(file))
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
    summary="Upload statements and their labels, get the AUC",
    # content: None keeps the docs page from carrying a made-up example.
    # Leaving `responses` out entirely does the opposite of what that sounds
    # like -- FastAPI then fills in a default json schema, and Swagger renders
    # it as the placeholder "string". The real shape shows up under Execute.
    responses={200: {"description": "The area under the ROC curve, 0.5 being "
                                    "no better than chance and 1.0 perfect.",
                     "content": None}},
)
async def evaluate(
    response: Response,
    data: UploadFile = File(
        ..., description="Statement rows, parquet or csv. e.g. test_data.parquet"),
    labels: UploadFile = File(
        ..., description="customer_ID and target. e.g. test_labels.parquet"),
):
    scores = _score(await _read_upload(data))
    truth = await _read_upload(labels)

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

    # The body is the one number that was asked for, but an AUC computed on a
    # subset because half the labels were missing looks exactly like one
    # computed on everything. The headers keep that visible without putting it
    # in the reader's way.
    response.headers["X-Run-Id"] = RUN_ID
    response.headers["X-Customers-Evaluated"] = str(len(merged))
    response.headers["X-Customers-Unmatched"] = str(len(scores) - len(merged))

    return {"auc": round(metric["auc"], 6)}


def _openapi():
    """Trim the auto-generated validation block out of the docs page.

    FastAPI adds a 422 response and its schema to every route that takes a
    body. Both endpoints here already return 422 with a readable message, so
    the generated version only adds a schema tree to scroll past.
    """
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(title=app.title, version=app.version,
                         description=app.description, routes=app.routes,
                         tags=app.openapi_tags)
    for path in schema.get("paths", {}).values():
        for operation in path.values():
            operation.get("responses", {}).pop("422", None)
    for name in ("HTTPValidationError", "ValidationError"):
        schema.get("components", {}).get("schemas", {}).pop(name, None)

    app.openapi_schema = schema
    return schema


app.openapi = _openapi
