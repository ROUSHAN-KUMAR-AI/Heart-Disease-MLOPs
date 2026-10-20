"""
FastAPI model-serving application — assignment tasks 6 & 8.

Endpoints
  GET  /            -> service banner
  GET  /health      -> liveness/readiness probe (used by k8s)
  POST /predict     -> JSON in, {prediction, probability, label} out
  GET  /metrics     -> Prometheus exposition format (scraped by Prometheus)
  GET  /docs        -> interactive Swagger UI (FastAPI built-in)

Observability
  * every request is logged as a structured JSON line (stdout -> container logs)
  * Prometheus counters/histograms track request volume, latency and predictions

Run locally:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import json
import logging
import sys
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest,
)

from src import config
from src.api.schemas import HealthResponse, PatientFeatures, PredictionResponse
from src.models import predict as predict_mod

# ---------------------------------------------------------------------------
# Structured JSON logging -> stdout (captured by Docker / k8s / Loki)
# ---------------------------------------------------------------------------
logger = logging.getLogger("heart-api")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.propagate = False


def log_event(**fields) -> None:
    logger.info(json.dumps({"ts": time.time(), **fields}))


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "Request latency (s)", ["endpoint"]
)
PREDICTIONS = Counter(
    "model_predictions_total", "Predictions served", ["label"]
)
PREDICTION_SCORE = Histogram(
    "model_prediction_probability", "Distribution of predicted probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Heart Disease Risk API",
    description="Predict the risk of heart disease from patient health data.",
    version="1.0.0",
)


@app.on_event("startup")
def _startup() -> None:
    """Warm the model cache so the first request isn't slow (and fail fast if absent)."""
    try:
        predict_mod.load_model()
        meta = predict_mod.load_metadata()
        log_event(event="startup", model_loaded=True,
                  model_family=meta.get("model_family"))
    except FileNotFoundError as exc:
        # Don't crash the container; /health will report model_loaded=False.
        log_event(event="startup", model_loaded=False, error=str(exc))


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    endpoint = request.url.path
    REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint).observe(elapsed)
    log_event(event="request", method=request.method, path=endpoint,
              status=response.status_code, latency_ms=round(elapsed * 1000, 2))
    return response


@app.get("/")
def root():
    return {"service": "heart-disease-risk-api", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        predict_mod.load_model()
        meta = predict_mod.load_metadata()
        return HealthResponse(status="ok", model_loaded=True,
                              model_family=meta.get("model_family"))
    except FileNotFoundError:
        return HealthResponse(status="degraded", model_loaded=False)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    result = predict_mod.predict(features.model_dump())[0]
    meta = predict_mod.load_metadata()

    PREDICTIONS.labels(result["label"]).inc()
    PREDICTION_SCORE.observe(result["probability"])
    log_event(event="prediction", **result)

    return PredictionResponse(model_family=meta.get("model_family", "unknown"), **result)


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    log_event(event="error", path=request.url.path, error=str(exc))
    return JSONResponse(status_code=422, content={"detail": str(exc)})
