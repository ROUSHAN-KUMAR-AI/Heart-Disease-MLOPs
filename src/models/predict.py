"""
Inference utilities shared by the API and the CLI.

Because the whole preprocessing + estimator chain was saved as one sklearn
Pipeline, prediction is just `pipe.predict_proba(df)`. No bespoke transform code
has to be kept in sync with training — that is the point of packaging the
pipeline rather than the bare estimator.

CLI:
    python -m src.models.predict '{"age":63,"sex":1,"cp":3,...}'
"""
from __future__ import annotations

import functools
import json
import sys

import pandas as pd

from src import config


@functools.lru_cache(maxsize=1)
def load_model():
    """Load (and cache) the packaged pipeline. Raises if the model is missing."""
    import joblib

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{config.MODEL_PATH} not found. Train first: `python -m src.models.train`."
        )
    return joblib.load(config.MODEL_PATH)


@functools.lru_cache(maxsize=1)
def load_metadata() -> dict:
    if config.METADATA_PATH.exists():
        return json.loads(config.METADATA_PATH.read_text())
    return {}


def _to_frame(record: dict | list[dict]) -> pd.DataFrame:
    records = [record] if isinstance(record, dict) else record
    df = pd.DataFrame(records)
    missing = set(config.FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing required features: {sorted(missing)}")
    return df[config.FEATURE_COLUMNS]


def predict(record: dict | list[dict]) -> list[dict]:
    """Return [{prediction, probability, label}] for each input record."""
    model = load_model()
    df = _to_frame(record)
    proba = model.predict_proba(df)[:, 1]
    out = []
    for p in proba:
        pred = int(p >= 0.5)
        out.append({
            "prediction": pred,
            "probability": round(float(p), 4),
            "label": "heart disease present" if pred == 1 else "no heart disease",
        })
    return out


if __name__ == "__main__":
    payload = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(json.dumps(predict(payload), indent=2))
