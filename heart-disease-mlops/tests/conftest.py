"""
Shared fixtures.

We deliberately avoid any network access in the test suite: the synthetic
fixture from the download module gives us a schema-correct dataframe, and we
train a tiny model on it so the API/model tests have a real artifact to hit.
This is what lets the GitHub Actions pipeline run fully offline.
"""
from __future__ import annotations

import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.data.download_data import _synthetic
from src.data.preprocess import clean
from src.features.build_features import build_preprocessor
from src.models import predict as predict_mod


@pytest.fixture(scope="session")
def raw_df():
    """Deterministic synthetic raw frame (same schema as the UCI file)."""
    return _synthetic(n=300, seed=7)


@pytest.fixture(scope="session")
def clean_df(raw_df):
    return clean(raw_df)


@pytest.fixture(scope="session")
def trained_model_path(clean_df, tmp_path_factory):
    """Fit a small pipeline, persist it to the configured MODEL_PATH, return it."""
    X = clean_df[config.FEATURE_COLUMNS]
    y = clean_df[config.TARGET]
    pipe = Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=500, random_state=config.RANDOM_STATE)),
    ]).fit(X, y)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, config.MODEL_PATH)
    config.METADATA_PATH.write_text('{"model_family": "logreg_test"}')

    # bust the lru_cache so the API picks up the freshly-written model
    predict_mod.load_model.cache_clear()
    predict_mod.load_metadata.cache_clear()
    return config.MODEL_PATH
