"""Tests for model training behaviour."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.features.build_features import build_preprocessor


def test_pipeline_fits_and_predicts_probabilities(clean_df):
    X = clean_df[config.FEATURE_COLUMNS]
    y = clean_df[config.TARGET]
    pipe = Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=500, random_state=config.RANDOM_STATE)),
    ]).fit(X, y)

    proba = pipe.predict_proba(X)[:, 1]
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    assert set(np.unique(pipe.predict(X))) <= {0, 1}


def test_model_learns_better_than_chance(clean_df):
    from sklearn.metrics import roc_auc_score
    X = clean_df[config.FEATURE_COLUMNS]
    y = clean_df[config.TARGET]
    pipe = Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=500, random_state=config.RANDOM_STATE)),
    ]).fit(X, y)
    auc = roc_auc_score(y, pipe.predict_proba(X)[:, 1])
    assert auc > 0.6  # sanity: signal exists in the (synthetic) data
