"""
Build the feature-transformation pipeline.

The transformer is returned *unfitted* so it can be dropped into a single
sklearn Pipeline together with the estimator. Fitting then happens only on the
training fold during cross-validation / final fit, which guarantees there is no
information leakage from the test set into the scaler or the encoder.

Design:
  numeric      -> median impute  -> StandardScaler
  categorical  -> most-frequent impute -> OneHotEncoder(handle_unknown="ignore")
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for the project's feature schema."""
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop=None)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, config.NUMERIC_FEATURES),
            ("cat", categorical_pipe, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Expanded feature names after fitting (useful for RF importances / SHAP)."""
    return list(preprocessor.get_feature_names_out())
