"""Tests for the feature-transformation pipeline."""
import numpy as np

from src import config
from src.features.build_features import build_preprocessor, get_feature_names


def test_preprocessor_fit_transform_shape(clean_df):
    pre = build_preprocessor()
    X = clean_df[config.FEATURE_COLUMNS]
    Xt = pre.fit_transform(X)
    # rows preserved; columns expand because of one-hot encoding
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] >= len(config.NUMERIC_FEATURES)


def test_preprocessor_handles_missing_values(clean_df):
    pre = build_preprocessor()
    X = clean_df[config.FEATURE_COLUMNS].copy()
    X.loc[0, "ca"] = np.nan
    X.loc[1, "chol"] = np.nan
    Xt = pre.fit_transform(X)
    assert not np.isnan(np.asarray(Xt.todense() if hasattr(Xt, "todense") else Xt)).any()


def test_unknown_category_is_ignored(clean_df):
    pre = build_preprocessor()
    X = clean_df[config.FEATURE_COLUMNS]
    pre.fit(X)
    novel = X.iloc[[0]].copy()
    novel.loc[novel.index[0], "thal"] = 999  # unseen category
    # handle_unknown="ignore" => must not raise
    out = pre.transform(novel)
    assert out.shape[0] == 1


def test_feature_names_available(clean_df):
    pre = build_preprocessor()
    pre.fit(clean_df[config.FEATURE_COLUMNS])
    names = get_feature_names(pre)
    assert len(names) >= len(config.NUMERIC_FEATURES)
