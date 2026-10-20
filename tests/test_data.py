"""Tests for data cleaning / preprocessing."""

from src import config
from src.data.download_data import _synthetic
from src.data.preprocess import clean


def test_synthetic_has_expected_schema():
    df = _synthetic(n=50, seed=1)
    assert list(df.columns) == config.RAW_COLUMNS
    assert len(df) == 50


def test_clean_binarises_target(raw_df):
    out = clean(raw_df)
    assert set(out[config.TARGET].unique()) <= {0, 1}
    assert config.RAW_TARGET not in out.columns


def test_clean_replaces_question_marks_with_nan():
    df = _synthetic(n=60, seed=2)  # injects "?" into ca/thal
    assert (df == "?").any().any()
    out = clean(df)
    # "?" must be gone; columns are numeric now
    assert out["ca"].dtype.kind in "fi"
    assert out["thal"].dtype.kind in "fi"


def test_clean_keeps_only_schema_columns(clean_df):
    assert list(clean_df.columns) == config.FEATURE_COLUMNS + [config.TARGET]


def test_no_missing_target(clean_df):
    assert clean_df[config.TARGET].isna().sum() == 0
