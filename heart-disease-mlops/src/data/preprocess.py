"""
Clean the raw Heart Disease CSV and produce train/test splits.

Responsibilities (kept separate from feature *transformation*, which lives in
src/features/build_features.py and is fitted only on the training fold):

  * replace the UCI "?" sentinels with proper NaN
  * coerce every column to numeric
  * binarise the 0-4 severity target into {0, 1}
  * drop exact duplicate rows
  * stratified train/test split

Run:
    python -m src.data.preprocess
"""
from __future__ import annotations

import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_raw() -> pd.DataFrame:
    if not config.RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{config.RAW_DATA_PATH} not found. Run `python -m src.data.download_data` first."
        )
    return pd.read_csv(config.RAW_DATA_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy frame with FEATURE_COLUMNS + binary TARGET."""
    df = df.copy()

    # 1. "?" -> NaN, then force numeric (ca/thal arrive as strings in the raw file)
    df = df.replace("?", pd.NA)
    for col in config.RAW_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Binarise target: any severity >= 1 counts as "disease present"
    df[config.TARGET] = (df[config.RAW_TARGET] > 0).astype(int)
    df = df.drop(columns=[config.RAW_TARGET])

    # 3. Drop rows with a missing target (none expected) and exact duplicates
    df = df.dropna(subset=[config.TARGET]).drop_duplicates().reset_index(drop=True)

    # NOTE: feature-level missing values (ca/thal) are intentionally left as NaN.
    # They are imputed *inside* the fitted sklearn pipeline to avoid leakage.
    ordered = config.FEATURE_COLUMNS + [config.TARGET]
    return df[ordered]


def split_and_save(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df[config.TARGET],
    )
    df.to_csv(config.CLEAN_DATA_PATH, index=False)
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    test_df.to_csv(config.TEST_DATA_PATH, index=False)

    print(f"[preprocess] clean={df.shape}  train={train_df.shape}  test={test_df.shape}")
    print(f"[preprocess] class balance (full): "
          f"{df[config.TARGET].value_counts(normalize=True).round(3).to_dict()}")
    return train_df, test_df


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_raw()
    clean_df = clean(raw)
    return split_and_save(clean_df)


if __name__ == "__main__":
    run()
    sys.exit(0)
