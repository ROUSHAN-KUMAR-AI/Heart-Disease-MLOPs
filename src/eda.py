"""
Exploratory Data Analysis — assignment task 1.

Generates professional visualisations into reports/figures/ and a summary CSV:
  * class-balance bar chart
  * numeric-feature histograms (by target)
  * correlation heatmap
  * missing-value summary

These same figures are embedded in the final report and can be logged to MLflow.

Run (after download + preprocess):
    python -m src.eda
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config

sns.set_theme(style="whitegrid")


def _load() -> pd.DataFrame:
    if config.CLEAN_DATA_PATH.exists():
        return pd.read_csv(config.CLEAN_DATA_PATH)
    # fall back to raw if preprocess hasn't run yet
    from src.data.preprocess import clean, load_raw
    return clean(load_raw())


def run() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = _load()
    print(f"[eda] data shape: {df.shape}")

    # --- class balance ---
    fig, ax = plt.subplots(figsize=(4.5, 4))
    counts = df[config.TARGET].value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, hue=counts.index.astype(str),
                palette="Set2", legend=False)
    ax.set(xlabel="target (0 = no disease, 1 = disease)", ylabel="count",
           title="Class balance")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha="center")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "class_balance.png", dpi=120); plt.close(fig)

    # --- numeric histograms, split by target ---
    n = len(config.NUMERIC_FEATURES)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.4))
    for ax, col in zip(axes, config.NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=config.TARGET, kde=True, ax=ax,
                     palette="Set1", alpha=0.5, legend=(ax is axes[-1]))
        ax.set_title(col)
    fig.suptitle("Numeric feature distributions by target", y=1.03)
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "histograms.png", dpi=120,
                                    bbox_inches="tight"); plt.close(fig)

    # --- correlation heatmap ---
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation heatmap")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "correlation_heatmap.png", dpi=120); plt.close(fig)

    # --- numeric summary + missingness ---
    summary = df.describe().T
    summary["missing"] = df.isna().sum()
    summary.to_csv(config.FIGURES_DIR / "numeric_summary.csv")

    print(f"[eda] figures written to {config.FIGURES_DIR}")
    print(f"[eda] missing values per column:\n{df.isna().sum()[lambda s: s > 0]}")


if __name__ == "__main__":
    run()
    sys.exit(0)
