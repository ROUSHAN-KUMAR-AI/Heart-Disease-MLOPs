"""
Acquire the Heart Disease (UCI) dataset and write a raw CSV to data/raw/.

The function tries three strategies, in order, so the pipeline is reproducible
in every environment (your laptop, a CI runner, a teammate's machine):

  1. `ucimlrepo` package           -> cleanest, fully reproducible (needs internet)
  2. direct download from UCI URL  -> fallback if the package is unavailable
  3. deterministic synthetic data  -> offline / air-gapped fallback for CI & tests

Run:
    python -m src.data.download_data
    python -m src.data.download_data --synthetic   # force the offline fixture
"""
from __future__ import annotations

import argparse
import io
import sys

import numpy as np
import pandas as pd

from src import config


def _from_ucimlrepo() -> pd.DataFrame:
    """Strategy 1 — the official UCI helper package."""
    from ucimlrepo import fetch_ucirepo  # imported lazily so it isn't a hard dep

    repo = fetch_ucirepo(id=config.UCI_DATASET_ID)
    df = pd.concat([repo.data.features, repo.data.targets], axis=1)
    # The package already names the target column; normalise it to "num".
    target_col = repo.data.targets.columns[0]
    df = df.rename(columns={target_col: config.RAW_TARGET})
    return df[config.RAW_COLUMNS]


def _from_direct_url() -> pd.DataFrame:
    """Strategy 2 — raw headerless file straight from the UCI mirror."""
    import urllib.request

    with urllib.request.urlopen(config.UCI_DIRECT_URL, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), header=None, names=config.RAW_COLUMNS)
    return df


def _synthetic(n: int = 303, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """
    Strategy 3 — a *clearly synthetic* stand-in with the same schema and plausible
    ranges, so the whole pipeline (and the CI tests) can run with no internet.

    WARNING: this is NOT the real clinical data. Never report metrics produced
    on this fixture in your submission — run the real download first.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(29, 78, n)
    sex = rng.integers(0, 2, n)
    cp = rng.integers(0, 4, n)
    trestbps = rng.normal(131, 17, n).clip(90, 200).round()
    chol = rng.normal(246, 52, n).clip(120, 560).round()
    fbs = (rng.random(n) < 0.15).astype(int)
    restecg = rng.integers(0, 3, n)
    thalach = rng.normal(149, 23, n).clip(70, 205).round()
    exang = rng.integers(0, 2, n)
    oldpeak = rng.gamma(1.4, 0.7, n).clip(0, 6.2).round(1)
    slope = rng.integers(0, 3, n)
    ca = rng.integers(0, 4, n)
    thal = rng.choice([3, 6, 7], size=n, p=[0.55, 0.08, 0.37])

    # Build a latent risk score so the target is *learnable* (not pure noise).
    logit = (
        -4.0  # intercept chosen so the positive class sits near ~45-50%
        + 0.03 * (age - 54) + 0.9 * sex + 0.6 * cp - 0.025 * (thalach - 150)
        + 0.8 * exang + 0.7 * oldpeak + 0.9 * ca + 0.5 * (thal == 7)
    )
    prob = 1 / (1 + np.exp(-logit))
    target_sev = (rng.random(n) < prob).astype(int) * rng.integers(1, 5, n)

    # Inject a few "?" missing values into ca/thal, like the real file.
    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
        "num": target_sev,
    }).astype(object)
    miss_ca = rng.choice(n, size=4, replace=False)
    miss_thal = rng.choice(n, size=2, replace=False)
    df.loc[miss_ca, "ca"] = "?"
    df.loc[miss_thal, "thal"] = "?"
    return df[config.RAW_COLUMNS]


def download(force_synthetic: bool = False) -> pd.DataFrame:
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df, source = None, None
    if not force_synthetic:
        for name, fn in (("ucimlrepo", _from_ucimlrepo), ("direct-url", _from_direct_url)):
            try:
                df, source = fn(), name
                print(f"[download] fetched real data via {name}: {df.shape}")
                break
            except Exception as exc:  # noqa: BLE001 - we want a broad fallback
                print(f"[download] {name} failed ({exc!r}); trying next strategy")

    if df is None:
        df, source = _synthetic(), "SYNTHETIC"
        print(f"[download] !! using SYNTHETIC fixture: {df.shape} -- not for reporting")

    df.to_csv(config.RAW_DATA_PATH, index=False)
    print(f"[download] wrote {config.RAW_DATA_PATH} (source={source})")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the Heart Disease dataset")
    parser.add_argument("--synthetic", action="store_true",
                        help="skip the network and use the offline fixture")
    args = parser.parse_args()
    download(force_synthetic=args.synthetic)
    return 0


if __name__ == "__main__":
    sys.exit(main())
