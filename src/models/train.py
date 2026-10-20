"""
Train, tune, track and package the Heart Disease classifier.

Covers assignment tasks 2, 3 and 4:
  * two models (Logistic Regression + Random Forest) tuned with GridSearchCV
  * 5-fold cross-validation on the training set
  * test-set metrics: accuracy, precision, recall, F1, ROC-AUC
  * MLflow logging of params, metrics, plots and the model artifact
  * the winning model is saved as a single joblib Pipeline (preprocess + clf)
    so inference needs zero bespoke code -> full reproducibility

MLflow is imported lazily and guarded: if it is not installed the script still
runs and just skips tracking (handy for quick local smoke tests). For the real
submission, install it (it is in requirements.txt) so runs are logged.

Run:
    python -m src.models.train
Then inspect the runs with:
    mlflow ui            # opens http://127.0.0.1:5000
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend so plots render on CI runners
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src import config
from src.features.build_features import build_preprocessor, get_feature_names

# ---- optional MLflow -------------------------------------------------------
try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW = True
except ImportError:  # pragma: no cover
    _MLFLOW = False
    print("[train] mlflow not installed -> tracking disabled (install it for submission)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_split() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = pd.read_csv(config.TRAIN_DATA_PATH)
    test = pd.read_csv(config.TEST_DATA_PATH)
    X_train, y_train = train[config.FEATURE_COLUMNS], train[config.TARGET]
    X_test, y_test = test[config.FEATURE_COLUMNS], test[config.TARGET]
    return X_train, y_train, X_test, y_test


def _candidate_models() -> dict[str, tuple[object, dict]]:
    """Map name -> (estimator, param_grid). Add your own model to experiment."""
    return {
        "logreg": (
            LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
            config.LOGREG_PARAM_GRID,
        ),
        "random_forest": (
            RandomForestClassifier(random_state=config.RANDOM_STATE),
            config.RF_PARAM_GRID,
        ),
    }


def _evaluate(pipe: Pipeline, X_test, y_test) -> dict[str, float]:
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def _save_eval_plots(pipe: Pipeline, X_test, y_test, name: str) -> list:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix — {name}")
    p = config.FIGURES_DIR / f"confusion_matrix_{name}.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title(f"ROC curve — {name}")
    p = config.FIGURES_DIR / f"roc_curve_{name}.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    # Feature importance (tree models only)
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        names = get_feature_names(pipe.named_steps["preprocess"])
        imp = pd.Series(clf.feature_importances_, index=names).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(6, 5))
        imp.plot.barh(ax=ax)
        ax.set_title(f"Top feature importances — {name}")
        p = config.FIGURES_DIR / f"feature_importance_{name}.png"
        fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train() -> dict:
    X_train, y_train, X_test, y_test = _load_split()

    if _MLFLOW:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    results: dict[str, dict] = {}
    best_name, best_pipe, best_score = None, None, -np.inf

    parent_ctx = mlflow.start_run(run_name=f"train-{datetime.now():%Y%m%d-%H%M%S}") if _MLFLOW else None
    try:
        for name, (estimator, grid) in _candidate_models().items():
            print(f"\n[train] === {name} ===")
            pipe = Pipeline(steps=[
                ("preprocess", build_preprocessor()),
                ("clf", estimator),
            ])
            search = GridSearchCV(
                pipe, grid, scoring=config.SCORING, cv=config.CV_FOLDS, n_jobs=-1, refit=True,
            )
            search.fit(X_train, y_train)
            tuned = search.best_estimator_

            cv_score = search.best_score_
            test_metrics = _evaluate(tuned, X_test, y_test)
            plot_paths = _save_eval_plots(tuned, X_test, y_test, name)

            print(f"[train] best params: {search.best_params_}")
            print(f"[train] CV {config.SCORING}={cv_score:.4f}  "
                  f"test={ {k: round(v, 4) for k, v in test_metrics.items()} }")

            results[name] = {
                "best_params": search.best_params_,
                "cv_roc_auc": cv_score,
                **test_metrics,
            }

            # ---- MLflow: one nested run per candidate model ----
            if _MLFLOW:
                with mlflow.start_run(run_name=name, nested=True):
                    mlflow.log_params({k.replace("clf__", ""): v
                                       for k, v in search.best_params_.items()})
                    mlflow.log_param("model_family", name)
                    mlflow.log_param("cv_folds", config.CV_FOLDS)
                    mlflow.log_metric(f"cv_{config.SCORING}", cv_score)
                    for k, v in test_metrics.items():
                        mlflow.log_metric(f"test_{k}", v)
                    for pth in plot_paths:
                        mlflow.log_artifact(str(pth), artifact_path="plots")
                    mlflow.sklearn.log_model(tuned, name="model")

            # selection criterion: best cross-validated ROC-AUC (no test peeking)
            if cv_score > best_score:
                best_name, best_pipe, best_score = name, tuned, cv_score

        print(f"\n[train] winner = {best_name}  (CV {config.SCORING}={best_score:.4f})")
        if _MLFLOW:
            mlflow.log_param("winner", best_name)
            mlflow.log_metric("winner_cv_roc_auc", best_score)
    finally:
        if parent_ctx is not None:
            mlflow.end_run()

    _package(best_pipe, best_name, results[best_name], X_test, y_test)
    return results


def _package(pipe: Pipeline, name: str, metrics: dict, X_test, y_test) -> None:
    """Save the winning pipeline + metadata for the serving layer."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, config.MODEL_PATH)

    metadata = {
        "model_family": name,
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()
                    if isinstance(v, (int, float))},
        "best_params": metrics.get("best_params", {}),
        "feature_columns": config.FEATURE_COLUMNS,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_pipeline_steps": [s for s, _ in pipe.steps],
        "random_state": config.RANDOM_STATE,
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"[train] saved model -> {config.MODEL_PATH}")
    print(f"[train] saved metadata -> {config.METADATA_PATH}")


if __name__ == "__main__":
    train()
    sys.exit(0)
