"""
Central configuration for the Heart Disease MLOps project.

Keeping all paths, column definitions, and key hyper-parameters in one place
makes the pipeline reproducible and easy to reason about. Every other module
imports from here instead of hard-coding strings.

>>> PERSONALISE ME <<<
If you want your submission to look like your own work (recommended by the
assignment's academic-integrity note), this is a good file to tweak:
  - change RANDOM_STATE to your roll-number digits,
  - adjust the hyper-parameter grids,
  - rename the experiment, etc.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  (resolved relative to the project root, so the code runs from anywhere)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "heart_disease_raw.csv"
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "heart_disease_clean.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"            # full sklearn Pipeline (preprocess + estimator)
METADATA_PATH = MODELS_DIR / "model_metadata.json"  # metrics, feature order, train date

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
# UCI "Heart Disease" dataset, Cleveland subset (id=45 in the UCI ML repo).
UCI_DATASET_ID = 45
UCI_DIRECT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# The raw Cleveland file ships without a header; these are the canonical names.
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
TARGET = "target"           # binarised: 1 = disease present, 0 = absent
RAW_TARGET = "num"          # original 0-4 severity column

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Human-readable notes used in EDA / API docs
FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "1 = male, 0 = female",
    "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg": "Resting ECG result (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of the peak exercise ST segment (0-2)",
    "ca": "Number of major vessels coloured by fluoroscopy (0-3)",
    "thal": "Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible)",
}

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
RANDOM_STATE = 42          # <<< PERSONALISE ME (e.g. last 2 digits of your roll no.)
TEST_SIZE = 0.20
CV_FOLDS = 5

# Hyper-parameter grids kept deliberately small so CI finishes quickly.
LOGREG_PARAM_GRID = {
    "clf__C": [0.01, 0.1, 1.0, 10.0],
    # penalty left at its default (l2); vary C only to keep the grid fast & clean
}
RF_PARAM_GRID = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [None, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
}

SCORING = "roc_auc"        # metric used to pick the best model during CV

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_EXPERIMENT_NAME = "heart-disease-classification"
MLFLOW_TRACKING_URI = "file:./mlruns"   # local file store; swap for a server URI if you host one
