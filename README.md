# Heart Disease Risk — End-to-End MLOps

Predict the risk of heart disease from patient health data and serve it as a
cloud-ready, monitored API. Built for the BITS Pilani MLOps assignment
(AIMLCZG523) and covering the full lifecycle: data → EDA → modelling →
experiment tracking → packaging → CI/CD → containerisation → Kubernetes →
monitoring.

> **Dataset:** UCI Heart Disease (Cleveland). The download script fetches the
> real data; an offline synthetic fixture lets tests and CI run with no network.
> **Never report metrics produced on the synthetic fixture** — run the real
> download first.

## Architecture

![architecture](reports/figures/architecture.png)

## Quickstart

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. data -> EDA -> train (writes models/model.joblib + metrics + plots)
python -m src.data.download_data
python -m src.data.preprocess
python -m src.eda
python -m src.models.train

# 3. inspect experiments
mlflow ui          # http://127.0.0.1:5000

# 4. serve
uvicorn src.api.main:app --reload       # http://localhost:8000/docs
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json

# 5. tests + lint
pytest
ruff check src tests
```

Or just: `make pipeline` then `make serve`. Run `make help` for all targets.

## Container + monitoring stack

```bash
docker compose up --build
# API        -> http://localhost:8000/docs
# Prometheus -> http://localhost:9090
# Grafana    -> http://localhost:3000  (admin / admin), dashboard "Heart Disease API"
```

## Kubernetes (Minikube example)

```bash
minikube start
eval $(minikube docker-env)            # build into minikube's docker
docker build -t heart-disease-api:latest .
kubectl apply -f k8s/                  # or: helm install heart helm/heart-disease-api
kubectl get pods,svc
minikube service heart-disease-api --url
```

## Project layout

```
src/
  config.py               central paths, feature schema, hyper-params  <- personalise
  data/download_data.py   UCI download w/ offline synthetic fallback
  data/preprocess.py      clean, binarise target, stratified split
  features/build_features.py  leakage-safe ColumnTransformer
  eda.py                  histograms, heatmap, class balance
  models/train.py         GridSearchCV + MLflow + evaluation + packaging
  models/predict.py       load pipeline, score records (shared by API + CLI)
  api/main.py             FastAPI /predict /health /metrics + JSON logging
notebooks/                01 EDA · 02 training · 03 inference
tests/                    pytest: data, features, model, api
.github/workflows/        CI/CD: lint -> test -> train -> docker build+smoke
k8s/ , helm/              deployment manifests
monitoring/               Prometheus + Grafana provisioning
reports/                  report + figures
```

## Design decisions (short version)

- **One sklearn `Pipeline` saved as the model.** Preprocessing travels with the
  estimator, so inference needs zero duplicated transform code — true reproducibility.
- **Transforms fitted inside CV**, never on the full set → no train/test leakage.
- **Model selected on cross-validated ROC-AUC**, not test metrics (no peeking).
- **MLflow** logs params, metrics, plots and the model for every run.
- **Prometheus client built into the app**; Grafana dashboard provisioned as code.
- **CI fails on any lint/test/train error** and uploads artifacts per run.

See `reports/report.md` (and the PDF/DOCX) for the full write-up.
