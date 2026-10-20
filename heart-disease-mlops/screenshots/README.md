# Screenshots to capture for the report

Drop PNGs here as you run the pipeline. The report references these filenames:

1. `mlflow_runs.png`        — MLflow UI, runs comparison table (`mlflow ui`)
2. `mlflow_metrics.png`     — MLflow UI, metrics/params of the best run
3. `github_actions.png`     — GitHub Actions, all CI jobs green
4. `docker_build.png`       — `docker build` success + `docker run` logs
5. `swagger_ui.png`         — http://localhost:8000/docs with a /predict response
6. `kubectl_get_all.png`    — `kubectl get pods,svc` showing Running pods
7. `grafana_dashboard.png`  — Grafana "Heart Disease API" dashboard with traffic
8. `prometheus_targets.png` — Prometheus, target UP (http://localhost:9090/targets)
