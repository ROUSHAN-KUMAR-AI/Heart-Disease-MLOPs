# Convenience targets. Run `make help` to list them.
.PHONY: help setup data train eda test lint serve docker-build docker-run compose-up clean pipeline

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:         ## install python dependencies
	pip install -r requirements.txt

data:          ## download + preprocess the dataset
	python -m src.data.download_data
	python -m src.data.preprocess

eda:           ## generate EDA figures
	python -m src.eda

train:         ## train + tune + track models, package the winner
	python -m src.models.train

test:          ## run the unit tests
	pytest

lint:          ## static analysis
	ruff check src tests

serve:         ## run the API locally with autoreload
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:  ## build the serving image
	docker build -t heart-disease-api:latest .

docker-run:    ## run the serving image
	docker run --rm -p 8000:8000 heart-disease-api:latest

compose-up:    ## API + Prometheus + Grafana
	docker compose up --build

pipeline: data eda train test ## full local pipeline

clean:         ## remove generated artifacts
	rm -rf mlruns models/*.joblib models/*.json reports/figures/*.png data/raw/*.csv data/processed/*.csv
