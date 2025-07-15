.PHONY: install test lint run-monitoring run-api run-training build-image push-image terraform-init terraform-apply terraform-destroy

IMAGE_NAME=gcr.io/plucky-haven-463121-j1/intrusion-api
TAG=latest

install:
	pip install pipenv
	pipenv install --deploy --ignore-pipfile

shell:
	pipenv shell

test:
	pytest tests/

lint:
	pre-commit run --all-files

run-monitoring:
	python monitoring/monitor.py

run-api:
	docker build -t intrusion-api .
	docker run --rm -p 8080:8080 intrusion-api

run-prefect:
	prefect server start

run-mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./models

run-training:
	python pipelines/training_flow.py

terraform-deploy:
	terraform -chdir=infra init
	terraform -chdir=infra plan
	terraform -chdir=infra apply -auto-approve

terraform-destroy:
	terraform -chdir=infra destroy -auto-approve
