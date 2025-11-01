.PHONY: dev_up build_images deploy destroy

dev_up:
python -m venv .venv && . .venv/bin/activate && pip install -r services/backend/requirements.txt && python services/backend/app.py

build_images:
docker build -t backend:local services/backend
docker build -t frontend:local services/frontend

deploy:
kubectl apply -f infra/k8s

destroy:
kubectl delete -f infra/k8s --ignore-not-found
