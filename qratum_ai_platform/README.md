# QRATUM AI Platform

Production-grade AI/ML platform with security, observability, and MLOps capabilities.

## Architecture

```
qratum/
├── services/              # Microservices
│   ├── model_server/      # FastAPI model serving
│   ├── orchestrator/      # Request routing
│   ├── rag/               # RAG connector & embedder
│   ├── safety/            # Policy engine
│   └── explainability/    # SHAP explainer
├── infra/helm/            # Kubernetes Helm charts
│   ├── model-server/      # Model server deployment
│   └── vector-db/         # Vector database
├── pipelines/argo/        # Argo workflows
│   ├── train.yaml         # Training pipeline
│   └── canary.yaml        # Canary deployment
├── mlflow/hooks/          # MLflow artifact hooks
├── ci/                    # CI/CD scripts
└── tests/                 # Test suite

```

## Features

- **Model Serving**: FastAPI-based model server with GPU/CPU support
- **RAG Pipeline**: Retrieval-Augmented Generation with vector DB
- **Safety**: Policy engine for output validation
- **Explainability**: SHAP-based model interpretability
- **MLOps**: Artifact signing, SBOM generation, Argo workflows
- **Security**: RBAC, audit logging, signed artifacts

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run model server
python qratum/services/model_server/app.py

# Run orchestrator
python qratum/services/orchestrator/app.py

# Test health endpoint
curl http://localhost:8080/health
```

### Kubernetes Deployment

```bash
# Deploy model server
helm install model-server qratum/infra/helm/model-server

# Deploy vector DB
helm install vector-db qratum/infra/helm/vector-db
```

## Testing

```bash
# Run unit tests
pytest qratum/tests/

# Run RAG integration tests
pytest qratum/tests/test_rag.py

# Run smoke tests
pytest -m smoke
```

## CI/CD

GitHub Actions pipeline includes:

- Linting and unit tests
- RAG integration tests
- Quantization smoke tests
- Secret scanning
- SBOM generation
- Artifact signing

## Security

- Bearer token authentication
- Input sanitization
- Audit logging
- Signed artifacts (SHA256)
- Network isolation by default

## Documentation

- [Runbook](docs/runbook.md) - Operations guide
- [Acceptance Criteria](docs/acceptance_criteria.md) - Testing checklist
- [Helm Charts](infra/helm/) - Deployment configuration

## License

Apache 2.0
