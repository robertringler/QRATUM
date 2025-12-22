# Acceptance Criteria - QRATUM AI Platform

## Infrastructure & Deployment

### Helm Charts

- [ ] Model-server chart renders without errors: `helm template`
- [ ] Vector-DB chart renders without errors: `helm template`
- [ ] Charts support both CPU and GPU variants
- [ ] Persistence is enabled and configurable
- [ ] Resource limits are defined
- [ ] Security contexts are configured (runAsNonRoot)
- [ ] Health probes are defined

### Kubernetes

- [ ] Deployments create pods successfully
- [ ] Services are accessible within cluster
- [ ] PVCs are created and bound
- [ ] Autoscaling works (HPA)
- [ ] Pod anti-affinity configured for HA

## Services

### Model Server

- [ ] Health endpoint returns `{"status": "ok"}`
- [ ] Prediction endpoint requires authentication
- [ ] Input validation rejects invalid requests
- [ ] Returns predictions with confidence scores
- [ ] Logs requests and responses
- [ ] Handles errors gracefully (4xx, 5xx)

### Orchestrator

- [ ] Routes requests to correct model-server
- [ ] Falls back to default route if intent unknown
- [ ] Handles model-server unavailability (503)
- [ ] Returns proper error codes
- [ ] Lists available routes via `/routes` endpoint

### RAG Pipeline

- [ ] Query returns top-k passages
- [ ] Embeddings are consistent (same input → same output)
- [ ] Document indexing validates format
- [ ] Retrieval scores are descending
- [ ] Handles empty queries gracefully

### Safety Engine

- [ ] Blocks content matching blocked patterns
- [ ] Flags content matching flagged patterns
- [ ] Sanitizes blocked content ([REDACTED])
- [ ] Returns violation details
- [ ] Supports custom patterns

### Explainability

- [ ] Returns feature importances for predictions
- [ ] Explanations are ordered by importance
- [ ] Supports text and tabular inputs
- [ ] Batch processing works
- [ ] Importance values sum to ~1.0

## Testing

### Unit Tests

- [ ] All unit tests pass: `pytest qratum/tests/`
- [ ] Code coverage >80%
- [ ] No test failures in CI
- [ ] Tests run in <5 minutes

### Integration Tests

- [ ] RAG integration tests pass
- [ ] Orchestrator routing test passes
- [ ] Model server responds to requests
- [ ] Vector DB connectivity works

### Smoke Tests

- [ ] Quantization smoke test passes
- [ ] Services start without errors
- [ ] Basic end-to-end flow works

## CI/CD

### GitHub Actions

- [ ] Lint job passes (ruff)
- [ ] Unit tests job passes
- [ ] RAG integration job passes
- [ ] Quantize smoke job passes
- [ ] Secret scan job completes
- [ ] SBOM generation succeeds
- [ ] Cosign verification step runs
- [ ] Helm validation passes

### Artifact Signing

- [ ] `sign_artifact.sh` creates `.sig` file
- [ ] Signature verification succeeds
- [ ] Metadata JSON is generated
- [ ] SHA256 checksums match

### SBOM

- [ ] SBOM generation script runs
- [ ] Output is valid JSON
- [ ] Lists Python dependencies
- [ ] Includes metadata timestamp

## MLOps

### Argo Workflows

- [ ] Training workflow validates: `argo lint`
- [ ] Canary workflow validates: `argo lint`
- [ ] Workflows can be submitted
- [ ] GPU resources are requested
- [ ] Steps execute in order

### Model Registry

- [ ] Models can be logged to MLflow
- [ ] Artifacts are signed on registration
- [ ] Versioning works correctly
- [ ] Models can be retrieved

## Security

### Authentication

- [ ] Bearer token authentication works
- [ ] Invalid tokens return 401
- [ ] Tokens can be rotated

### Input Validation

- [ ] SQL injection attempts blocked
- [ ] XSS patterns blocked
- [ ] Oversized inputs rejected (>10KB)
- [ ] Invalid JSON rejected

### Network

- [ ] Services use TLS (in production)
- [ ] Network policies restrict traffic
- [ ] Secrets not in logs
- [ ] Secrets not in code

## Performance

### Latency

- [ ] Health endpoint <10ms
- [ ] Prediction endpoint <200ms (CPU)
- [ ] Prediction endpoint <50ms (GPU)
- [ ] RAG retrieval <100ms

### Throughput

- [ ] Model server handles >10 req/sec
- [ ] Orchestrator handles >50 req/sec
- [ ] Vector DB handles >100 queries/sec

### Resource Usage

- [ ] Model server: <2GB memory (CPU)
- [ ] Model server: <8GB memory (GPU)
- [ ] Orchestrator: <512MB memory
- [ ] No memory leaks over 24h

## Documentation

- [ ] README is comprehensive
- [ ] Runbook covers all operations
- [ ] API docs are generated
- [ ] Examples run successfully
- [ ] Architecture diagrams present

## Observability

### Logging

- [ ] Structured logs (JSON)
- [ ] Log levels configurable
- [ ] No sensitive data in logs
- [ ] Errors include stack traces

### Metrics

- [ ] Prometheus endpoints expose metrics
- [ ] Request count metrics exist
- [ ] Latency histograms exist
- [ ] Error rate metrics exist
- [ ] GPU utilization tracked (if applicable)

### Tracing

- [ ] Requests have trace IDs
- [ ] Spans are created for operations
- [ ] Trace context propagated

## Compliance

- [ ] No hardcoded secrets
- [ ] All images from approved registries
- [ ] Security contexts configured
- [ ] Resource limits enforced
- [ ] Audit logs enabled

## Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Engineer | | | |
| QA | | | |
| Security | | | |
| Ops | | | |

## Notes

- All items must be checked before production deployment
- Failed items must have mitigation plans
- Re-test after any code changes
- Update this checklist as requirements evolve
