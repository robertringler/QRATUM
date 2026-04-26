# QRATUM AI Platform - Operations Runbook

## Service Architecture

### Components
- **Model Server** (port 8080): FastAPI-based ML model serving
- **Orchestrator** (port 8000): Request routing and load balancing
- **Vector DB** (port 19530): Milvus vector database for RAG
- **MLflow**: Model registry and experiment tracking

## Deployment

### Prerequisites
- Kubernetes cluster (1.24+)
- Helm 3.12+
- kubectl configured
- Docker registry access

### Deploy Model Server

```bash
# Install Helm chart
helm install model-server qratum/infra/helm/model-server \
  --set variant=gpu \
  --set image.tag=1.0.0 \
  --namespace qratum-ml \
  --create-namespace

# Verify deployment
kubectl get pods -n qratum-ml
kubectl logs -f -n qratum-ml -l app=model-server
```

### Deploy Vector Database

```bash
# Install vector DB with persistence
helm install vector-db qratum/infra/helm/vector-db \
  --set persistence.enabled=true \
  --set persistence.size=100Gi \
  --namespace qratum-ml

# Check status
kubectl get pvc -n qratum-ml
```

## Operations

### Health Checks

```bash
# Model server health
curl http://model-server:8080/health

# Orchestrator health
curl http://orchestrator:8000/health

# Vector DB health (requires port-forward)
kubectl port-forward svc/vector-db 19530:19530 -n qratum-ml
# Then check with vector DB client
```

### Scaling

```bash
# Scale model server replicas
kubectl scale deployment model-server --replicas=5 -n qratum-ml

# Enable autoscaling
kubectl autoscale deployment model-server \
  --min=2 --max=10 \
  --cpu-percent=70 \
  -n qratum-ml
```

### Model Updates

```bash
# Update model version
helm upgrade model-server qratum/infra/helm/model-server \
  --set image.tag=1.1.0 \
  --reuse-values

# Canary deployment via Argo
argo submit qratum/pipelines/argo/canary.yaml \
  -p model-version=1.1.0 \
  -p canary-percent=10
```

## Monitoring

### Metrics

- **Prometheus**: Scrapes `/metrics` endpoints
- **Grafana**: Dashboards for service health
- **Key metrics**:
  - Request rate (requests/sec)
  - Latency (p50, p95, p99)
  - Error rate (%)
  - GPU utilization (%)
  - Model inference time (ms)

### Alerts

```yaml
# Example Prometheus alert rule
- alert: HighErrorRate
  expr: rate(http_requests_total{status="500"}[5m]) > 0.05
  for: 5m
  annotations:
    summary: "High error rate detected"
```

## Troubleshooting

### Model Server Issues

```bash
# Check logs
kubectl logs -f deployment/model-server -n qratum-ml

# Check resource usage
kubectl top pods -n qratum-ml

# Restart pods
kubectl rollout restart deployment/model-server -n qratum-ml
```

### Vector DB Issues

```bash
# Check Milvus logs
kubectl logs -f deployment/vector-db -n qratum-ml

# Check storage
kubectl get pvc -n qratum-ml

# Rebuild index if corrupted
kubectl exec -it vector-db-0 -n qratum-ml -- \
  milvusctl index rebuild --collection=documents
```

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| OOM Kill | Pods restarting, memory errors | Increase memory limits |
| GPU Not Found | CUDA errors in logs | Check nodeSelector, GPU drivers |
| Slow Inference | High latency metrics | Scale up replicas, check GPU usage |
| Auth Failures | 401 errors | Check secret configuration |

## Security

### Secrets Management

```bash
# Create auth token secret
kubectl create secret generic model-server-secret \
  --from-literal=auth-token=YOUR_TOKEN_HERE \
  -n qratum-ml

# Rotate secrets
kubectl delete secret model-server-secret -n qratum-ml
kubectl create secret generic model-server-secret \
  --from-literal=auth-token=NEW_TOKEN \
  -n qratum-ml
kubectl rollout restart deployment/model-server -n qratum-ml
```

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: model-server-policy
spec:
  podSelector:
    matchLabels:
      app: model-server
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: orchestrator
    ports:
    - protocol: TCP
      port: 8080
```

## Backup & Recovery

### Vector DB Backup

```bash
# Create backup
kubectl exec -it vector-db-0 -n qratum-ml -- \
  milvusctl backup create --collection=all --output=/backup

# Copy backup to S3
kubectl cp vector-db-0:/backup ./backup -n qratum-ml
aws s3 cp ./backup s3://qratum-backups/$(date +%Y%m%d)/
```

### Model Registry Backup

```bash
# Backup MLflow artifacts
mlflow artifacts download --artifact-uri s3://mlflow/models \
  --dst-path /backup/models
```

## Performance Tuning

### Model Server

- Use GPU variant for compute-intensive models
- Enable batch processing for higher throughput
- Configure appropriate timeout values
- Use model quantization (INT8) for faster inference

### Vector DB

- Tune index parameters (nlist, nprobe)
- Use IVF_FLAT for balance, IVF_SQ8 for memory efficiency
- Enable auto-flush for real-time updates
- Shard collections for large datasets

## Contact

- **On-call**: Slack #qratum-oncall
- **Escalation**: ops@qratum.io
- **Documentation**: https://docs.qratum.io
