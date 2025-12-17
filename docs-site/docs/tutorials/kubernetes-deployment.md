# Kubernetes Deployment

Deploy QRATUM to production Kubernetes clusters with high availability and auto-scaling.

## Prerequisites

- Kubernetes cluster (EKS, GKE, AKS, or local)
- kubectl configured
- Helm 3.x installed
- Docker for building images

## Quick Deployment

### Using Helm

```bash
# Add QRATUM Helm repository (if available)
helm repo add qratum https://charts.qratum.io
helm repo update

# Install QRATUM
helm install qratum qratum/qratum \
  --namespace qratum \
  --create-namespace \
  --set image.tag=latest
```

### Using kubectl

```bash
# Create namespace
kubectl create namespace qratum

# Apply manifests
kubectl apply -f infra/k8s/ -n qratum
```

## Architecture Overview

```mermaid
graph TB
    LB[Load Balancer] --> API1[API Pod 1]
    LB --> API2[API Pod 2]
    LB --> API3[API Pod 3]
    API1 --> Redis[Redis Cluster]
    API2 --> Redis
    API3 --> Redis
    API1 --> PG[(PostgreSQL)]
    API2 --> PG
    API3 --> PG
```

## Deployment Manifests

### Deployment

```yaml
# infra/k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qratum-api
  labels:
    app: qratum
    component: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: qratum
      component: api
  template:
    metadata:
      labels:
        app: qratum
        component: api
    spec:
      containers:
        - name: api
          image: qratum/api:latest
          ports:
            - containerPort: 8000
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
          env:
            - name: JAX_PLATFORM_NAME
              value: "cpu"
            - name: LOG_LEVEL
              value: "INFO"
```

### Service

```yaml
# infra/k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: qratum-api
  labels:
    app: qratum
spec:
  type: ClusterIP
  ports:
    - port: 80
      targetPort: 8000
      protocol: TCP
      name: http
  selector:
    app: qratum
    component: api
```

### Ingress

```yaml
# infra/k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: qratum-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - api.qratum.example.com
      secretName: qratum-tls
  rules:
    - host: api.qratum.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: qratum-api
                port:
                  number: 80
```

### Horizontal Pod Autoscaler

```yaml
# infra/k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: qratum-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: qratum-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

## Configuration

### ConfigMap

```yaml
# infra/k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: qratum-config
data:
  LOG_LEVEL: "INFO"
  JAX_PLATFORM_NAME: "cpu"
  MAX_WORKERS: "4"
  ENABLE_METRICS: "true"
```

### Secrets

```yaml
# infra/k8s/secrets.yaml (example - use sealed-secrets in production)
apiVersion: v1
kind: Secret
metadata:
  name: qratum-secrets
type: Opaque
stringData:
  DATABASE_URL: "postgresql://user:pass@postgres:5432/qratum"
  REDIS_URL: "redis://redis:6379"
```

## Monitoring

### Prometheus ServiceMonitor

```yaml
# infra/k8s/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: qratum-monitor
  labels:
    app: qratum
spec:
  selector:
    matchLabels:
      app: qratum
  endpoints:
    - port: http
      path: /metrics
      interval: 30s
```

## Cloud-Specific Guides

### Amazon EKS

```bash
# Create EKS cluster
eksctl create cluster \
  --name qratum-prod \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type t3.large \
  --nodes 3

# Configure kubectl
aws eks update-kubeconfig --name qratum-prod --region us-west-2

# Deploy
kubectl apply -f infra/k8s/
```

### Google GKE

```bash
# Create GKE cluster
gcloud container clusters create qratum-prod \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type e2-standard-4

# Get credentials
gcloud container clusters get-credentials qratum-prod --zone us-central1-a

# Deploy
kubectl apply -f infra/k8s/
```

### Azure AKS

```bash
# Create AKS cluster
az aks create \
  --resource-group qratum-rg \
  --name qratum-prod \
  --node-count 3 \
  --node-vm-size Standard_DS2_v2 \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group qratum-rg --name qratum-prod

# Deploy
kubectl apply -f infra/k8s/
```

## Health Checks

```bash
# Check deployment status
kubectl get deployments -n qratum

# Check pods
kubectl get pods -n qratum

# Check services
kubectl get services -n qratum

# View logs
kubectl logs -f deployment/qratum-api -n qratum

# Test API
kubectl port-forward svc/qratum-api 8000:80 -n qratum
curl http://localhost:8000/health
```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n qratum

# Check events
kubectl get events -n qratum --sort-by='.lastTimestamp'
```

### Service Unavailable

```bash
# Check endpoints
kubectl get endpoints qratum-api -n qratum

# Test DNS
kubectl run debug --image=busybox -it --rm -- nslookup qratum-api
```

## Next Steps

- [CI/CD Integration](cicd-integration.md) - Automated deployments
- [Performance Tuning](../advanced/performance-tuning.md) - Optimization
- [Compliance](../compliance/index.md) - Security requirements
