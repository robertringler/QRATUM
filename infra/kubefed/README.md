# KubeFed v2 Multi-Region Configuration

This directory contains Kubernetes Federation v2 (KubeFed) manifests for deploying QuASIM across multiple regions.

## Prerequisites

1. **Install KubeFed v2**:

```bash
helm repo add kubefed-charts https://raw.githubusercontent.com/kubernetes-sigs/kubefed/master/charts
helm install kubefed kubefed-charts/kubefed --namespace kube-federation-system --create-namespace
```

1. **Join clusters to federation**:

```bash
kubefedctl join us-west-2 --cluster-context us-west-2-context --host-cluster-context host-cluster
kubefedctl join us-east-1 --cluster-context us-east-1-context --host-cluster-context host-cluster
kubefedctl join eu-west-1 --cluster-context eu-west-1-context --host-cluster-context host-cluster
kubefedctl join ap-southeast-1 --cluster-context ap-southeast-1-context --host-cluster-context host-cluster
```

## Deploy Federated Resources

### 1. Create namespace across all clusters

```bash
kubectl apply -f federated-namespace.yaml
```

### 2. Deploy API services

```bash
kubectl apply -f federated-deployment.yaml
kubectl apply -f federated-service.yaml
```

### 3. Verify deployment

```bash
# Check federated deployment status
kubectl get federateddeployment -n quasim-platform

# Check deployment in each cluster
for cluster in us-west-2 us-east-1 eu-west-1 ap-southeast-1; do
  echo "=== Cluster: $cluster ==="
  kubectl --context $cluster get deployments -n quasim-platform
done
```

## Architecture

QuASIM uses KubeFed to distribute workloads across multiple regions:

- **us-west-2**: Primary region (5 API replicas, 6 GPU workers)
- **us-east-1**: Secondary region (3 API replicas, 4 GPU workers)  
- **eu-west-1**: European region (3 API replicas, 4 GPU workers)
- **ap-southeast-1**: Asia Pacific region (2 API replicas, 2 GPU workers)

### Traffic Routing

Global traffic is routed using:

- AWS Route53 with latency-based routing
- Kubernetes Ingress with regional endpoints
- Session affinity for stateful workloads

### Data Locality

- Ray clusters are deployed per-region
- Data is replicated across regions using object storage (S3)
- Inter-region communication uses VPC peering

## Scaling

### Manual Scaling

```bash
# Scale API replicas in specific region
kubectl patch federateddeployment quasim-api -n quasim-platform \
  --type='json' \
  -p='[{"op": "replace", "path": "/spec/overrides/0/clusterOverrides/0/value", "value": 10}]'
```

### Auto-scaling

Each region has HPA configured. See `../helm/quasim-platform/values.yaml` for autoscaling configuration.

## Monitoring

Federated monitoring is available via:

- Prometheus federation across regions
- Grafana dashboards with multi-cluster view
- Thanos for long-term metrics storage

## Disaster Recovery

In case of regional failure:

```bash
# Remove failed cluster from federation
kubefedctl unjoin <cluster-name> --cluster-context <context> --host-cluster-context host-cluster

# Traffic will automatically route to healthy regions
```
