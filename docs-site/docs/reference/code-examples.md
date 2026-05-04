# Code Examples

Copy-paste ready code examples for common QRATUM tasks.

## Python Examples

### Basic VQE Simulation

```python
"""
VQE simulation for H₂ molecule ground state energy.
"""
from quasim.quantum.core import QuantumConfig
from quasim.quantum.vqe_molecule import MolecularVQE

# Configure backend
config = QuantumConfig(
    backend_type="simulator",
    shots=1024,
    seed=42
)

# Create VQE solver
vqe = MolecularVQE(config)

# Compute ground state
result = vqe.compute_h2_energy(
    bond_length=0.735,  # Angstroms
    basis="sto3g",
    max_iterations=100
)

print(f"Energy: {result.energy:.6f} Hartree")
print(f"Converged: {result.converged}")
```

### QAOA MaxCut Optimization

```python
"""
QAOA for MaxCut graph partitioning.
"""
from quasim.quantum.core import QuantumConfig
from quasim.quantum.qaoa_optimization import QAOA

# Configure
config = QuantumConfig(backend_type="simulator", shots=1024)
qaoa = QAOA(config, p_layers=3)

# Define graph
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]

# Solve
result = qaoa.solve_maxcut(edges, max_iterations=100)

print(f"Partition: {result.solution}")
print(f"Cut value: {abs(result.energy):.0f}")
print(f"Approximation: {result.approximation_ratio:.2%}")
```

### Potential Energy Surface Scan

```python
"""
Scan H₂ bond length to generate potential energy surface.
"""
import numpy as np
import matplotlib.pyplot as plt
from quasim.quantum.core import QuantumConfig
from quasim.quantum.vqe_molecule import MolecularVQE

config = QuantumConfig(backend_type="simulator", shots=1024, seed=42)
vqe = MolecularVQE(config)

# Scan bond lengths
bond_lengths = np.linspace(0.4, 2.5, 20)
energies = []

for r in bond_lengths:
    result = vqe.compute_h2_energy(bond_length=r, max_iterations=50)
    energies.append(result.energy)
    print(f"R = {r:.2f} Å, E = {result.energy:.4f} Ha")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(bond_lengths, energies, 'b-o')
plt.xlabel('Bond Length (Å)')
plt.ylabel('Energy (Hartree)')
plt.title('H₂ Potential Energy Surface')
plt.grid(True)
plt.savefig('h2_pes.png', dpi=150)
plt.show()
```

### REST API Client

```python
"""
Python client for QRATUM REST API.
"""
import requests
from typing import Optional

class QRATUMClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def health(self) -> dict:
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def run_kernel(self, seed: int = 42, scale: float = 1.0) -> dict:
        """Run quantum kernel computation."""
        response = requests.post(
            f"{self.base_url}/kernel",
            json={"seed": seed, "scale": scale}
        )
        response.raise_for_status()
        return response.json()
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics."""
        response = requests.get(f"{self.base_url}/metrics")
        response.raise_for_status()
        return response.text

# Usage
client = QRATUMClient()
print(client.health())
result = client.run_kernel(seed=42, scale=2.0)
print(f"Energy: {result['energy']}")
```

---

## Kubernetes YAML

### Basic Deployment

```yaml
# qratum-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qratum-api
  labels:
    app: qratum
spec:
  replicas: 3
  selector:
    matchLabels:
      app: qratum
  template:
    metadata:
      labels:
        app: qratum
    spec:
      containers:
        - name: api
          image: qratum/api:latest
          ports:
            - containerPort: 8000
          env:
            - name: JAX_PLATFORM_NAME
              value: "cpu"
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
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: qratum-api
spec:
  selector:
    app: qratum
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
```

### Horizontal Pod Autoscaler

```yaml
# qratum-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: qratum-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: qratum-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### Ingress with TLS

```yaml
# qratum-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: qratum-ingress
  annotations:
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

---

## Terraform HCL

### AWS EKS Cluster

```hcl
# main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    default = {
      instance_types = ["t3.large"]
      min_size       = 2
      max_size       = 10
      desired_size   = 3
    }
  }
}

# Variables
variable "aws_region" {
  default = "us-west-2"
}

variable "cluster_name" {
  default = "qratum-prod"
}

# Outputs
output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "cluster_name" {
  value = module.eks.cluster_name
}
```

### GCP GKE Cluster

```hcl
# gcp-main.tf
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_container_cluster" "qratum" {
  name     = var.cluster_name
  location = var.zone

  initial_node_count = 3

  node_config {
    machine_type = "e2-standard-4"
    disk_size_gb = 100

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}

variable "project_id" {}
variable "region" { default = "us-central1" }
variable "zone" { default = "us-central1-a" }
variable "cluster_name" { default = "qratum-prod" }
```

---

## Docker Compose

### Development Stack

```yaml
# docker-compose.dev.yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - .:/app
      - /app/.venv
    environment:
      - JAX_PLATFORM_NAME=cpu
      - QRATUM_LOG_LEVEL=DEBUG
    command: python -m uvicorn app:app --reload --host 0.0.0.0

  frontend:
    build: ./autonomous_systems_platform/services/frontend
    ports:
      - "8080:80"
    volumes:
      - ./autonomous_systems_platform/services/frontend:/usr/share/nginx/html:ro

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

### Production Stack

```yaml
# docker-compose.prod.yaml
version: '3.8'

services:
  backend:
    image: qratum/api:${VERSION:-latest}
    ports:
      - "8000:8000"
    environment:
      - JAX_PLATFORM_NAME=cpu
      - QRATUM_LOG_LEVEL=INFO
      - QRATUM_API_WORKERS=4
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - backend
```

---

## GitHub Actions

### CI Pipeline

```yaml
# .github/workflows/ci.yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: pip
      
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: pytest tests/ --cov

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t qratum:${{ github.sha }} .
```
