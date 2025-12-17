# Configuration

Configuration options for QRATUM components.

## Configuration Files

QRATUM uses YAML configuration files:

```
~/.qratum/
├── config.yaml          # Main configuration
├── backends.yaml        # Backend settings
└── credentials.yaml     # API tokens (gitignored)
```

## Main Configuration

`~/.qratum/config.yaml`:

```yaml
# QRATUM Configuration
version: "2.0"

# Logging settings
logging:
  level: INFO              # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: ~/.qratum/logs/qratum.log

# Default quantum settings
quantum:
  backend: simulator       # simulator, ibmq
  shots: 1024
  seed: null              # null for random
  max_qubits: 20

# Optimization settings
optimization:
  default_optimizer: COBYLA
  max_iterations: 100
  tolerance: 1.0e-6

# Hardware settings
hardware:
  prefer_gpu: true
  max_memory_gb: null     # null for auto
  threads: null           # null for auto
```

## Backend Configuration

`~/.qratum/backends.yaml`:

```yaml
# Backend Configuration
backends:
  simulator:
    type: qiskit_aer
    options:
      method: statevector
      precision: double
      max_parallel_threads: 0  # 0 for auto

  ibmq:
    type: ibmq
    hub: ibm-q
    group: open
    project: main
    # Token from credentials.yaml or environment

  gpu:
    type: cuquantum
    device: 0
    precision: single
```

## Credentials

`~/.qratum/credentials.yaml` (DO NOT commit):

```yaml
# Credentials - Keep this file secure!
ibmq:
  token: "YOUR_IBM_QUANTUM_TOKEN"

aws:
  access_key_id: "YOUR_AWS_ACCESS_KEY"
  secret_access_key: "YOUR_AWS_SECRET_KEY"
  region: us-west-2
```

## Environment Variables

Override any configuration with environment variables:

### Logging

| Variable | Description | Default |
|----------|-------------|---------|
| `QRATUM_LOG_LEVEL` | Log level | `INFO` |
| `QRATUM_LOG_FILE` | Log file path | `None` |
| `QRATUM_LOG_FORMAT` | Log format | See above |

### Quantum

| Variable | Description | Default |
|----------|-------------|---------|
| `QRATUM_BACKEND` | Quantum backend | `simulator` |
| `QRATUM_SHOTS` | Measurement shots | `1024` |
| `QRATUM_SEED` | Random seed | `None` |
| `QRATUM_MAX_QUBITS` | Max qubit limit | `20` |

### Hardware

| Variable | Description | Default |
|----------|-------------|---------|
| `JAX_PLATFORM_NAME` | JAX platform | `cpu` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | All |
| `OMP_NUM_THREADS` | OpenMP threads | Auto |
| `MKL_NUM_THREADS` | MKL threads | Auto |

### API

| Variable | Description | Default |
|----------|-------------|---------|
| `QRATUM_API_HOST` | Bind host | `0.0.0.0` |
| `QRATUM_API_PORT` | Bind port | `8000` |
| `QRATUM_API_WORKERS` | Worker count | `4` |
| `QRATUM_API_TIMEOUT` | Request timeout | `300` |

### Credentials

| Variable | Description | Default |
|----------|-------------|---------|
| `IBMQ_TOKEN` | IBM Quantum token | `None` |
| `AWS_ACCESS_KEY_ID` | AWS access key | `None` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | `None` |
| `AWS_REGION` | AWS region | `us-west-2` |

## Docker Compose Configuration

`docker-compose.yml`:

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - JAX_PLATFORM_NAME=cpu
      - QRATUM_LOG_LEVEL=INFO
      - QRATUM_API_WORKERS=4
    volumes:
      - ./configs:/app/configs:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'

  frontend:
    build:
      context: ./autonomous_systems_platform/services/frontend
    ports:
      - "8080:80"
    depends_on:
      - backend

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

## Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: qratum-config
data:
  config.yaml: |
    version: "2.0"
    logging:
      level: INFO
    quantum:
      backend: simulator
      shots: 1024
    optimization:
      default_optimizer: COBYLA
      max_iterations: 100
```

## Kubernetes Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: qratum-credentials
type: Opaque
stringData:
  IBMQ_TOKEN: "your-token-here"
  DATABASE_URL: "postgresql://user:pass@host:5432/db"
```

## Terraform Variables

`terraform.tfvars`:

```hcl
# QRATUM Infrastructure Variables

# AWS
aws_region     = "us-west-2"
environment    = "production"
cluster_name   = "qratum-prod"

# Kubernetes
k8s_namespace  = "qratum"
replica_count  = 3
min_replicas   = 2
max_replicas   = 20

# Resources
cpu_request    = "250m"
cpu_limit      = "1000m"
memory_request = "512Mi"
memory_limit   = "2Gi"

# Networking
enable_tls     = true
domain_name    = "api.qratum.example.com"
```

## Validation

Validate configuration:

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Validate with schema (if available)
python -m quasim.config.validate ~/.qratum/config.yaml
```

## Precedence

Configuration is loaded in order (later overrides earlier):

1. Built-in defaults
2. System config (`/etc/qratum/config.yaml`)
3. User config (`~/.qratum/config.yaml`)
4. Project config (`./qratum.yaml`)
5. Environment variables
6. Command-line arguments
