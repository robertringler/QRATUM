# QRATUM Full Stack Deployment Guide

## Overview

This guide covers deploying the complete QRATUM stack including:
- QRADLE deterministic execution engine
- QRATUM Platform with 14 verticals
- QRATUM-ASI orchestration layer
- Monitoring and audit infrastructure

## Prerequisites

### Hardware Requirements

**Minimum (Development):**
- 16 GB RAM
- 4-core CPU
- 100 GB storage
- Linux/macOS/Windows (WSL2)

**Recommended (Production):**
- 64 GB RAM (128 GB for ASI workloads)
- 16+ core CPU (32+ recommended)
- 500 GB SSD storage (NVMe preferred)
- GPU optional (NVIDIA A100/H100 for inference)
- 10 Gbps network

**Air-Gapped Deployment:**
- Additional 200 GB for offline dependencies
- Removable media for updates
- Internal mirror for package repositories

### Software Requirements

- Docker 24.0+ and Docker Compose 2.20+
- Python 3.11+
- Git
- OpenSSL (for certificate generation)

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/robertringler/QRATUM.git
cd QRATUM
```

### 2. Environment Configuration

Create `.env` file:

```bash
# QRATUM Configuration
QRATUM_VERSION=1.0.0
QRATUM_ENV=production
QRATUM_LOG_LEVEL=INFO

# Security - REQUIRED: Set a strong password for Grafana
GRAFANA_PASSWORD=<secure-password>

# Persistence
MERKLE_CHAIN_PERSISTENCE=true

# ASI Configuration
ASI_MODE=simulation
CRSI_ENABLED=false
```

**Important:** The `GRAFANA_PASSWORD` is required and must be a strong password. Never use default passwords in production.

### 3. Start Services

```bash
# Build images
docker-compose -f docker-compose.production.yml build

# Start all services
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps
```

### 4. Verify Deployment

```bash
# Health checks
curl http://localhost:8001/health  # QRADLE
curl http://localhost:8002/health  # Platform
curl http://localhost:8003/health  # ASI

# Grafana dashboard
open http://localhost:3000
```

---

## Detailed Deployment

### Step 1: Pre-Deployment Checks

```bash
# Check Docker
docker --version
docker-compose --version

# Check resources
free -h  # RAM
df -h    # Disk space
nproc    # CPU cores

# Check ports availability
netstat -tuln | grep -E '8001|8002|8003|9090|3000|24224'
```

### Step 2: Certificate Generation (Production)

```bash
# Generate SSL certificates
mkdir -p certs
cd certs

# Self-signed certificate (for internal use)
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout qratum-key.pem -out qratum-cert.pem \
  -days 365 -subj "/CN=qratum.local"

# Or use Let's Encrypt for public endpoints
certbot certonly --standalone -d api.qratum.io
```

### Step 3: Configuration Files

#### Prometheus Configuration

Create `configs/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'qradle'
    static_configs:
      - targets: ['qradle:8000']
  
  - job_name: 'platform'
    static_configs:
      - targets: ['qratum-platform:8000']
  
  - job_name: 'asi'
    static_configs:
      - targets: ['qratum-asi:8000']
```

#### Fluentd Configuration

Create `configs/fluentd.conf`:

```xml
<source>
  @type forward
  port 24224
  bind 0.0.0.0
</source>

<match qratum.**>
  @type file
  path /fluentd/log/qratum
  append true
  <buffer>
    flush_interval 10s
  </buffer>
  <format>
    @type json
  </format>
</match>
```

### Step 4: Start Services

```bash
# Build production images
docker-compose -f docker-compose.production.yml build \
  --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
  --build-arg VERSION=1.0.0

# Start infrastructure services first
docker-compose -f docker-compose.production.yml up -d \
  audit-logger prometheus grafana

# Wait for infrastructure
sleep 10

# Start QRATUM services
docker-compose -f docker-compose.production.yml up -d \
  qradle qratum-platform qratum-asi

# Check logs
docker-compose -f docker-compose.production.yml logs -f
```

### Step 5: Initial Configuration

```bash
# Create initial checkpoint
curl -X POST http://localhost:8001/api/v1/system/checkpoint \
  -H "Content-Type: application/json" \
  -d '{"description": "Initial deployment checkpoint"}'

# Verify integrity
curl http://localhost:8001/api/v1/system/proof
```

---

## Air-Gapped Deployment

### 1. Prepare Offline Package

On internet-connected machine:

```bash
# Pull all images
docker-compose -f docker-compose.production.yml pull

# Save images
docker save -o qratum-images.tar \
  python:3.11-slim \
  fluent/fluentd:v1.16-1 \
  prom/prometheus:v2.48.0 \
  grafana/grafana:10.2.2

# Package application
tar -czf qratum-deployment.tar.gz \
  qradle/ qratum_platform/ qratum_asi/ verticals/ \
  docker-compose.production.yml Dockerfile.production \
  configs/ requirements.txt requirements-prod.txt
```

### 2. Transfer to Air-Gapped Environment

```bash
# Copy packages via removable media
# - qratum-images.tar
# - qratum-deployment.tar.gz
# - Python wheels (pip download -r requirements-prod.txt -d ./wheels)
```

### 3. Deploy Offline

```bash
# Load Docker images
docker load -i qratum-images.tar

# Extract application
tar -xzf qratum-deployment.tar.gz

# Install Python dependencies
pip install --no-index --find-links=./wheels -r requirements-prod.txt

# Start services
docker-compose -f docker-compose.production.yml up -d
```

---

## High Availability Setup

### Multi-Node Deployment

```yaml
# docker-compose.ha.yml
version: '3.8'

services:
  qradle:
    deploy:
      replicas: 3
      placement:
        constraints:
          - node.role == worker
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

  # Load balancer
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - qradle
      - qratum-platform
```

### Database Persistence

For production, use external PostgreSQL:

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: qratum
      POSTGRES_USER: qratum
      POSTGRES_PASSWORD: <secure-password>
    volumes:
      - postgres-data:/var/lib/postgresql/data
    
  qradle:
    environment:
      - DATABASE_URL=postgresql://qratum:<password>@postgres:5432/qratum
```

---

## Monitoring & Observability

### Grafana Dashboards

Access Grafana at http://localhost:3000 using username `admin` and the password you set in the `GRAFANA_PASSWORD` environment variable.

**Security Note:** Never run Grafana with default credentials. Always set a strong password via the `GRAFANA_PASSWORD` environment variable before starting the services.

**Import dashboards:**
1. QRADLE Operations Dashboard
2. Platform Verticals Dashboard
3. ASI Orchestrator Dashboard
4. System Health Dashboard

### Prometheus Queries

```promql
# Request rate
rate(qratum_requests_total[5m])

# Error rate
rate(qratum_errors_total[5m]) / rate(qratum_requests_total[5m])

# Contract execution time
histogram_quantile(0.95, qratum_execution_duration_seconds)

# Merkle chain length
qratum_merkle_chain_length

# Active checkpoints
qratum_checkpoints_total
```

### Log Analysis

```bash
# View all logs
docker-compose -f docker-compose.production.yml logs

# Filter by service
docker-compose -f docker-compose.production.yml logs qradle

# Follow logs
docker-compose -f docker-compose.production.yml logs -f --tail=100

# Access audit logs
ls -lh audit-logs/
```

---

## Backup & Recovery

### Create Backup

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR=/backups/qratum-$DATE

mkdir -p $BACKUP_DIR

# Backup volumes
docker run --rm \
  -v qradle-data:/data \
  -v $BACKUP_DIR:/backup \
  alpine tar czf /backup/qradle-data.tar.gz -C /data .

docker run --rm \
  -v qradle-audit:/data \
  -v $BACKUP_DIR:/backup \
  alpine tar czf /backup/audit-logs.tar.gz -C /data .

# Backup configurations
cp -r configs $BACKUP_DIR/
cp docker-compose.production.yml $BACKUP_DIR/
cp .env $BACKUP_DIR/

echo "Backup completed: $BACKUP_DIR"
```

### Restore from Backup

```bash
#!/bin/bash
# restore.sh

BACKUP_DIR=$1

# Stop services
docker-compose -f docker-compose.production.yml down

# Restore volumes
docker run --rm \
  -v qradle-data:/data \
  -v $BACKUP_DIR:/backup \
  alpine tar xzf /backup/qradle-data.tar.gz -C /data

docker run --rm \
  -v qradle-audit:/data \
  -v $BACKUP_DIR:/backup \
  alpine tar xzf /backup/audit-logs.tar.gz -C /data

# Restore configurations
cp -r $BACKUP_DIR/configs ./
cp $BACKUP_DIR/docker-compose.production.yml ./
cp $BACKUP_DIR/.env ./

# Restart services
docker-compose -f docker-compose.production.yml up -d

echo "Restore completed"
```

---

## Security Hardening

### 1. Network Security

```bash
# Configure firewall
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp   # SSH
ufw allow 443/tcp  # HTTPS
ufw enable
```

### 2. Container Security

Already implemented in Dockerfile.production:
- Non-root user
- Read-only filesystem
- No new privileges
- Minimal base image
- Security scanning

### 3. Secrets Management

Use Docker secrets for sensitive data:

```yaml
secrets:
  db_password:
    file: ./secrets/db_password.txt

services:
  qradle:
    secrets:
      - db_password
    environment:
      DB_PASSWORD_FILE: /run/secrets/db_password
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose -f docker-compose.production.yml logs qradle

# Check resources
docker stats

# Verify configuration
docker-compose -f docker-compose.production.yml config
```

### Performance Issues

```bash
# Check CPU usage
docker stats --no-stream

# Check disk I/O
iostat -x 1

# Check memory
free -m

# Check network
netstat -i
```

### Integrity Verification Failed

```bash
# Get system proof
curl http://localhost:8001/api/v1/system/proof

# Check Merkle chain
curl http://localhost:8001/api/v1/audit/trail | jq .

# Rollback to checkpoint
curl -X POST http://localhost:8001/api/v1/system/rollback \
  -d '{"checkpoint_id": "ckpt_abc123"}'
```

---

## Maintenance

### Updates

```bash
# Pull latest images
docker-compose -f docker-compose.production.yml pull

# Create checkpoint before update
curl -X POST http://localhost:8001/api/v1/system/checkpoint \
  -d '{"description": "Pre-update checkpoint"}'

# Rolling update
docker-compose -f docker-compose.production.yml up -d --no-deps qradle

# Verify
curl http://localhost:8001/health
```

### Log Rotation

```bash
# Configure Docker log rotation
cat > /etc/docker/daemon.json <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF

systemctl restart docker
```

---

## Production Checklist

- [ ] Hardware requirements met
- [ ] SSL certificates configured
- [ ] Firewall rules configured
- [ ] Secrets properly secured
- [ ] Monitoring dashboards configured
- [ ] Backup system tested
- [ ] Disaster recovery plan documented
- [ ] Load testing completed
- [ ] Security scan passed
- [ ] Audit logging verified
- [ ] Checkpoint system tested
- [ ] Rollback procedure verified
- [ ] Team trained on operations
- [ ] Documentation updated

---

## Support

- Documentation: https://docs.qratum.io
- GitHub Issues: https://github.com/robertringler/QRATUM/issues
- Email: support@qratum.io
- Emergency: emergency@qratum.io (24/7 for Enterprise customers)
