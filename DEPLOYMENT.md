# QRATUM Production Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the complete QRATUM stack in production environments, including:

- **QRADLE**: Deterministic execution foundation
- **QRATUM Platform**: 14 vertical AI modules
- **QRATUM-ASI**: Theoretical ASI scaffolding (simulation mode)

## Deployment Modes

### 1. On-Premises Deployment

Full stack on your own infrastructure with complete data sovereignty.

### 2. Air-Gapped Deployment

Completely isolated network with no internet connectivity.

### 3. Private Cloud Deployment

Dedicated VPC with no public internet egress.

## Prerequisites

### Hardware Requirements

**Minimum (Development/Testing)**:

- 16 GB RAM
- 4-core CPU
- 100 GB SSD storage
- Linux (Ubuntu 20.04+, RHEL 8+, or compatible)

**Recommended (Production)**:

- 64 GB RAM (128 GB for large-scale)
- 16+ core CPU (32+ recommended)
- 500 GB NVMe SSD storage
- GPU: NVIDIA A100/H100 (optional, for accelerated inference)
- 10 Gbps network

### Software Requirements

- Docker 20.10+ and Docker Compose 1.29+
- Python 3.11+
- Git (for initial clone only)

## Installation Steps

### Step 1: Clone Repository

```bash
# On internet-connected machine (for air-gapped, transfer archive)
git clone https://github.com/robertringler/QRATUM.git
cd QRATUM
```

For air-gapped deployment, create archive:

```bash
# On internet-connected machine
tar -czf qratum-deployment.tar.gz QRATUM/
# Transfer qratum-deployment.tar.gz to air-gapped environment
```

### Step 2: Configuration

Create production configuration:

```bash
# Copy example configuration
cp config/production.example.yml config/production.yml

# Edit configuration for your environment
vim config/production.yml
```

Key configuration parameters:

- `QRADLE_MODE`: Set to `production`
- `ENABLE_INVARIANTS`: Always `true` in production
- `LOG_LEVEL`: `INFO` or `WARNING` for production
- `HIPAA_COMPLIANCE`: Enable if handling PHI
- `FINRA_COMPLIANCE`: Enable for financial services
- `CMMC_LEVEL`: Set to required level (1-3)

### Step 3: Build Docker Images

```bash
# Build all images
docker-compose -f docker-compose.production.yml build

# Or build specific services
docker-compose -f docker-compose.production.yml build qradle
docker-compose -f docker-compose.production.yml build qratum-api
```

### Step 4: Initialize System

```bash
# Initialize QRADLE with genesis block
docker-compose -f docker-compose.production.yml run --rm qradle \
    python -c "from qradle import MerkleChain; chain = MerkleChain(); print(f'Genesis: {chain.get_root_hash()}')"
```

### Step 5: Deploy Stack

```bash
# Start all services
docker-compose -f docker-compose.production.yml up -d

# Verify services are running
docker-compose -f docker-compose.production.yml ps

# Check logs
docker-compose -f docker-compose.production.yml logs -f qratum-api
```

### Step 6: Health Verification

```bash
# Check API health
curl http://localhost:8000/health

# List available verticals
curl http://localhost:8000/api/v1/verticals

# Get QRADLE statistics
docker-compose -f docker-compose.production.yml exec qradle \
    python -c "from qradle import DeterministicEngine; e = DeterministicEngine(); print(e.get_stats())"
```

## Post-Deployment Configuration

### SSL/TLS Certificates

For production, enable HTTPS:

```bash
# Generate self-signed certificate (for testing)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout certs/qratum.key -out certs/qratum.crt

# Or use your organization's CA-signed certificates
cp /path/to/cert.crt certs/qratum.crt
cp /path/to/cert.key certs/qratum.key
```

Update `docker-compose.production.yml` to mount certificates.

### Authentication & Authorization

Configure API authentication:

1. **API Keys**: Environment variables for service-to-service auth
2. **OAuth2/OIDC**: For user authentication (if required)
3. **mTLS**: For service mesh security

```yaml
# In docker-compose.production.yml
environment:
  - API_AUTH_ENABLED=true
  - API_KEY_HEADER=X-API-Key
  - REQUIRE_CLIENT_CERT=true  # For mTLS
```

### Monitoring & Alerting

Access monitoring dashboards:

- **Prometheus**: <http://localhost:9090>
- **Grafana**: <http://localhost:3000> (default: admin/changeme)
- **Logs**: Check with `docker-compose logs -f`

Configure alerts in `config/prometheus-alerts.yml`.

### Backup & Disaster Recovery

#### Automated Backups

```bash
# Backup script (run via cron)
#!/bin/bash
BACKUP_DIR=/backups/qratum-$(date +%Y%m%d-%H%M%S)
mkdir -p $BACKUP_DIR

# Backup volumes
docker run --rm -v qradle-data:/data -v $BACKUP_DIR:/backup \
    alpine tar czf /backup/qradle-data.tar.gz -C /data .

docker run --rm -v qradle-checkpoints:/data -v $BACKUP_DIR:/backup \
    alpine tar czf /backup/qradle-checkpoints.tar.gz -C /data .

# Backup configuration
cp -r config/ $BACKUP_DIR/config/

echo "Backup completed: $BACKUP_DIR"
```

#### Restore from Backup

```bash
# Stop services
docker-compose -f docker-compose.production.yml down

# Restore volumes
docker run --rm -v qradle-data:/data -v /backups/latest:/backup \
    alpine sh -c "cd /data && tar xzf /backup/qradle-data.tar.gz"

# Start services
docker-compose -f docker-compose.production.yml up -d
```

## Air-Gapped Deployment

### Prepare Deployment Package

On internet-connected machine:

```bash
# Build images
docker-compose -f docker-compose.production.yml build

# Save images to archive
docker save \
    qratum-qradle:latest \
    qratum-api:latest \
    qratum-juris:latest \
    qratum-vitra:latest \
    qratum-ecora:latest \
    qratum-capra:latest \
    qratum-sentra:latest \
    > qratum-images.tar

# Create full deployment package
tar -czf qratum-airgapped-deployment.tar.gz \
    QRATUM/ \
    qratum-images.tar
```

### Deploy in Air-Gapped Environment

```bash
# Transfer and extract
tar -xzf qratum-airgapped-deployment.tar.gz
cd QRATUM

# Load Docker images
docker load < ../qratum-images.tar

# Deploy
docker-compose -f docker-compose.production.yml up -d
```

## Scaling & Performance

### Horizontal Scaling

Scale specific verticals:

```bash
# Scale JURIS vertical to 3 instances
docker-compose -f docker-compose.production.yml up -d --scale qratum-juris=3

# Scale API gateway
docker-compose -f docker-compose.production.yml up -d --scale qratum-api=3
```

### GPU Acceleration

Enable GPU for inference:

```yaml
# In docker-compose.production.yml
qratum-vitra:
  # ... existing config
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

## Security Hardening

### Network Isolation

```bash
# Restrict external access
iptables -A INPUT -p tcp --dport 8000 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 8000 -j DROP
```

### Secrets Management

Use Docker secrets for sensitive data:

```bash
# Create secrets
echo "your-api-key" | docker secret create qratum_api_key -

# Reference in docker-compose.yml
secrets:
  qratum_api_key:
    external: true
```

## Compliance & Certification

### DO-178C Level A Preparation

1. Enable deterministic mode: `QRADLE_DETERMINISTIC=strict`
2. Maintain complete audit logs
3. Generate certification artifacts:

   ```bash
   docker-compose exec qradle python -m qradle.compliance.do178c
   ```

### CMMC Level 3 Configuration

```yaml
environment:
  - CMMC_LEVEL=3
  - ENABLE_FIPS_MODE=true
  - LOG_ALL_ACCESS=true
  - REQUIRE_MFA=true
```

### HIPAA Compliance

For healthcare deployments:

```yaml
environment:
  - HIPAA_COMPLIANCE=true
  - PHI_ENCRYPTION=AES-256
  - AUDIT_LOG_RETENTION_DAYS=2555  # 7 years
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose -f docker-compose.production.yml logs qradle

# Verify configuration
docker-compose -f docker-compose.production.yml config

# Check resource usage
docker stats
```

### Merkle Chain Verification Failed

```bash
# Verify chain integrity
docker-compose exec qradle python -c \
    "from qradle import DeterministicEngine; e = DeterministicEngine(); print(e.merkle_chain.verify_chain_integrity())"
```

### Performance Issues

```bash
# Check QRADLE stats
docker-compose exec qradle python -m qradle.diagnostics

# Monitor resource usage
docker stats --no-stream

# Analyze slow queries
docker-compose logs qratum-api | grep "execution_time"
```

## Maintenance

### Regular Tasks

1. **Daily**: Check health endpoints, review logs
2. **Weekly**: Verify backups, test restore process
3. **Monthly**: Update dependencies (security patches), prune old checkpoints
4. **Quarterly**: Performance optimization, capacity planning

### Updating QRATUM

```bash
# Backup current deployment
./scripts/backup.sh

# Pull latest changes
git pull origin main

# Rebuild images
docker-compose -f docker-compose.production.yml build

# Rolling update
docker-compose -f docker-compose.production.yml up -d --no-deps qratum-api
```

## Support

For production support:

- **Email**: <support@qratum.io>
- **Security Issues**: <security@qratum.io>
- **Documentation**: <https://docs.qratum.io>
- **GitHub Issues**: <https://github.com/robertringler/QRATUM/issues>

## License

Copyright 2025 QRATUM Contributors
Licensed under Apache 2.0 License
