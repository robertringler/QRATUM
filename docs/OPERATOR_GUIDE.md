# QRATUM Operator Quick-Start Guide

## Overview

This guide provides essential information for QRATUM platform operators responsible for day-to-day operations, monitoring, and maintenance.

## Quick Reference

### Service Endpoints

| Service | Port | URL | Purpose |
|---------|------|-----|---------|
| QRADLE | 8001 | <http://localhost:8001> | Deterministic execution engine |
| Platform | 8002 | <http://localhost:8002> | 14 vertical modules |
| ASI | 8003 | <http://localhost:8003> | ASI orchestrator (simulation) |
| Prometheus | 9090 | <http://localhost:9090> | Metrics |
| Grafana | 3000 | <http://localhost:3000> | Dashboards |

### Common Commands

```bash
# Check service health
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f --tail=100

# Restart service
docker-compose -f docker-compose.production.yml restart qradle

# Create checkpoint
curl -X POST http://localhost:8001/api/v1/system/checkpoint \
  -H "Content-Type: application/json" \
  -d '{"description": "Manual checkpoint"}'

# Check system proof
curl http://localhost:8001/api/v1/system/proof | jq .
```

---

## Daily Operations

### Morning Checklist

1. **Check Service Health**

   ```bash
   # All services should show "healthy"
   curl http://localhost:8001/health
   curl http://localhost:8002/health
   curl http://localhost:8003/health
   ```

2. **Review Overnight Metrics**
   - Open Grafana: <http://localhost:3000>
   - Check "System Health" dashboard
   - Look for anomalies in request rates, error rates, latency

3. **Verify Merkle Chain Integrity**

   ```bash
   curl http://localhost:8001/api/v1/system/proof | jq '.integrity_verified'
   # Should return: true
   ```

4. **Check Disk Space**

   ```bash
   df -h | grep -E 'qradle|platform|asi'
   ```

5. **Review Error Logs**

   ```bash
   docker-compose -f docker-compose.production.yml logs --since=24h | grep ERROR
   ```

### Hourly Monitoring

- Check Grafana "System Health" dashboard
- Monitor request rates and latencies
- Watch for spike in error rates
- Verify all services are running

### Weekly Tasks

1. **Create Weekly Checkpoint**

   ```bash
   DATE=$(date +%Y-%m-%d)
   curl -X POST http://localhost:8001/api/v1/system/checkpoint \
     -H "Content-Type: application/json" \
     -d "{\"description\": \"Weekly checkpoint $DATE\"}"
   ```

2. **Review Audit Logs**

   ```bash
   ls -lh audit-logs/
   # Archive logs older than 90 days
   ```

3. **Check Backup Status**

   ```bash
   ./backup.sh
   ls -lh /backups/ | tail -5
   ```

4. **Update Monitoring Dashboard**
   - Review alerts in Grafana
   - Update alert thresholds if needed

---

## Monitoring & Alerts

### Key Metrics

1. **Request Rate**
   - Normal: 10-1000 req/s
   - Alert: Sudden drop or spike >5000 req/s

2. **Error Rate**
   - Normal: <0.1%
   - Warning: 0.1-1%
   - Critical: >1%

3. **Response Time**
   - p50: <100ms
   - p95: <500ms
   - p99: <1000ms

4. **Merkle Chain Length**
   - Continuously increasing
   - Alert: No growth for >5 minutes

5. **Checkpoint Count**
   - Should increase regularly
   - Alert: No new checkpoints for >24 hours

### Grafana Dashboards

**System Health Dashboard:**

- Overall system status
- Service health indicators
- Resource utilization (CPU, memory, disk)
- Network throughput

**QRADLE Operations Dashboard:**

- Contract execution rate
- Execution time distribution
- Merkle chain metrics
- Checkpoint timeline

**Platform Verticals Dashboard:**

- Per-vertical request rates
- Vertical-specific latencies
- Error rates by vertical
- Cross-domain synthesis metrics

**ASI Orchestrator Dashboard:**

- ASI component status
- Safety level distribution
- CRSI proposal metrics (when enabled)

### Alert Configuration

Edit `configs/prometheus.yml` for custom alerts:

```yaml
groups:
  - name: qratum_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(qratum_errors_total[5m]) / rate(qratum_requests_total[5m]) > 0.01
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
      
      - alert: ServiceDown
        expr: up{job=~"qradle|platform|asi"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"
```

---

## Incident Response

### Service Down

1. **Identify affected service:**

   ```bash
   docker-compose -f docker-compose.production.yml ps
   ```

2. **Check logs:**

   ```bash
   docker-compose -f docker-compose.production.yml logs qradle --tail=200
   ```

3. **Restart service:**

   ```bash
   docker-compose -f docker-compose.production.yml restart qradle
   ```

4. **Verify recovery:**

   ```bash
   curl http://localhost:8001/health
   ```

5. **Document incident** in operations log

### High Error Rate

1. **Check recent logs:**

   ```bash
   docker-compose -f docker-compose.production.yml logs --since=10m | grep ERROR
   ```

2. **Identify error pattern:**
   - Specific vertical?
   - Specific operation?
   - Time-based?

3. **Review recent changes:**
   - Recent deployments?
   - Configuration changes?
   - External dependencies?

4. **Take corrective action:**
   - Rollback deployment if needed
   - Adjust configuration
   - Scale resources if load-related

### Integrity Violation

**CRITICAL - Requires immediate action!**

1. **Stop accepting new requests:**

   ```bash
   # Add firewall rule to block incoming traffic
   ufw deny 8001/tcp
   ufw deny 8002/tcp
   ```

2. **Capture current state:**

   ```bash
   curl http://localhost:8001/api/v1/system/proof > integrity-violation-$(date +%s).json
   curl http://localhost:8001/api/v1/audit/trail > audit-trail-$(date +%s).json
   ```

3. **Identify last valid checkpoint:**

   ```bash
   # List recent checkpoints
   curl http://localhost:8001/api/v1/system/checkpoints | jq .
   ```

4. **Rollback to checkpoint:**

   ```bash
   curl -X POST http://localhost:8001/api/v1/system/rollback \
     -H "Content-Type: application/json" \
     -d '{"checkpoint_id": "<last-valid-checkpoint>"}'
   ```

5. **Notify security team**

6. **Begin investigation:**
   - Review audit logs
   - Check for unauthorized access
   - Analyze Merkle chain for tampering

---

## Maintenance Procedures

### Routine Restart

```bash
# Create pre-restart checkpoint
curl -X POST http://localhost:8001/api/v1/system/checkpoint \
  -d '{"description": "Pre-restart checkpoint"}'

# Restart service
docker-compose -f docker-compose.production.yml restart qradle

# Verify health
sleep 10
curl http://localhost:8001/health
```

### Rolling Update

```bash
# Pull latest images
docker-compose -f docker-compose.production.yml pull

# Create checkpoint
curl -X POST http://localhost:8001/api/v1/system/checkpoint \
  -d '{"description": "Pre-update checkpoint"}'

# Update one service at a time
docker-compose -f docker-compose.production.yml up -d --no-deps qradle

# Wait and verify
sleep 30
curl http://localhost:8001/health

# Repeat for other services
docker-compose -f docker-compose.production.yml up -d --no-deps qratum-platform
# ... etc
```

### Log Rotation

```bash
# Manual log rotation
docker-compose -f docker-compose.production.yml logs > logs-backup-$(date +%Y%m%d).txt
docker-compose -f docker-compose.production.yml restart

# Archive old logs
gzip logs-backup-*.txt
mv logs-backup-*.txt.gz /backups/logs/
```

### Certificate Renewal

```bash
# Renew Let's Encrypt certificate
certbot renew

# Copy new certificates
cp /etc/letsencrypt/live/qratum.io/fullchain.pem certs/
cp /etc/letsencrypt/live/qratum.io/privkey.pem certs/

# Restart services to pick up new certs
docker-compose -f docker-compose.production.yml restart
```

---

## Troubleshooting Guide

### Service Won't Start

**Symptoms:** Service shows "restarting" or "exited"

**Solutions:**

1. Check logs: `docker-compose logs qradle`
2. Verify configuration: `docker-compose config`
3. Check port conflicts: `netstat -tuln | grep 8001`
4. Verify resources: `docker stats`
5. Check permissions: `ls -la /app`

### Slow Response Times

**Symptoms:** High latency, timeouts

**Solutions:**

1. Check CPU/memory: `docker stats`
2. Check disk I/O: `iostat -x 1`
3. Review slow queries in logs
4. Scale horizontally if needed
5. Add caching layer

### Checkpoint Creation Fails

**Symptoms:** Error when creating checkpoint

**Solutions:**

1. Check disk space: `df -h`
2. Verify write permissions: `ls -la checkpoints/`
3. Check Merkle chain integrity
4. Review logs for specific error
5. Rollback to previous checkpoint if needed

### Authentication Errors

**Symptoms:** 401/403 errors

**Solutions:**

1. Verify API key is valid
2. Check user permissions
3. Review authorization logs
4. Verify token hasn't expired
5. Check certificate validity

---

## Performance Optimization

### Resource Tuning

```yaml
# docker-compose.production.yml
services:
  qradle:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

### Caching Configuration

```bash
# Add Redis for caching
docker-compose -f docker-compose.production.yml \
  -f docker-compose.cache.yml up -d redis
```

### Database Optimization

```bash
# PostgreSQL tuning
# Edit postgresql.conf
shared_buffers = 4GB
effective_cache_size = 16GB
work_mem = 128MB
maintenance_work_mem = 1GB
```

---

## Security Operations

### Daily Security Checks

1. **Review failed login attempts**
2. **Check for unauthorized API calls**
3. **Verify SSL certificate validity**
4. **Review firewall logs**
5. **Check for security updates**

### Security Scanning

```bash
# Scan for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image qratum/fullstack:latest

# Check for secrets in logs
grep -ri "password\|token\|secret\|key" logs/
```

### Access Control

```bash
# Review active sessions
curl http://localhost:8001/api/v1/admin/sessions

# Revoke API key
curl -X DELETE http://localhost:8001/api/v1/admin/keys/<key-id>
```

---

## Escalation Procedures

### Severity Levels

**P0 - Critical (Response: Immediate)**

- All services down
- Data integrity violation
- Security breach
- Data loss

**P1 - High (Response: < 1 hour)**

- Single service down
- High error rate (>5%)
- Performance degradation (>3x latency)

**P2 - Medium (Response: < 4 hours)**

- Intermittent errors
- Non-critical feature broken
- Monitoring issues

**P3 - Low (Response: < 24 hours)**

- Minor bugs
- Documentation issues
- Feature requests

### Contact Information

- **On-Call Engineer:** +1-XXX-XXX-XXXX
- **Engineering Manager:** <engineer@qratum.io>
- **Security Team:** <security@qratum.io>
- **24/7 Support:** <support@qratum.io>

---

## Useful Scripts

### Health Check Script

```bash
#!/bin/bash
# health-check.sh

services=("qradle:8001" "platform:8002" "asi:8003")

for service in "${services[@]}"; do
  name="${service%%:*}"
  port="${service##*:}"
  
  if curl -sf "http://localhost:$port/health" > /dev/null; then
    echo "✓ $name is healthy"
  else
    echo "✗ $name is unhealthy"
    exit 1
  fi
done

echo "All services healthy"
```

### Backup Script

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="/backups/qratum-$DATE"

mkdir -p "$BACKUP_DIR"

# Backup volumes
for volume in qradle-data qradle-audit platform-data asi-checkpoints; do
  docker run --rm \
    -v "$volume:/data" \
    -v "$BACKUP_DIR:/backup" \
    alpine tar czf "/backup/$volume.tar.gz" -C /data .
done

# Backup configs
tar czf "$BACKUP_DIR/configs.tar.gz" configs/ docker-compose.production.yml .env

echo "Backup completed: $BACKUP_DIR"
```

### Metrics Collection

```bash
#!/bin/bash
# collect-metrics.sh

curl -s http://localhost:8001/api/v1/system/proof | \
  jq '{merkle_root, chain_length, integrity_verified}' > metrics-$(date +%s).json
```

---

## Training Resources

- **Video Tutorials:** <https://training.qratum.io>
- **Documentation:** <https://docs.qratum.io>
- **API Reference:** [API_REFERENCE.md](./API_REFERENCE.md)
- **Deployment Guide:** [DEPLOYMENT.md](./DEPLOYMENT.md)

---

## Support

- **Operator Slack Channel:** #qratum-ops
- **Internal Wiki:** <https://wiki.qratum.io>
- **Runbook Repository:** <https://github.com/qratum/runbooks>
- **On-Call Schedule:** <https://oncall.qratum.io>
