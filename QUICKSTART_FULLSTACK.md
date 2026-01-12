# QRATUM Full Stack - Production Deployment

## 🚀 Quick Start

Deploy the complete QRATUM stack (QRADLE + 14 Verticals + ASI) in under 5 minutes:

```bash
# Clone repository
git clone https://github.com/robertringler/QRATUM.git
cd QRATUM

# Start services
docker-compose -f docker-compose.production.yml up -d

# Verify deployment
curl http://localhost:8001/health  # QRADLE
curl http://localhost:8002/health  # Platform (14 verticals)
curl http://localhost:8003/health  # ASI

# Access dashboards
open http://localhost:3000  # Grafana (use GRAFANA_PASSWORD from .env)
open http://localhost:9090  # Prometheus
```

## 📦 What's Included

### QRADLE Foundation

Deterministic execution engine with:

- ✅ Merkle-chained audit trails
- ✅ Checkpoint & rollback capability
- ✅ 8 Fatal Invariants enforcement
- ✅ Cryptographic proof generation

### QRATUM Platform - 14 Verticals

1. **JURIS** - Legal AI & Compliance
2. **VITRA** - Bioinformatics & Drug Discovery
3. **ECORA** - Climate & Energy
4. **CAPRA** - Financial Risk
5. **SENTRA** - Aerospace & Defense
6. **NEURA** - Neuroscience & BCI
7. **FLUXA** - Supply Chain
8. **SPECTRA** - Spectrum Management
9. **AEGIS** - Cybersecurity
10. **LOGOS** - Education & Training
11. **SYNTHOS** - Materials Science
12. **TERAGON** - Geospatial Intelligence
13. **HELIX** - Genomic Medicine
14. **NEXUS** - Cross-Domain Synthesis

### QRATUM-ASI Layer

- Q-REALITY: World model
- Q-MIND: Unified reasoning
- Q-EVOLVE: Safe self-improvement (simulation)
- Q-WILL: Intent generation (simulation)
- Q-FORGE: Discovery engine

### Infrastructure

- 🐳 Production Docker containers
- 📊 Prometheus monitoring
- 📈 Grafana dashboards
- 📝 Fluentd audit logging
- 🔄 CI/CD pipeline
- 🔒 Security hardening

## 📖 Documentation

| Document | Description | Size |
|----------|-------------|------|
| [API Reference](docs/API_REFERENCE.md) | Complete API docs for all 14 verticals | 8.7KB |
| [Deployment Guide](docs/DEPLOYMENT.md) | Production deployment instructions | 11KB |
| [Operator Guide](docs/OPERATOR_GUIDE.md) | Daily operations & monitoring | 12KB |
| [Audit Log Examples](docs/AUDIT_LOG_EXAMPLES.md) | Audit trail examples & compliance | 11KB |
| [Multi-Domain Workflows](docs/MULTI_DOMAIN_WORKFLOWS.md) | Cross-domain synthesis examples | 14KB |
| [Deployment Summary](DEPLOYMENT_SUMMARY.md) | Complete implementation summary | 10KB |

## 🎯 Example Usage

### Execute a Contract (QRADLE)

```bash
curl -X POST http://localhost:8001/api/v1/qradle/contract \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "add",
    "inputs": {"a": 5, "b": 3},
    "user_id": "user_123"
  }'
```

### Run Legal Analysis (JURIS)

```bash
curl -X POST http://localhost:8002/api/v1/platform/execute \
  -H "Content-Type: application/json" \
  -d '{
    "vertical": "juris",
    "operation": "legal_reasoning",
    "parameters": {
      "facts": "Party A breached contract with Party B",
      "area_of_law": "contract"
    },
    "user_id": "lawyer_456"
  }'
```

### Multi-Domain Synthesis (NEXUS)

```bash
curl -X POST http://localhost:8002/api/v1/platform/execute \
  -H "Content-Type: application/json" \
  -d '{
    "vertical": "nexus",
    "operation": "multi_domain_synthesis",
    "parameters": {
      "domains": ["VITRA", "ECORA", "FLUXA"],
      "query": "Sustainable pharmaceutical manufacturing"
    },
    "user_id": "researcher_789"
  }'
```

### Create Checkpoint

```bash
curl -X POST http://localhost:8001/api/v1/system/checkpoint \
  -H "Content-Type: application/json" \
  -d '{"description": "Daily checkpoint"}'
```

### Get Audit Trail

```bash
curl http://localhost:8001/api/v1/audit/trail | jq .
```

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────┐
│              QRATUM Full Stack (Ports)             │
├────────────────────────────────────────────────────┤
│  QRATUM-ASI (8003)    │ Q-REALITY │ Q-MIND │ ...  │
│  ─────────────────────┼──────────────────────────  │
│  Platform (8002)      │ 14 Verticals               │
│  ─────────────────────┼──────────────────────────  │
│  QRADLE (8001)        │ Engine │ Merkle │ Rollback│
└────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────┐
│        Infrastructure & Monitoring                 │
│  Prometheus (9090) │ Grafana (3000) │ Fluentd     │
└────────────────────────────────────────────────────┘
```

## 🔐 Security Features

- ✅ Non-root container execution
- ✅ Read-only filesystem
- ✅ No new privileges
- ✅ Security scanning (Trivy)
- ✅ Audit logging (tamper-evident)
- ✅ Network isolation
- ✅ TLS/SSL ready

## 📋 Compliance Ready

Designed for certification:

- **DO-178C** (airborne systems)
- **CMMC Level 3** (defense)
- **ISO 27001** (security)
- **HIPAA** (healthcare)
- **SOC 2 Type 2** (controls)
- **GDPR** (data protection)

## 🚦 Health Monitoring

```bash
# Check all services
for port in 8001 8002 8003; do
  curl -s http://localhost:$port/health | jq .
done

# View metrics
open http://localhost:9090/graph

# View dashboards
open http://localhost:3000
```

## 🔄 Backup & Recovery

```bash
# Create backup
./backup.sh

# Restore from backup
./restore.sh /backups/qratum-20231222-120000
```

## 📊 Monitoring Dashboards

Access Grafana at <http://localhost:3000> using username `admin` and the password set in your `GRAFANA_PASSWORD` environment variable.

**Security Warning:** The `GRAFANA_PASSWORD` environment variable must be set before starting services. Never use default credentials in production.

Pre-configured dashboards:

1. **System Health** - Overall status
2. **QRADLE Operations** - Contract execution metrics
3. **Platform Verticals** - Per-vertical performance
4. **ASI Orchestrator** - ASI component status

## 🐛 Troubleshooting

### Service won't start

```bash
docker-compose -f docker-compose.production.yml logs qradle
```

### Check resource usage

```bash
docker stats
```

### Verify integrity

```bash
curl http://localhost:8001/api/v1/system/proof
```

## 🔧 Configuration

Create `.env` file:

```bash
QRATUM_VERSION=1.0.0
QRATUM_ENV=production
QRATUM_LOG_LEVEL=INFO
GRAFANA_PASSWORD=<secure-password>
MERKLE_CHAIN_PERSISTENCE=true
ASI_MODE=simulation
CRSI_ENABLED=false
```

## 🌐 Deployment Options

### On-Premises

Standard Docker Compose deployment on your infrastructure.

### Air-Gapped

Complete offline deployment package available. See [DEPLOYMENT.md](docs/DEPLOYMENT.md).

### High Availability

Multi-node deployment with load balancing. See [DEPLOYMENT.md](docs/DEPLOYMENT.md#high-availability-setup).

## 🚀 CI/CD Pipeline

Automated pipeline in `.github/workflows/qratum-production-ci.yml`:

1. ✅ Lint & Type Check
2. ✅ Unit Tests
3. ✅ Integration Tests
4. ✅ Security Scanning
5. ✅ Docker Build
6. ✅ Determinism Verification
7. ✅ Staging Deploy
8. ✅ Production Deploy
9. ✅ Audit Report

## 🎓 Learning Resources

- **Video Tutorials:** Coming soon
- **API Documentation:** [API_REFERENCE.md](docs/API_REFERENCE.md)
- **Workflow Examples:** [MULTI_DOMAIN_WORKFLOWS.md](docs/MULTI_DOMAIN_WORKFLOWS.md)
- **Operations Guide:** [OPERATOR_GUIDE.md](docs/OPERATOR_GUIDE.md)

## 🤝 Support

- **Documentation:** `/docs/`
- **GitHub Issues:** <https://github.com/robertringler/QRATUM/issues>
- **Email:** <support@qratum.io>
- **Security:** <security@qratum.io>

## 📝 License

Apache 2.0 - See [LICENSE](LICENSE)

## 🙏 Acknowledgments

Built on the QRATUM platform architecture with contributions from domain experts across legal, medical, climate, finance, defense, and AI safety.

---

**Status: Production Ready ✓**

Deploy the future of sovereign, auditable AI today.
