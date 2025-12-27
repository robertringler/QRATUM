# QRATUM Sandbox Environment - Quick Reference

## 🚀 One Command Launch

```bash
./sandbox/launch.sh
```

This single command will:
1. ✅ Create isolated Python virtual environment
2. ✅ Install all dependencies
3. ✅ Initialize QRADLE genesis block
4. ✅ Start QRADLE service (port 8001)
5. ✅ Start QRATUM Platform (port 8002)
6. ✅ Verify health checks

## 📁 Files Created

```
sandbox/
├── launch.sh              # Main launch script (local Python)
├── stop.sh                # Stop all services
├── test_sandbox.py        # Automated test suite
├── docker-compose.sandbox.yml  # Docker deployment
└── README.md              # Full documentation

Dockerfile.sandbox-qradle     # QRADLE service Docker image
Dockerfile.sandbox-platform   # Platform service Docker image

.devcontainer/
└── devcontainer.json      # GitHub Codespaces configuration
```

## 🌐 Services & Endpoints

### QRADLE Service (Port 8001)
- **Home**: http://localhost:8001/
- **Health**: http://localhost:8001/health
- **Chain Status**: http://localhost:8001/api/chain/status
- **Engine Status**: http://localhost:8001/api/engine/status

### QRATUM Platform (Port 8002)
- **Dashboard**: http://localhost:8002/
- **API Status**: http://localhost:8002/api/status
- **Metrics**: http://localhost:8002/api/metrics

## 🧪 Testing

```bash
# Run automated tests
python3 sandbox/test_sandbox.py
```

Expected output:
```
✓ All tests passed!

Sandbox Status: HEALTHY

Available Services:
  🛡️  QRADLE:          http://localhost:8001
  🚀 QRATUM Platform: http://localhost:8002
```

## 🐳 Docker Alternative

```bash
# Start with Docker
docker-compose -f sandbox/docker-compose.sandbox.yml up

# Stop with Docker
docker-compose -f sandbox/docker-compose.sandbox.yml down
```

## 🛑 Stopping Services

```bash
# Use the stop script
./sandbox/stop.sh

# Or press Ctrl+C in the launch.sh terminal
```

## 💡 GitHub Codespaces

The devcontainer is pre-configured with:
- Python 3.11
- Auto-installed dependencies
- Pre-forwarded ports (8000, 8001, 8002, 8080)
- VS Code extensions for Python, Docker, YAML

## 📊 Architecture

```
┌─────────────────────────────────────────┐
│        QRATUM Platform (8002)           │
│  ┌──────────────────────────────────┐   │
│  │  QuASIM  │ XENON  │ QUBIC        │   │
│  │  Quantum │ BioInfo │ Visualization│   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│         QRADLE Service (8001)           │
│  ┌──────────────────────────────────┐   │
│  │  Merkle Chain │ Deterministic    │   │
│  │  Genesis Block│ Execution Engine │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## ✅ Verification Checklist

After running `./sandbox/launch.sh`:

- [ ] No error messages during startup
- [ ] Both services show "healthy" status
- [ ] QRADLE health endpoint responds
- [ ] Platform dashboard loads
- [ ] `sandbox/test_sandbox.py` passes all tests
- [ ] Logs show no errors

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | Run `./sandbox/stop.sh` or change ports |
| Dependencies fail | Check Python version (3.10+) |
| Services won't start | Check logs in `/tmp/qradle.log` and `/tmp/platform.log` |
| Health checks fail | Wait a few more seconds, services may be starting |

## 📚 Documentation

- **Full Guide**: [sandbox/README.md](README.md)
- **QRATUM Docs**: [../README.md](../README.md)
- **QRADLE Docs**: [../qradle/README.md](../qradle/README.md)

---

**Quick Start**: `./sandbox/launch.sh` → Wait 10 seconds → Visit http://localhost:8002
