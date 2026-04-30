# QRATUM Sandbox Environment

Launch the complete QRATUM production-ready platform with a single command. The sandbox provides an isolated, fully-configured environment for testing and development.

## 🚀 Quick Start

### Option 1: Local Python Launch (Recommended)

Launch the full QRATUM stack using Python virtual environment:

```bash
./sandbox/launch.sh
```

This will:
- ✅ Create an isolated Python virtual environment
- ✅ Install all dependencies from `requirements.txt` and `requirements-prod.txt`
- ✅ Initialize QRADLE Merkle chain with genesis block
- ✅ Start QRADLE service on port 8001
- ✅ Start QRATUM Platform on port 8002
- ✅ Verify health checks for all services

### Option 2: Docker Compose

Launch using Docker containers:

```bash
docker-compose -f sandbox/docker-compose.sandbox.yml up
```

To run in detached mode:

```bash
docker-compose -f sandbox/docker-compose.sandbox.yml up -d
```

To stop services:

```bash
docker-compose -f sandbox/docker-compose.sandbox.yml down
```

## 📋 Prerequisites

### For Local Python Launch
- **Python 3.10+** (Python 3.11 recommended)
- **pip** (Python package manager)
- **curl** (for health checks)

### For Docker Launch
- **Docker** 20.10+
- **Docker Compose** v2.0+

## 🌐 Available Services

Once the sandbox is running, you can access:

### QRADLE (Port 8001)
The Quantum-Resilient Auditable Deterministic Ledger Engine provides:
- **Landing Page**: http://localhost:8001/
- **Health Check**: http://localhost:8001/health
- **Chain Status**: http://localhost:8001/api/chain/status
- **Engine Status**: http://localhost:8001/api/engine/status

### QRATUM Platform (Port 8002)
The unified QRATUM platform integrating all components:
- **Dashboard**: http://localhost:8002/
- **API Status**: http://localhost:8002/api/status
- **Metrics**: http://localhost:8002/api/metrics

## 🧪 Testing the Sandbox

Run the automated test suite to verify all services:

```bash
python3 sandbox/test_sandbox.py
```

Or with the virtual environment:

```bash
source venv-sandbox/bin/activate
python3 sandbox/test_sandbox.py
```

The test script validates:
- ✅ QRADLE service health
- ✅ Merkle chain initialization
- ✅ Deterministic engine status
- ✅ QRATUM Platform health
- ✅ All endpoints responding correctly

## 🔧 Troubleshooting

### Services Won't Start

**Problem**: Services fail to start or crash immediately

**Solutions**:
1. Check Python version: `python3 --version` (must be 3.10+)
2. Check if ports are already in use:
   ```bash
   lsof -i :8001
   lsof -i :8002
   ```
3. View service logs:
   ```bash
   cat /tmp/qradle.log
   cat /tmp/platform.log
   ```

### Port Already in Use

**Problem**: Error: `Address already in use`

**Solutions**:
1. Find and stop the process using the port:
   ```bash
   lsof -ti:8001 | xargs kill -9
   lsof -ti:8002 | xargs kill -9
   ```
2. Or change the ports in the launch script

### Dependencies Installation Fails

**Problem**: `pip install` fails with errors

**Solutions**:
1. Upgrade pip: `pip install --upgrade pip`
2. Install build dependencies:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3-dev build-essential
   
   # macOS
   xcode-select --install
   ```
3. Use a clean virtual environment:
   ```bash
   rm -rf venv-sandbox
   ./sandbox/launch.sh
   ```

### QRADLE Genesis Block Fails

**Problem**: Genesis block initialization fails

**Solutions**:
1. Check QRADLE module installation:
   ```bash
   python3 -c "from qradle.core.merkle import MerkleChain; print('OK')"
   ```
2. Verify Python path includes the repository root
3. Check for import errors in logs

### Docker Issues

**Problem**: Docker containers fail to build or start

**Solutions**:
1. Rebuild without cache:
   ```bash
   docker-compose -f sandbox/docker-compose.sandbox.yml build --no-cache
   ```
2. Check Docker logs:
   ```bash
   docker-compose -f sandbox/docker-compose.sandbox.yml logs qradle
   docker-compose -f sandbox/docker-compose.sandbox.yml logs platform
   ```
3. Ensure Docker daemon is running:
   ```bash
   docker ps
   ```

### Health Checks Fail

**Problem**: Services start but health checks fail

**Solutions**:
1. Wait a few more seconds - services may still be initializing
2. Test manually:
   ```bash
   curl http://localhost:8001/health
   curl http://localhost:8002/
   ```
3. Check if services are listening:
   ```bash
   netstat -an | grep -E '8001|8002'
   ```

## 🐛 Debug Mode

To run services with verbose logging:

### Local Python
Edit `/tmp/qradle_server.py` and change:
```python
app.run(host='0.0.0.0', port=8001, debug=True)
```

### Docker
Add to `docker-compose.sandbox.yml`:
```yaml
environment:
  - FLASK_DEBUG=1
  - PYTHONUNBUFFERED=1
```

## 📊 Monitoring

### View Logs (Local)
```bash
# Follow logs in real-time
tail -f /tmp/qradle.log
tail -f /tmp/platform.log
```

### View Logs (Docker)
```bash
# Follow all logs
docker-compose -f sandbox/docker-compose.sandbox.yml logs -f

# Follow specific service
docker-compose -f sandbox/docker-compose.sandbox.yml logs -f qradle
docker-compose -f sandbox/docker-compose.sandbox.yml logs -f platform
```

### Check Service Status
```bash
# Local
ps aux | grep -E 'qradle_server|qratum_platform'

# Docker
docker-compose -f sandbox/docker-compose.sandbox.yml ps
```

## 🛑 Stopping Services

### Local Python
Press `Ctrl+C` in the terminal where `launch.sh` is running, or use the stop script:

```bash
./sandbox/stop.sh
```

Alternatively, find and kill processes manually:
```bash
pkill -f qradle_server
pkill -f qratum_platform
```

### Docker
```bash
docker-compose -f sandbox/docker-compose.sandbox.yml down

# Remove volumes too
docker-compose -f sandbox/docker-compose.sandbox.yml down -v
```

## 🔐 Security Notes

- The sandbox is intended for **development and testing only**
- Uses default configurations - not suitable for production
- No authentication is enabled
- Services are exposed on localhost by default

## 📝 Configuration

### Environment Variables

#### QRADLE Service
- `QRADLE_PORT`: Port for QRADLE service (default: 8001)
- `QRADLE_NETWORK`: Network identifier (default: sandbox)

#### QRATUM Platform
- `PORT`: Port for platform service (default: 8002)
- `QRADLE_URL`: URL to QRADLE service (default: http://localhost:8001)

### Customization

To customize the sandbox:

1. **Change Ports**: Edit `sandbox/launch.sh` or `sandbox/docker-compose.sandbox.yml`
2. **Modify Services**: Edit Dockerfile templates in root directory
3. **Add Components**: Extend the docker-compose file with additional services

## 🌟 Features

### QRADLE Features
- ✅ Deterministic execution with cryptographic proofs
- ✅ Merkle-chained audit trails
- ✅ Contract-based operations
- ✅ Rollback capability
- ✅ 8 Fatal Invariants enforcement

### QRATUM Platform Features
- ✅ Unified command center dashboard
- ✅ Multi-component integration (QuASIM, XENON, QUBIC)
- ✅ Real-time metrics and monitoring
- ✅ RESTful API endpoints
- ✅ Cross-domain synthesis

## 🆘 Getting Help

If you encounter issues not covered in this guide:

1. Check existing [GitHub Issues](https://github.com/robertringler/QRATUM/issues)
2. Review the main [QRATUM README](../README.md)
3. Check component-specific documentation:
   - [QRADLE Documentation](../qradle/README.md)
   - [QuASIM Documentation](../quasim/README.md)
4. Open a new issue with:
   - Error messages
   - Log outputs
   - System information (OS, Python version)
   - Steps to reproduce

## 📚 Additional Resources

- **Main Documentation**: [README.md](../README.md)
- **Architecture Guide**: [QRATUM_ARCHITECTURE.md](../QRATUM_ARCHITECTURE.md)
- **API Reference**: [API_REFERENCE.md](../API_REFERENCE.md)
- **Contributing**: [CONTRIBUTING.md](../CONTRIBUTING.md)

## ✅ Verification Checklist

After launching the sandbox, verify:

- [ ] Python virtual environment created successfully
- [ ] All dependencies installed without errors
- [ ] QRADLE genesis block initialized
- [ ] QRADLE service responding on port 8001
- [ ] QRATUM Platform responding on port 8002
- [ ] Health checks passing for both services
- [ ] Test suite passes (`sandbox/test_sandbox.py`)
- [ ] Logs show no errors

---

**Version**: 1.0.0  
**Last Updated**: 2025-12-27  
**Maintainer**: QRATUM Development Team
