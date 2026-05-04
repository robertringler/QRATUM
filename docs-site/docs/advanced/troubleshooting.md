# Troubleshooting

Common issues and solutions for QRATUM.

## Installation Issues

### Python Version Errors

**Error:**
```
ERROR: Package requires Python >=3.10 but you have Python 3.9
```

**Solution:**
```bash
# Install Python 3.10+
# macOS
brew install python@3.11

# Ubuntu
sudo apt install python3.11

# Use pyenv
pyenv install 3.11.5
pyenv local 3.11.5
```

### Qiskit Installation Fails

**Error:**
```
ERROR: Could not build wheels for qiskit
```

**Solution:**
```bash
# Install build dependencies
pip install --upgrade pip setuptools wheel

# On Ubuntu, install system deps
sudo apt install python3-dev build-essential

# On macOS
xcode-select --install
```

### cuQuantum Not Found

**Error:**
```python
ImportError: cuQuantum not available
```

**Solution:**
```bash
# Install cuQuantum
pip install cuquantum-python

# Or install from NVIDIA
# https://docs.nvidia.com/cuda/cuquantum/
```

## Runtime Errors

### Out of Memory

**Error:**
```
numpy.core._exceptions.MemoryError: Unable to allocate array
```

**Solution:**
```python
# Reduce qubit count
config = QuantumConfig(max_qubits=20)

# Or limit memory per process
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
```

### Convergence Failure

**Error:**
```python
ConvergenceError: VQE did not converge after 100 iterations
```

**Solution:**
```python
# Increase iterations
result = vqe.compute_h2_energy(
    bond_length=0.735,
    max_iterations=500
)

# Or try different optimizer
result = vqe.compute_h2_energy(
    bond_length=0.735,
    optimizer="L-BFGS-B"
)

# Or increase ansatz depth
result = vqe.compute_h2_energy(
    bond_length=0.735,
    ansatz_layers=4
)
```

### Backend Connection Failed

**Error:**
```python
IBMQBackendError: Could not connect to IBM Quantum
```

**Solution:**
```python
# Check token
import os
print(os.environ.get('IBMQ_TOKEN'))

# Or configure directly
config = QuantumConfig(
    backend_type="ibmq",
    ibmq_token="your_actual_token"
)

# Use simulator as fallback
config = QuantumConfig(backend_type="simulator")
```

## Docker Issues

### Container Won't Start

**Error:**
```
Error response from daemon: driver failed programming external connectivity
```

**Solution:**
```bash
# Check port availability
lsof -i :8000
lsof -i :8080

# Stop conflicting containers
docker-compose down
docker stop $(docker ps -q)

# Restart Docker
sudo systemctl restart docker
```

### Build Fails

**Error:**
```
ERROR: failed to solve: failed to compute cache key
```

**Solution:**
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

### Services Not Connecting

**Error:**
```
ConnectionRefusedError: Connection refused on localhost:8000
```

**Solution:**
```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs backend

# Ensure services started in correct order
docker-compose up -d redis
docker-compose up -d backend
docker-compose up -d frontend
```

## Kubernetes Issues

### Pods Not Starting

**Error:**
```
CrashLoopBackOff
```

**Solution:**
```bash
# Check pod status
kubectl describe pod <pod-name> -n qratum

# Check logs
kubectl logs <pod-name> -n qratum

# Check events
kubectl get events -n qratum --sort-by='.lastTimestamp'
```

### Service Unavailable

**Error:**
```
503 Service Unavailable
```

**Solution:**
```bash
# Check endpoints
kubectl get endpoints qratum-api -n qratum

# Check pod readiness
kubectl get pods -n qratum -o wide

# Test connectivity
kubectl exec -it <pod-name> -n qratum -- curl localhost:8000/health
```

### Resource Limits

**Error:**
```
OOMKilled
```

**Solution:**
```yaml
# Increase memory limits in deployment.yaml
resources:
  limits:
    memory: "4Gi"  # Increase from default
    cpu: "2000m"
  requests:
    memory: "2Gi"
    cpu: "500m"
```

## API Issues

### Authentication Failed

**Error:**
```json
{"error": "Unauthorized", "code": 401}
```

**Solution:**
```python
# Check API key
import requests

response = requests.get(
    "http://localhost:8000/health",
    headers={"Authorization": "Bearer your_token"}
)
print(response.status_code)
```

### Rate Limited

**Error:**
```json
{"error": "Rate limit exceeded", "code": 429}
```

**Solution:**
```python
import time

def api_call_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url)
        if response.status_code == 429:
            wait_time = int(response.headers.get('Retry-After', 60))
            time.sleep(wait_time)
            continue
        return response
    raise Exception("Max retries exceeded")
```

### Invalid Input

**Error:**
```json
{"error": "Validation error", "details": "bond_length must be positive"}
```

**Solution:**
```python
# Validate inputs before calling API
def validate_input(bond_length: float) -> None:
    if bond_length <= 0:
        raise ValueError("bond_length must be positive")
    if bond_length > 10:
        raise ValueError("bond_length too large (max 10 Å)")
```

## Performance Issues

### Slow Execution

**Symptom:** Computation takes longer than expected

**Solutions:**

1. **Reduce shots for development:**
   ```python
   config = QuantumConfig(shots=100)  # Instead of 1024
   ```

2. **Enable GPU acceleration:**
   ```python
   config = QuantumConfig(backend_type="cuquantum")
   ```

3. **Use circuit caching:**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=100)
   def get_compiled_circuit(n_qubits, depth):
       return build_circuit(n_qubits, depth)
   ```

### High Memory Usage

**Symptom:** Memory grows unbounded

**Solutions:**

1. **Force garbage collection:**
   ```python
   import gc
   gc.collect()
   ```

2. **Process in batches:**
   ```python
   batch_size = 10
   for i in range(0, len(data), batch_size):
       batch = data[i:i+batch_size]
       process(batch)
       gc.collect()
   ```

## Decision Tree

```
Problem?
    │
    ├── Installation → Check Python version, dependencies
    │
    ├── Runtime Error
    │   ├── Memory → Reduce qubits, increase limits
    │   ├── Convergence → Increase iterations, change optimizer
    │   └── Backend → Check token, network, use simulator
    │
    ├── Docker
    │   ├── Build fails → Clean cache, rebuild
    │   └── Container issues → Check ports, logs
    │
    ├── Kubernetes
    │   ├── Pod issues → kubectl describe, check resources
    │   └── Network → Check endpoints, services
    │
    └── Performance
        ├── Slow → Reduce shots, enable GPU
        └── Memory → GC, batching
```

## Getting Help

If these solutions don't work:

1. **Search existing issues:** [GitHub Issues](https://github.com/robertringler/QRATUM/issues)
2. **Check FAQ:** [FAQ](faq.md)
3. **Open new issue:** Include:
   - QRATUM version
   - Python version
   - Operating system
   - Full error traceback
   - Steps to reproduce
