# QRATUM Full Stack Quick Start

This guide shows you how to quickly run the full QRATUM Autonomous Systems platform locally using Docker Compose.

**Note:** QRATUM (formerly QuASIM) is the world's first Certifiable Quantum-Classical Convergence (CQCC) platform.

## Prerequisites

- Docker Engine 20.10+ or Docker Desktop
- Docker Compose 2.0+
- 4GB RAM minimum

## Running the Full Stack

### Option 1: Docker Compose (Recommended for Local Development)

1. **Start all services:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   - Frontend Dashboard: http://localhost:8080
   - Backend API: http://localhost:8000
   - Health Check: http://localhost:8000/health
   - Metrics: http://localhost:8000/metrics

3. **Stop all services:**
   ```bash
   docker-compose down
   ```

### Option 2: Run Services Individually

#### Backend Only

```bash
cd autonomous_systems_platform/services/backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
export JAX_PLATFORM_NAME=cpu  # On Windows: $env:JAX_PLATFORM_NAME="cpu"
python app.py
```

Backend will be available at http://localhost:8000

#### Frontend Only

```bash
cd autonomous_systems_platform/services/frontend
python -m http.server 8080
```

Frontend will be available at http://localhost:8080

**Note:** When running services individually, make sure the backend is running before accessing the frontend, and update the backend URL in client.js if needed.

## Testing the Application

Once the services are running:

1. Open http://localhost:8080 in your browser
2. Click the "Run Kernel" button
3. You should see output with state_vector, energy, and convergence values

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /kernel` - Run autonomous systems kernel
  - Body: `{"seed": 0, "scale": 1.0}`
- `GET /metrics` - Prometheus metrics

## Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Run kernel computation
curl -X POST http://localhost:8000/kernel \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "scale": 2.0}'

# View metrics
curl http://localhost:8000/metrics
```

## Troubleshooting

### Port Already in Use

If you get a "port already in use" error:

```bash
# Check what's using the port (Linux/Mac)
lsof -i :8000
lsof -i :8080

# Stop the conflicting service or use different ports in docker-compose.yml
```

### Backend Not Responding

Check the backend logs:
```bash
docker-compose logs backend
```

### Frontend Can't Connect to Backend

Ensure:
1. Backend is healthy: `curl http://localhost:8000/health`
2. CORS is not blocking requests (shouldn't be an issue for localhost)
3. Check browser console for errors

## Next Steps

- See [README.md](README.md) for architecture overview
- See [autonomous_systems_platform/README.md](autonomous_systems_platform/README.md) for deployment options
- Review Kubernetes manifests in `autonomous_systems_platform/infra/k8s/`
- Explore Terraform infrastructure in `infra/terraform/`

## Development

To make changes and see them reflected:

1. Edit code in `autonomous_systems_platform/services/`
2. Rebuild and restart:
   ```bash
   docker-compose up --build
   ```

For faster iteration during development, consider running services individually (Option 2) so you can restart just the service you're working on.
