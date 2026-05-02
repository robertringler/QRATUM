# Quick Start (5 Minutes)

Get QRATUM running locally in just 5 minutes using Docker Compose.

## Prerequisites

- Docker Engine 20.10+ or Docker Desktop
- Docker Compose 2.0+
- 4GB RAM minimum

## Step 1: Clone the Repository

```bash
git clone https://github.com/robertringler/QRATUM.git
cd QRATUM
```

## Step 2: Start All Services

```bash
docker-compose up --build
```

This command starts:

- **Backend API** on `http://localhost:8000`
- **Frontend Dashboard** on `http://localhost:8080`

## Step 3: Verify Installation

Open your browser and navigate to:

- **Dashboard**: [http://localhost:8080](http://localhost:8080)
- **API Health**: [http://localhost:8000/health](http://localhost:8000/health)
- **Metrics**: [http://localhost:8000/metrics](http://localhost:8000/metrics)

## Step 4: Run a Test Computation

Click the "Run Kernel" button in the dashboard, or use curl:

```bash
curl -X POST http://localhost:8000/kernel \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "scale": 2.0}'
```

Expected output:

```json
{
  "state_vector": [...],
  "energy": -1.137,
  "convergence": true
}
```

## Step 5: Stop Services

```bash
docker-compose down
```

---

## What's Next?

<div class="grid cards" markdown>

-   :material-book-open:{ .lg .middle } __First Simulation__

    ---

    Learn to run VQE for H₂ molecule simulation
    
    [:octicons-arrow-right-24: First Simulation](first-simulation.md)

-   :material-code-braces:{ .lg .middle } __API Reference__

    ---

    Explore the full API documentation
    
    [:octicons-arrow-right-24: API Reference](../reference/api-reference.md)

-   :material-kubernetes:{ .lg .middle } __Kubernetes Deployment__

    ---

    Deploy to production Kubernetes clusters
    
    [:octicons-arrow-right-24: Kubernetes Tutorial](../tutorials/kubernetes-deployment.md)

</div>

## Troubleshooting

### Port Already in Use

If port 8000 or 8080 is already in use:

```bash
# Check what's using the port
lsof -i :8000

# Or use different ports in docker-compose.yml
```

### Backend Not Responding

Check the logs:

```bash
docker-compose logs backend
```

### Need More Help?

- See [Troubleshooting Guide](../advanced/troubleshooting.md)
- Check [FAQ](../advanced/faq.md)
- Open a [GitHub Issue](https://github.com/robertringler/QRATUM/issues)
