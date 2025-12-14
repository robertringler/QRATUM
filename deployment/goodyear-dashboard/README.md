# Goodyear Quantum Tire Visualization Suite

Real-time QuASIM integration with Claude 4.5 Opus for interactive Gen-6 tire compound analysis and sustainability metrics visualization.

## Status: ✅ Complete

### Components
- [x] **Dockerfile** - NVIDIA CUDA 12.2 base with Python 3.11, all dependencies
- [x] **launch_dashboard.py** - Full Dash application with:
  - 3D Molecular Visualization
  - Performance Radar Charts
  - Sustainability Gauges (CO₂e, Bio-content, Recycled, Circular Economy)
  - Optimization Trace Plots
  - Material Properties Table
  - Claude Opus AI Insights Integration
  - Live QuASIM API connection
  - Prometheus metrics
- [x] **requirements.txt** - All Python dependencies
- [x] **k8s/pvc.yaml** - 500GB PersistentVolumeClaim for simulation datasets
- [x] **k8s/deployment.yaml** - 2 GPU replicas with secrets, health probes, PodDisruptionBudget
- [x] **k8s/service.yaml** - LoadBalancer with multi-cloud annotations (AWS/GCP/Azure)
- [x] **k8s/secrets.yaml.example** - Template for API key secrets

## Quick Start (Local Development)

```bash
# Build the Docker image
docker build -t goodyear-dashboard:latest .

# Run locally (without GPU)
docker run -p 8050:8050 \
  -e QUASIM_API_KEY=your_key \
  -e CLAUDE_OPUS_API_KEY=your_key \
  goodyear-dashboard:latest

# Open dashboard
open http://localhost:8050
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `QUASIM_API_URL` | QuASIM API endpoint | No (default: https://api.quasim.io/v1) |
| `QUASIM_API_KEY` | QuASIM API authentication key | Yes |
| `CLAUDE_OPUS_API_KEY` | Anthropic Claude API key | Yes |
| `DASH_HOST` | Dashboard host binding | No (default: 0.0.0.0) |
| `DASH_PORT` | Dashboard port | No (default: 8050) |
| `DASH_DEBUG` | Enable debug mode | No (default: false) |

## Features

### 3D Molecular Visualization
- Interactive Three.js/Plotly 3D scatter plot
- Atom coloring by element (C, H, O, Si, S, N)
- Bond visualization with order-based thickness
- Rotation, zoom, and pan controls

### Performance Metrics
- Radar chart comparing Gen-6 compound vs baseline
- Metrics: Wet Grip, Dry Grip, Rolling Resistance, Durability, Thermal, Comfort
- EU tire label grade display

### Sustainability Analytics
- CO₂e per tire gauge (kg)
- Bio-content percentage
- Recycled content percentage
- Circular economy score
- Lifecycle analysis

### Quantum Optimization Trace
- Energy convergence plot
- Gradient norm visualization
- Parameter evolution (polymer ratio, filler loading, crosslink density)
- Quantum fidelity metrics

### Claude Opus AI Insights
- Natural language queries about compound optimization
- Real-time AI analysis of simulation data
- Technical recommendations for manufacturing

## Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace goodyear

# Create secrets (copy and edit secrets.yaml.example first)
cp k8s/secrets.yaml.example k8s/secrets.yaml
# Edit k8s/secrets.yaml with your actual API keys (base64 encoded)
kubectl apply -f k8s/secrets.yaml

# Apply manifests in order
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Get LoadBalancer URL
kubectl get svc goodyear-dashboard -n goodyear
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐     ┌─────────────────┐                   │
│  │   Pod (GPU)     │     │   Pod (GPU)     │                   │
│  │  Dashboard      │     │  Dashboard      │                   │
│  │  Replica 1      │     │  Replica 2      │                   │
│  └────────┬────────┘     └────────┬────────┘                   │
│           │                       │                             │
│           └───────────┬───────────┘                             │
│                       │                                         │
│              ┌────────▼────────┐                                │
│              │  LoadBalancer   │                                │
│              │  Service :80    │                                │
│              └────────┬────────┘                                │
│                       │                                         │
│         ┌─────────────┴─────────────┐                          │
│         │                           │                          │
│  ┌──────▼──────┐            ┌───────▼───────┐                  │
│  │   PVC       │            │   Secrets     │                  │
│  │   500GB     │            │   API Keys    │                  │
│  │   Dataset   │            │               │                  │
│  └─────────────┘            └───────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐  ┌─────▼─────┐  ┌────▼────┐
    │ QuASIM  │  │  Claude   │  │  User   │
    │   API   │  │   API     │  │ Browser │
    └─────────┘  └───────────┘  └─────────┘
```

## Simulation Configuration

- **Simulation ID**: GY-SUSTAIN-2030-PILOT
- **Compound ID**: GY-SUST-2030-0005
- **Live Update Interval**: 5 seconds

## License

Apache 2.0 - See [LICENSE](../../LICENSE)
