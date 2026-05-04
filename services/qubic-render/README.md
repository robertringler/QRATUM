# QUBIC Render Service

Distributed GPU rendering service for QuASIM tire visualization.

## Features

- FastAPI REST API for render job submission
- WebSocket streaming for real-time rendering
- GPU-accelerated rendering with CUDA support
- Async job queue with worker pool
- Docker containerization with NVIDIA GPU support
- Health check and monitoring endpoints

## Quick Start

### Docker (Recommended)

```bash
# Build and run
docker-compose up --build

# Access API
curl http://localhost:8000/health
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python server.py
```

## API Endpoints

### Health Check

```bash
GET /health
```

Returns service health status and GPU availability.

### GPU Status

```bash
GET /gpu-status
```

Returns detailed GPU information.

### Render Frame

```bash
POST /render/frame
Content-Type: application/json

{
  "width": 1920,
  "height": 1080,
  "backend": "cuda",
  "simulation_id": "sim-001"
}
```

### Render Sequence

```bash
POST /render/sequence
Content-Type: application/json

{
  "width": 1920,
  "height": 1080,
  "backend": "cuda",
  "num_frames": 120
}
```

### Job Status

```bash
GET /render/status/{job_id}
```

Returns job status and progress.

### WebSocket Streaming

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/render');
ws.onmessage = (event) => {
  const frame = JSON.parse(event.data);
  // Process frame
};
```

## Configuration

Environment variables:

- `CUDA_VISIBLE_DEVICES`: GPU device indices (default: "0")
- `NUM_WORKERS`: Number of render workers (default: 1)
- `MAX_QUEUE_SIZE`: Maximum job queue size (default: 100)

## GPU Requirements

- NVIDIA GPU with CUDA 12.0+ support
- nvidia-docker or NVIDIA Container Toolkit
- 4GB+ GPU memory recommended

## Architecture

```
┌─────────────┐
│   FastAPI   │  REST API
└──────┬──────┘
       │
       v
┌─────────────┐
│  Scheduler  │  Job Queue
└──────┬──────┘
       │
       v
┌─────────────┐
│  GPU Worker │  Render Processing
└─────────────┘
```

## Performance

- Single frame rendering: ~50-200ms (GPU)
- Animation rendering: ~30-60 fps
- WebSocket latency: <50ms
- Concurrent jobs: Up to 8 (with multi-GPU)

## Monitoring

Access the API documentation at: `http://localhost:8000/docs`

## License

Apache License 2.0
