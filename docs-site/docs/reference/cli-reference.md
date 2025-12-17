# CLI Reference

Command-line interface documentation for QRATUM tools.

## Available Commands

| Command | Description |
|---------|-------------|
| `quasim-hcal` | Hardware calibration |
| `quasim-tire` | Tire simulation CLI |
| `qunimbus` | Distributed orchestration |
| `qubic-viz` | Visualization tools |
| `xenon` | Biological intelligence |
| `qnx` | QNX integration |
| `qstack` | Stack management |

---

## quasim-hcal

Hardware Calibration and Detection CLI.

### Usage

```bash
quasim-hcal [OPTIONS] COMMAND [ARGS]
```

### Commands

#### detect

Detect available hardware accelerators.

```bash
quasim-hcal detect [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--json` | Output as JSON | `false` |
| `--verbose` | Verbose output | `false` |

**Example:**

```bash
$ quasim-hcal detect --json
{
  "gpus": [
    {"index": 0, "name": "NVIDIA A100", "memory_gb": 40, "driver": "535.86.10"}
  ],
  "cpu": {"cores": 64, "threads": 128, "model": "AMD EPYC 7742"},
  "memory_gb": 512
}
```

#### calibrate

Run hardware calibration.

```bash
quasim-hcal calibrate [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--device` | Target device | `auto` |
| `--output` | Output file | `calibration.json` |

---

## quasim-tire

Tire simulation command-line interface.

### Usage

```bash
quasim-tire [OPTIONS] COMMAND [ARGS]
```

### Commands

#### simulate

Run tire simulation.

```bash
quasim-tire simulate [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--profile` | Tire profile file | Required |
| `--conditions` | Operating conditions | Required |
| `--output` | Output directory | `./results` |
| `--seed` | Random seed | `None` |

**Example:**

```bash
quasim-tire simulate \
  --profile configs/tires/goodyear_p255.json \
  --conditions configs/conditions/highway.json \
  --output ./tire_results \
  --seed 42
```

#### validate

Validate simulation results.

```bash
quasim-tire validate --results ./tire_results
```

---

## qunimbus

Distributed orchestration CLI for QuNimbus.

### Usage

```bash
qunimbus [OPTIONS] COMMAND [ARGS]
```

### Commands

#### deploy

Deploy to cluster.

```bash
qunimbus deploy [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--cluster` | Target cluster | Required |
| `--manifest` | Deployment manifest | `deploy.yaml` |
| `--namespace` | Kubernetes namespace | `qratum` |
| `--dry-run` | Preview without deploying | `false` |

**Example:**

```bash
qunimbus deploy \
  --cluster prod-west \
  --manifest infra/k8s/deployment.yaml \
  --namespace qratum
```

#### status

Check deployment status.

```bash
qunimbus status --cluster prod-west
```

#### scale

Scale deployment.

```bash
qunimbus scale --cluster prod-west --replicas 5
```

---

## qubic-viz

Visualization tools CLI.

### Usage

```bash
qubic-viz [OPTIONS] COMMAND [ARGS]
```

### Commands

#### render

Render visualization.

```bash
qubic-viz render [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input data file | Required |
| `--output` | Output file | `output.png` |
| `--format` | Output format | `png` |
| `--width` | Image width | `1920` |
| `--height` | Image height | `1080` |

**Example:**

```bash
qubic-viz render \
  --input results.json \
  --output visualization.png \
  --width 3840 \
  --height 2160
```

#### animate

Create animation.

```bash
qubic-viz animate \
  --input simulation_frames/ \
  --output simulation.mp4 \
  --fps 30
```

---

## xenon

Biological intelligence subsystem CLI.

### Usage

```bash
xenon [OPTIONS] COMMAND [ARGS]
```

### Commands

#### analyze

Run biological analysis.

```bash
xenon analyze [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input data | Required |
| `--model` | Analysis model | `default` |
| `--output` | Output file | `analysis.json` |

#### simulate

Run biological simulation.

```bash
xenon simulate \
  --config configs/bio/genome.yaml \
  --output results/
```

---

## Make Targets

The project includes make targets for common operations.

### Build and Test

```bash
# Format code
make fmt

# Run linters
make lint

# Run tests
make test

# Build components
make build
```

### Demos

```bash
# SpaceX demo
make spacex-demo

# All vertical demos
make demo-all-verticals

# Specific vertical
make demo-aerospace
make demo-healthcare
make demo-finance
```

### Docker

```bash
# Run full stack
make run-full-stack

# Stop full stack
make stop-full-stack
```

### Benchmarks

```bash
# Run benchmarks
make bench

# Generate video artifacts
make video
```

### Audit

```bash
# Full repository audit
make audit

# Phase VIII autonomy tests
make autonomy-test
```

---

## Environment Variables

### Global Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `QRATUM_LOG_LEVEL` | Log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |
| `QRATUM_SEED` | Default random seed | `None` |
| `QRATUM_CONFIG` | Config file path | `~/.qratum/config.yaml` |

### Backend Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `JAX_PLATFORM_NAME` | JAX backend (`cpu`, `cuda`, `rocm`) | `cpu` |
| `CUDA_VISIBLE_DEVICES` | GPU devices to use | All |
| `OMP_NUM_THREADS` | OpenMP threads | Auto |

### API Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `QRATUM_API_HOST` | API host | `0.0.0.0` |
| `QRATUM_API_PORT` | API port | `8000` |
| `QRATUM_API_WORKERS` | Worker processes | `4` |

---

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Configuration error |
| 4 | Runtime error |
| 5 | Timeout |
