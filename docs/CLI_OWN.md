# QuASIM-Own CLI - Deterministic AI with Symbolic-Latent Transformer

The `quasim-own` command-line interface provides tools for training, evaluating, benchmarking, and exporting models using the Symbolic-Latent Transformer (SLT) architecture with deterministic reproducibility.

## Installation

Install the QuASIM package to get access to the CLI:

```bash
pip install -e .
```

## Commands

### 1. Train a Model

Train a model on a specific dataset with deterministic seeding.

```bash
quasim-own train --model slt --task text-cls --dataset imdb-mini --seed 1337 --out runs/default
```

**Options:**
- `--model` - Model type: `slt` (default), `mlp`, `cnn`
- `--task` - Task type (e.g., `text-cls`)
- `--dataset` - Dataset name (e.g., `imdb-mini`)
- `--seed` - Random seed for reproducibility (default: 1337)
- `--out` - Output directory (default: `runs/default`)

**Example output:**
```
Training slt on text-cls/imdb-mini with seed=1337
✅ Training complete!
   Primary metric: 0.8523
   Results saved to: runs/default/results.json
```

**Features:**
- Deterministic training with seed control
- Automatic result logging to QGH ledger
- Prediction hash generation for verification
- Results saved in JSON format

### 2. Evaluate a Model

Evaluate a trained model from a run directory.

```bash
quasim-own eval --run runs/default --metrics all
```

**Options:**
- `--run` - Run directory containing training results (required)
- `--metrics` - Metrics to compute (default: `all`)

**Example output:**
```
📊 Evaluation Results for runs/default
   Model: slt
   Task: text-cls
   Primary Metric: 0.8523
   Secondary Metric: 0.9012
```

### 3. Run Benchmarks

Run comprehensive benchmark suites to evaluate model performance across multiple runs.

```bash
quasim-own benchmark --suite std --repeat 3 --cpu-only --report reports/own
```

**Options:**
- `--suite` - Suite type: `quick`, `std` (default), `full`
- `--repeat` - Number of repeats (default: 3)
- `--cpu-only` - CPU-only mode (default: True)
- `--report` - Report output directory (default: `reports/own`)

**Available suites:**
- `quick` - Fast validation suite
- `std` - Standard benchmark suite
- `full` - Comprehensive full benchmark suite

**Example output:**
```
🚀 Running std benchmark suite with 3 repeats...
   Running in CPU-only mode
   ✅ CSV saved: reports/own/results_20251214_115200.csv
   ✅ JSON saved: reports/own/results_20251214_115200.json
   ✅ Markdown report: reports/own/summary_20251214_115200.md
   ✅ TERC observables: reports/own/terc_observables_20251214_115200.json

📈 Benchmark Summary:
   Total runs: 15
   Stability margin: 0.985
   QGH consensus: ✅
   Goal progress: 0.923
```

**Generated artifacts:**
- CSV file with detailed results
- JSON file with structured data
- Markdown summary report
- TERC observables for validation

### 4. Export a Model

Export a trained model to various formats.

```bash
quasim-own export --run runs/default --format json --out exports/model.json
```

**Options:**
- `--run` - Run directory (required)
- `--format` - Export format: `json` (default), `onnx`
- `--out` - Output file path (required)

**Supported formats:**
- `json` - JSON format with model metadata
- `onnx` - ONNX format for deployment (future)

**Example output:**
```
Exporting model from runs/default to exports/model.json (format: json)
✅ Exported to: exports/model.json
```

### 5. Generate Model Card

Generate a model card documenting model metadata and performance.

```bash
quasim-own modelcard --run runs/default --out exports/model_card.md
```

**Options:**
- `--run` - Run directory (required)
- `--out` - Output model card path (required)

**Example output:**
```
✅ Model card generated: exports/model_card.md
```

**Model card includes:**
- Model architecture details
- Training parameters
- Performance metrics
- Dataset information
- Reproducibility information (seed, hash)

## Workflow Example

Complete workflow for training and benchmarking:

```bash
# 1. Train a model
quasim-own train --model slt --task text-cls --dataset imdb-mini --seed 42 --out runs/exp1

# 2. Evaluate the trained model
quasim-own eval --run runs/exp1 --metrics all

# 3. Run comprehensive benchmarks
quasim-own benchmark --suite std --repeat 5 --report reports/exp1

# 4. Export the model
quasim-own export --run runs/exp1 --format json --out exports/exp1_model.json

# 5. Generate model card
quasim-own modelcard --run runs/exp1 --out exports/exp1_card.md
```

## Determinism and Reproducibility

QuASIM-Own is designed for deterministic reproducibility:

- **Seed Control**: All random operations are seeded
- **Prediction Hashing**: SHA-256 hash of predictions for verification
- **QGH Ledger**: Automatic logging to quantum graph history
- **TERC Observables**: Compliance observables for validation tiers

### Verifying Reproducibility

To verify that two runs produce identical results:

```bash
# Run 1
quasim-own train --model slt --dataset imdb-mini --seed 42 --out runs/verify1

# Run 2 (same seed)
quasim-own train --model slt --dataset imdb-mini --seed 42 --out runs/verify2

# Compare prediction hashes - should be identical
cat runs/verify1/results.json | grep prediction_hash
cat runs/verify2/results.json | grep prediction_hash
```

## Integration with QuASIM

QuASIM-Own integrates with the QuASIM platform:

- **QGH Ledger**: Automatic logging of runs and results
- **TERC Validation**: Observable emission for compliance tiers
- **Deterministic Seeds**: Compatible with QuASIM seed management
- **Model Cards**: Documentation for certification workflows

## Troubleshooting

### Run directory not found
```bash
❌ Run directory not found: runs/missing
```
**Solution:** Train a model first to create the run directory.

### Results file missing
```bash
❌ Results file not found: runs/default/results.json
```
**Solution:** Ensure training completed successfully and results were saved.

### CPU-only mode warning
If you see warnings about JAX platform:
```bash
export JAX_PLATFORM_NAME=cpu  # Linux/Mac
$env:JAX_PLATFORM_NAME="cpu"  # Windows PowerShell
```

## Support

For issues or questions, refer to:
- Main README: [../README.md](../README.md)
- OwnAI Documentation: [ownai/ownai_overview.md](ownai/ownai_overview.md)
