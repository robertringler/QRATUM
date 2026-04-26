# QRATUM-Chess Automated Benchmarking

Complete automation system for running the Stage IV benchmark suite with automated motif extraction and comprehensive reporting.

## Quick Start

### Basic Usage

Run the full benchmark suite with default settings:

```bash
python run_full_benchmark.py
```

### Quick Mode

For faster iteration during development:

```bash
python run_full_benchmark.py --quick
```

This reduces:
- Torture suite depth: 15 → 8
- Resilience iterations: 10 → 3

### With Certification and Motif Extraction

```bash
python run_full_benchmark.py --certify --extract-motifs
```

### Custom Output Directory

```bash
python run_full_benchmark.py --output-dir /path/to/results
```

### GPU Acceleration

```bash
# Auto-detect GPU (default)
python run_full_benchmark.py

# Force GPU usage
python run_full_benchmark.py --gpu

# CPU only
python run_full_benchmark.py --cpu-only
```

## Command Line Options

### Mode Options

- `--quick`, `-q` - Run in quick mode with reduced iterations
- `--certify` - Run Stage III certification verification
- `--extract-motifs` - Enable motif extraction (default: enabled)
- `--no-extract-motifs` - Disable motif extraction

### Output Options

- `--output-dir DIR`, `-o DIR` - Base output directory (default: `benchmarks/auto_run`)

### Hardware Options

- `--gpu` - Force GPU acceleration if available
- `--cpu-only` - Disable GPU, run on CPU only

### Benchmark Component Toggles

- `--no-performance` - Skip performance benchmarks
- `--no-torture` - Skip strategic torture suite
- `--no-elo` - Skip Elo certification
- `--no-resilience` - Skip resilience tests
- `--no-telemetry` - Disable telemetry capture

### Parameters

- `--torture-depth N` - Search depth for torture suite (default: 15)
- `--resilience-iterations N` - Number of resilience test iterations (default: 10)

### Advanced Options

- `--no-checkpoint` - Disable checkpoint/resume capability
- `--verbose`, `-v` - Enable verbose output with detailed logging
- `--version` - Show version information

## Architecture

### Components

```
run_full_benchmark.py (CLI Wrapper)
    ↓
qratum_chess/benchmarks/auto_benchmark.py (Orchestrator)
    ↓
    ├─→ Environment Verification
    ├─→ BenchmarkRunner (existing)
    ├─→ MotifExtractor (new)
    └─→ Report Generation
```

### Data Flow

```
1. Environment Check
   ↓
2. Initialize Engine (AsymmetricAdaptiveSearch)
   ↓
3. Run Benchmarks
   ├─→ Performance Metrics
   ├─→ Torture Suite
   ├─→ Elo Certification
   ├─→ Resilience Tests
   └─→ Telemetry Capture
   ↓
4. Stage III Certification (optional)
   ↓
5. Motif Extraction
   ├─→ Parse Telemetry
   ├─→ Classify Patterns
   └─→ Score Novelty
   ↓
6. Report Generation
   ├─→ JSON (structured data)
   ├─→ CSV (tabular data)
   ├─→ HTML (visual reports)
   └─→ PGN (chess games)
```

## Output Structure

After running the benchmark, results are organized in a timestamped directory:

```
benchmarks/auto_run/YYYYMMDD_HHMMSS/
├── benchmark_results.json       # Complete benchmark results
├── benchmark_metrics.csv         # Metrics in tabular format
├── benchmark_report.html         # Visual benchmark report
├── certification_status.json     # Stage III certification status
├── environment_info.json         # System environment information
├── motifs/                       # Motif extraction results
│   ├── motif_catalog.json        # Complete motif catalog
│   ├── motifs_summary.csv        # Motif summary table
│   ├── motifs_report.html        # Visual motif report
│   ├── tactical_motifs.pgn       # Tactical motifs in PGN
│   ├── strategic_motifs.pgn      # Strategic motifs in PGN
│   ├── opening_motifs.pgn        # Opening motifs in PGN
│   ├── endgame_motifs.pgn        # Endgame motifs in PGN
│   └── conceptual_motifs.pgn     # Conceptual motifs in PGN
├── telemetry/                    # Telemetry data
│   └── telemetry_data.json       # Complete telemetry capture
└── logs/                         # Logs (if configured)
    └── benchmark.log
```

## Output Formats

### JSON Files

Structured data suitable for programmatic analysis:

- **benchmark_results.json**: Complete benchmark metrics and results
- **certification_status.json**: Stage III certification details
- **motif_catalog.json**: All discovered motifs with metadata
- **telemetry_data.json**: Raw telemetry data

### CSV Files

Tabular data for spreadsheet analysis:

- **benchmark_metrics.csv**: Benchmark metrics with pass/fail status
- **motifs_summary.csv**: Motif summary with key attributes

### HTML Reports

Visual reports for human review:

- **benchmark_report.html**: Comprehensive benchmark visualization
- **motifs_report.html**: Motif catalog with board positions

### PGN Files

Chess game format for analysis in chess software:

- **{type}_motifs.pgn**: Motifs grouped by type (tactical, strategic, etc.)

## Configuration

### Programmatic Usage

```python
from qratum_chess.benchmarks.auto_benchmark import (
    AutoBenchmark,
    AutoBenchmarkConfig,
)
from qratum_chess.benchmarks.runner import BenchmarkConfig

# Create configuration
config = AutoBenchmarkConfig(
    quick_mode=False,
    certify=True,
    extract_motifs=True,
    output_dir="custom/output",
    gpu_enabled=True,
    cpu_only=False,
    checkpoint_enabled=True,
    benchmark_config=BenchmarkConfig(
        run_performance=True,
        run_torture=True,
        run_elo=True,
        run_resilience=True,
        run_telemetry=True,
        torture_depth=15,
        resilience_iterations=10,
    ),
)

# Run benchmark
auto = AutoBenchmark(config)
results = auto.run_full_suite()

# Access results
print(f"Success: {results['success']}")
print(f"Motifs found: {len(results['motifs'])}")
print(f"Output path: {results['output_path']}")
```

## Motif Extraction

### What are Motifs?

Motifs are novel chess patterns discovered by QRATUM-Chess that diverge from standard engine play. They are classified into:

- **Tactical**: Combinations, traps, deflections
- **Strategic**: Pawn structures, piece coordination
- **Opening**: Novel opening lines
- **Endgame**: Conversion techniques, zugzwang
- **Conceptual**: Abstract patterns requiring conceptual understanding

### Extraction Process

1. **Telemetry Analysis**: Parse captured telemetry for novelty signals
2. **Pattern Detection**: Identify positions with high novelty pressure Ω(a)
3. **Divergence Analysis**: Find moves diverging from engine databases
4. **Classification**: Categorize by type and game phase
5. **Scoring**: Calculate novelty scores (0.0-1.0)
6. **Cortex Analysis**: Analyze which cortex (tactical/strategic/conceptual) activated

### Thresholds

Default thresholds for motif detection:

- **Novelty Threshold**: 0.6 (minimum novelty score)
- **Divergence Threshold**: 0.5 (minimum divergence from engines)
- **Min Cortex Activation**: 0.3 (minimum activation weight)

These can be adjusted in `MotifExtractor` initialization.

## Stage III Certification

Stage III promotion requires:

- **≥75% winrate** vs Stockfish-NNUE baseline
- **≥70% winrate** vs Lc0-class neural networks
- **Novel motif emergence** confirmed

Enable with `--certify` flag:

```bash
python run_full_benchmark.py --certify
```

Results are saved to `certification_status.json`.

## Checkpoint/Resume

Long-running benchmarks support checkpoint/resume:

```python
# Checkpoints are automatically saved to:
# benchmarks/auto_run/checkpoints/checkpoint_YYYYMMDD_HHMMSS.json

# Resume from checkpoint
auto = AutoBenchmark(config)
checkpoint_data = auto.load_checkpoint(Path("path/to/checkpoint.json"))
# Continue execution...
```

Disable with `--no-checkpoint` flag.

## Troubleshooting

### GPU Not Detected

If GPU is available but not detected:

```bash
# Check PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# Force GPU usage
python run_full_benchmark.py --gpu
```

### Missing Dependencies

If dependencies are missing:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- json (built-in)
- csv (built-in)

### Benchmark Failures

Enable verbose logging:

```bash
python run_full_benchmark.py --verbose
```

Check logs in output directory:
```
benchmarks/auto_run/YYYYMMDD_HHMMSS/logs/benchmark.log
```

### Memory Issues

For systems with limited memory:

```bash
# Use quick mode
python run_full_benchmark.py --quick

# Disable telemetry
python run_full_benchmark.py --no-telemetry

# Reduce torture depth
python run_full_benchmark.py --torture-depth 8
```

## Performance Tips

### Fast Iteration

```bash
python run_full_benchmark.py --quick --no-resilience --no-elo
```

### Full Comprehensive Run

```bash
python run_full_benchmark.py --certify --extract-motifs --verbose
```

### Production Benchmarking

```bash
python run_full_benchmark.py \
  --certify \
  --extract-motifs \
  --output-dir /production/benchmarks \
  --gpu \
  --torture-depth 20 \
  --resilience-iterations 20
```

## Integration

### CI/CD Pipeline

```yaml
# Example GitHub Actions workflow
- name: Run QRATUM-Chess Benchmark
  run: |
    python run_full_benchmark.py --quick --certify
    
- name: Upload Results
  uses: actions/upload-artifact@v3
  with:
    name: benchmark-results
    path: benchmarks/auto_run/
```

### Scheduled Benchmarking

```bash
#!/bin/bash
# cron job: 0 0 * * 0  (weekly)

cd /path/to/QRATUM
python run_full_benchmark.py \
  --certify \
  --extract-motifs \
  --output-dir /benchmarks/weekly
```

## API Reference

See source code documentation in:
- `qratum_chess/benchmarks/auto_benchmark.py`
- `qratum_chess/benchmarks/motif_extractor.py`
- `qratum_chess/benchmarks/telemetry.py`

## Support

For issues or questions:
- Check the troubleshooting section above
- Review example usage in this document
- Examine output logs with `--verbose`
- See `docs/MOTIF_EXTRACTION.md` for motif details
