# XENON Quantum Bioinformatics v5 - Full Genome Sequencing Pipeline

## Overview

The Full Genome Sequencing Pipeline is a production-ready, end-to-end automated system that integrates all four XENON v5 quantum bioinformatics engines to perform comprehensive genome analysis with deterministic reproducibility.

## Features

### Core Capabilities

1. **Quantum-Enhanced Sequence Alignment** (QuantumAlignmentEngine)
   - Adaptive circuit depth selection based on sequence entropy
   - Classical-quantum equivalence validation (ε < 1e-6)
   - Deterministic reproducibility guarantees

2. **Multi-Omics Information Fusion** (InformationFusionEngine)
   - Partial Information Decomposition (PID) using Williams & Beer framework
   - Conservation constraint enforcement
   - Unique, redundant, and synergistic information quantification

3. **Transfer Entropy Analysis** (TransferEntropyEngine)
   - Directed information flow measurement for time-series data
   - Batched computation for scalability
   - Optimal lag detection

4. **Neural-Symbolic Inference** (NeuralSymbolicEngine)
   - Graph Neural Network (GNN) based variant prediction
   - Symbolic constraint regularization
   - Classical fallback when PyTorch unavailable

### System Features

- **Deterministic Execution**: Global seed management ensures reproducible results
- **Thread-Safe Concurrency**: Multi-threaded execution with deterministic seed derivation
- **Cross-Hardware Compatibility**: Automatic detection and fallback (QPU → GPU → CPU)
- **Persistent Audit Logging**: SQLite-backed audit registry with JSON export
- **Performance Instrumentation**: Memory, GPU utilization, and timing metrics
- **Comprehensive Error Handling**: Robust exception handling with audit trail

## Quick Start

### Basic Usage

```bash
# Run with default settings (seed=42, max_threads=8)
python3 run_genome_sequencing.py

# Run with custom configuration
python3 run_genome_sequencing.py \
  --output-dir results/my_genome \
  --seed 12345 \
  --max-threads 4 \
  --prefer-gpu

# Run with QPU preference (if available)
python3 run_genome_sequencing.py --prefer-qpu
```

### Python API

```python
from xenon.bioinformatics.full_genome_sequencing import (
    FullGenomeSequencingPipeline,
    GenomeSequencingConfig,
)

# Create configuration
config = GenomeSequencingConfig(
    global_seed=42,
    max_threads=8,
    output_dir="results/full_genome",
    prefer_gpu=True,
    prefer_qpu=False,
)

# Initialize and run pipeline
pipeline = FullGenomeSequencingPipeline(config=config)
deployment_report = pipeline.run()

# Check results
print(f"Status: {deployment_report['status']}")
print(f"Sequences aligned: {deployment_report['metrics']['sequences_aligned']}")
print(f"Total duration: {deployment_report['metrics']['total_duration_ms']:.2f} ms")
```

## Output Files

The pipeline generates the following output files in the specified output directory:

### Primary Results

1. **`alignment_result.json`**
   - Sequence alignment results with scores and entropy
   - Circuit depth used for each alignment
   - Classical-quantum equivalence errors

2. **`fusion_result.json`**
   - Multi-omics information decomposition results
   - Unique, redundant, and synergistic information components
   - Conservation constraint validation status

3. **`transfer_entropy.json`**
   - Directed information flow measurements
   - Optimal lag values for each variable pair
   - Validity flags for statistical significance

4. **`functional_predictions.json`**
   - Neural-symbolic inference predictions for variants
   - Graph embeddings and attention weights
   - Constraint violation metrics

### Audit and Reporting

1. **`audit_summary.json`**
   - Aggregated audit statistics
   - Violation counts by type and severity
   - Resolution status

2. **`genome_audit.db`**
   - SQLite database with full audit trail
   - Queryable violation logs with context
   - Reproducibility metadata (seeds, config hashes)

3. **`deployment_report.json`**
   - Complete pipeline execution summary
   - Performance metrics (duration, memory, GPU utilization)
   - Reproducibility validation results
   - Hardware and backend information

4. **`sequencing.log`**
   - Detailed execution log with timestamps
   - Phase transitions and checkpoints
   - Error messages and warnings

## Configuration

### GenomeSequencingConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `global_seed` | int | 42 | Global random seed for all stochastic operations |
| `max_threads` | int | 8 | Maximum number of worker threads |
| `output_dir` | str | "results/full_genome" | Output directory for results |
| `prefer_gpu` | bool | True | Prefer GPU backend if available |
| `prefer_qpu` | bool | False | Prefer QPU backend if available |
| `audit_db_path` | str | "{output_dir}/genome_audit.db" | Path to audit database |

### Engine Configurations

Each engine has its own configuration that can be customized:

```python
from xenon.bioinformatics.quantum_alignment import AlignmentConfig
from xenon.bioinformatics.information_fusion import ConservationConstraints
from xenon.bioinformatics.transfer_entropy import TransferEntropyConfig

config = GenomeSequencingConfig(
    global_seed=42,
    # Alignment configuration
    alignment_config=AlignmentConfig(
        min_circuit_depth=2,
        max_circuit_depth=10,
        equivalence_tolerance=1e-6,
    ),
    # Fusion configuration
    fusion_constraints=ConservationConstraints(
        enforce_non_negativity=True,
        enforce_upper_bound=True,
        tolerance=1e-6,
    ),
    # Transfer entropy configuration
    te_config=TransferEntropyConfig(
        max_lag=5,
        n_bins=10,
        batch_size=1000,
    ),
    # Neural-symbolic inference configuration
    inference_input_dim=10,
    inference_hidden_dim=64,
    inference_output_dim=32,
)
```

## Input Data

### FASTQ Files (Optional)

If you have real FASTQ data:

```bash
python3 run_genome_sequencing.py --fastq data/my_genome.fastq
```

If no FASTQ file is provided, the pipeline generates synthetic sequences for demonstration.

### Synthetic Data Generation

The pipeline automatically generates:

- **Sequences**: 10 synthetic protein sequences (50-200 amino acids)
- **Multi-omics data**: 100 features across genomics, transcriptomics, epigenomics
- **Time-series data**: 100 timepoints for gene expression and protein abundance
- **Variant graph**: 20 nodes, 30 edges with 10-dimensional features

## Reproducibility

### Deterministic Execution

All random operations are seeded deterministically:

```python
# Same seed → identical results
pipeline1 = FullGenomeSequencingPipeline(config=GenomeSequencingConfig(global_seed=42))
result1 = pipeline1.run()

pipeline2 = FullGenomeSequencingPipeline(config=GenomeSequencingConfig(global_seed=42))
result2 = pipeline2.run()

# Results are identical
assert result1["metrics"]["sequences_aligned"] == result2["metrics"]["sequences_aligned"]
```

### Thread-Level Seed Derivation

Multi-threaded execution uses deterministic seed derivation:

```
thread_seed = hash(base_seed:thread_id)
```

This ensures reproducibility even with concurrent execution.

### Equivalence Testing

The pipeline automatically validates classical-quantum equivalence:

```
||Q(seq1, seq2) - C(seq1, seq2)|| < ε  (ε = 1e-6)
```

Any violations are logged to the audit registry.

## Performance

### Benchmarks (Synthetic Data)

On a standard CPU (Intel Xeon, 4 cores):

| Phase | Duration | Operations |
|-------|----------|------------|
| Alignment | ~90 ms | 9 sequence pairs |
| Fusion | ~3 ms | 2 omics layer pairs |
| Transfer Entropy | ~4 ms | 6 variable pairs |
| Inference | ~0.3 ms | 20 variants |
| **Total** | **~110 ms** | **Complete pipeline** |

Memory usage: ~42 MB peak

### Scaling

The pipeline scales efficiently with:

- **Sequence count**: Linear O(n) for alignment
- **Omics layers**: Quadratic O(n²) for pairwise fusion
- **Time-series length**: Linear O(t) for transfer entropy
- **Variant count**: Linear O(v) for inference

## Testing

### Run Tests

```bash
# Run all pipeline tests
pytest xenon/bioinformatics/tests/test_full_genome_sequencing.py -v -o addopts=""

# Run specific test
pytest xenon/bioinformatics/tests/test_full_genome_sequencing.py::TestFullGenomeSequencingPipeline::test_deterministic_execution -v -o addopts=""
```

### Test Coverage

The test suite includes:

- ✅ Pipeline initialization
- ✅ Synthetic data generation (sequences, omics, time-series, graphs)
- ✅ Individual phase execution (alignment, fusion, TE, inference)
- ✅ Full pipeline execution
- ✅ Deterministic reproducibility
- ✅ Hardware detection and backend selection
- ✅ Performance instrumentation
- ✅ Thread-safe execution
- ✅ Output structure validation

## Audit and Compliance

### Audit Registry

The pipeline maintains a comprehensive audit trail:

```python
# Query audit entries
pipeline = FullGenomeSequencingPipeline(config)
pipeline.run()

# Get statistics
stats = pipeline.audit.get_statistics()
print(f"Total violations: {sum(v['count'] for v in stats.values())}")

# Query specific violations
entries = pipeline.audit.query_entries(
    violation_type=ViolationType.CONSERVATION_VIOLATION,
    severity_min=7,
    unresolved_only=True,
)
```

### Violation Types

- **CONSERVATION_VIOLATION**: PID conservation constraints violated
- **EQUIVALENCE_VIOLATION**: Classical-quantum equivalence exceeded tolerance
- **NUMERICAL_INSTABILITY**: Condition number above threshold
- **CONSTRAINT_VIOLATION**: Symbolic constraints violated
- **DETERMINISM_VIOLATION**: Non-deterministic behavior detected

## Hardware Requirements

### Minimum Requirements

- **CPU**: 2+ cores
- **RAM**: 1 GB
- **Python**: 3.10+
- **Storage**: 100 MB for outputs

### Recommended Configuration

- **CPU**: 4+ cores for parallel execution
- **RAM**: 4 GB for larger datasets
- **GPU**: NVIDIA (CUDA) or AMD (ROCm) for acceleration (optional)
- **QPU**: Quantum processor for quantum backend (optional)

### Dependencies

Core:

- `numpy >= 1.24.0`
- `psutil` (for memory monitoring)

Optional:

- `torch >= 2.0.0` (for neural-symbolic inference)
- `pynvml` (for NVIDIA GPU monitoring)
- `qiskit` (for quantum backend)

## Troubleshooting

### Common Issues

**1. PyTorch not available warning**

```
UserWarning: PyTorch not available. Neural-symbolic inference will use classical fallback.
```

Solution: Install PyTorch: `pip install torch`

**2. GPU monitoring not available**

```
UserWarning: GPU monitoring not available (pynvml not found)
```

Solution: Install pynvml: `pip install nvidia-ml-py3`

**3. Memory issues with large datasets**

Solution: Reduce batch size or enable streaming:

```python
config = GenomeSequencingConfig(
    te_config=TransferEntropyConfig(batch_size=500)
)
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

pipeline = FullGenomeSequencingPipeline(config)
pipeline.run()
```

## Architecture

```
FullGenomeSequencingPipeline
├── Hardware Detection (CPU/GPU/QPU)
├── Backend Selection (Quantum/Classical)
├── Seed Management (Global + Thread-level)
├── Phase 1: Quantum Alignment
│   └── QuantumAlignmentEngine
├── Phase 2: Multi-Omics Fusion
│   └── InformationFusionEngine
├── Phase 3: Transfer Entropy
│   └── TransferEntropyEngine
├── Phase 4: Neural-Symbolic Inference
│   └── NeuralSymbolicEngine
├── Audit Summary Generation
│   └── AuditRegistry
├── Reproducibility Validation
└── Results Export (JSON + SQLite)
```

## References

1. **Quantum Alignment**: Adaptive circuit depth based on sequence entropy
2. **Information Fusion**: Williams & Beer (2010) - Partial Information Decomposition
3. **Transfer Entropy**: Schreiber (2000) - Measuring information transfer
4. **Neural-Symbolic**: Graph Neural Networks with constraint regularization

## Support

For issues, questions, or contributions:

- File an issue on GitHub
- See `TASKS_4_10_COMPLETION_REPORT.md` for implementation details
- See `xenon/bioinformatics/ENHANCEMENTS.md` for engine documentation

## License

Apache 2.0 - See LICENSE file for details

## Version

- **Pipeline Version**: 1.0
- **XENON Version**: v5
- **Date**: 2025-12-16

---

**Status**: ✅ Production Ready
