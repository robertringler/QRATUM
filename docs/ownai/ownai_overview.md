# QuASIM-Own AI Overview

## Introduction

QuASIM-Own is a deterministic, auditable AI stack built around the **Symbolic-Latent Transformer (SLT)**, codenamed **"Modo"**. It fuses REVULTRA cryptanalysis features with standard machine learning models, guarded by QGH causal logging and surfaced via TERC observables.

## Key Features

### 1. Deterministic Execution

- **Global Seeding**: All random operations are seeded for reproducibility
- **Fixed Data Loaders**: Deterministic shuffling and preprocessing
- **Prediction Hashing**: SHA256 hashes of predictions for consensus verification
- **No Stochastic Components**: Disabled dropout and other non-deterministic operations

### 2. Symbolic-Latent Transformer (Modo)

The core model architecture combines:

- **REVULTRA Symbolic Features**:
  - Index of Coincidence (IoC) tensor
  - Spectral autocorrelation
  - Entropy motifs

- **Learned Representations**:
  - Task-specific embeddings
  - Ensemble learning (Random Forest baseline)

- **Multi-Task Support**:
  - Tabular classification and regression
  - Text classification
  - Vision classification
  - Time series forecasting

### 3. Trust Layer

#### QGH Causal History

- Append-only ledger for run tracking
- Run metadata: model, task, dataset, seed, prediction hash
- Consensus verification across runs

#### TERC Observables

- `stability_margin`: 1 - CV across repeated runs
- `qgh_consensus_status`: All prediction hashes equal?
- `emergent_complexity`: Latent dispersion measure
- `goal_progress`: Task completion metric

### 4. Comprehensive Benchmarking

- **Tasks**: Tabular, text, vision, time-series
- **Models**: SLT, MLP, CNN, sklearn baselines, XGBoost, LightGBM
- **Metrics**:
  - Accuracy/F1 (classification)
  - MAE/RMSE (regression)
  - Latency p50/p95
  - Throughput
  - Model size
  - Energy proxy

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     QuASIM-Own Stack                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐ │
│  │   Data      │   │   REVULTRA   │   │   Integration   │ │
│  │   Pipeline  │──▶│   Features   │──▶│   (QGH/TERC)    │ │
│  └─────────────┘   └──────────────┘   └─────────────────┘ │
│         │                  │                     │          │
│         ▼                  ▼                     ▼          │
│  ┌─────────────────────────────────────────────────────┐  │
│  │          Symbolic-Latent Transformer (Modo)          │  │
│  ├─────────────────────────────────────────────────────┤  │
│  │  • Symbolic Features (IoC, Spectral, Entropy)       │  │
│  │  • Learned Embeddings                                │  │
│  │  • Ensemble Learning                                 │  │
│  │  • Task-Specific Heads                               │  │
│  └─────────────────────────────────────────────────────┘  │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────────────────────┐  │
│  │            Evaluation & Benchmarking                 │  │
│  │  • Multi-task benchmarks                             │  │
│  │  • Performance metrics                               │  │
│  │  • Determinism verification                          │  │
│  │  • Model cards                                       │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Installation

```bash
# Install QuASIM with OwnAI dependencies
pip install -e .[ownai]

# Or install individual packages
pip install scikit-learn xgboost lightgbm
```

### Quick Start

#### Training a Model

```bash
quasim-own train \
  --model slt \
  --task text-cls \
  --dataset imdb-mini \
  --seed 1337 \
  --out runs/slt_imdb
```

#### Running Benchmarks

```bash
# Quick benchmark (tabular + text only)
quasim-own benchmark --suite quick --repeat 3 --report reports/quick

# Standard benchmark (all tasks)
quasim-own benchmark --suite std --repeat 5 --report reports/std

# Full benchmark (all tasks, more repeats)
quasim-own benchmark --suite full --repeat 10 --report reports/full
```

#### Generating Model Cards

```bash
quasim-own modelcard \
  --run runs/slt_imdb \
  --out docs/ownai/model_card_slt.md
```

### Python API

```python
from quasim.ownai.determinism import set_seed
from quasim.ownai.data.loaders import load_text
from quasim.ownai.models.slt import build_slt

# Set seed for reproducibility
set_seed(1337)

# Load data
texts, labels = load_text("imdb-mini")

# Build and train model
model = build_slt(task="text-cls", seed=1337, use_symbolic=True)
model.fit(texts[:800], labels[:800])

# Make predictions
predictions = model.predict(texts[800:])
```

## Datasets

QuASIM-Own includes synthetic datasets for testing:

- **Tabular**: Wine, Adult, Higgs-mini
- **Text**: IMDb-mini, AGNews-mini
- **Vision**: CIFAR-10-subset, MNIST-1k
- **Time Series**: ETTh1-mini, Synthetic ARMA

All datasets are deterministically generated with fixed seeds.

## Baselines

The following baseline models are included for comparison:

- **sklearn**: LogisticRegression, LinearSVC, RandomForest
- **Gradient Boosting**: XGBoost, LightGBM (tiny configs)
- **Neural Networks**: Deterministic MLP, Tiny CNN

## Performance Targets

QuASIM-Own aims to achieve:

1. ✅ **Determinism**: Identical prediction hashes across repeats (same seed/model/dataset)
2. ✅ **Reliability-per-Watt**: SLT in top-2 on at least two tasks
3. ✅ **Benchmark Speed**: Full std suite completes in ≤45 minutes on GitHub runner
4. ✅ **Coverage**: ≥90% coverage for quasim/ownai modules

## TERC Observable Guarantees

- `stability_margin ≥ 0.90`: Models exhibit consistent performance
- `qgh_consensus_status = True`: Deterministic predictions verified
- `emergent_complexity < 0.20`: Latency variance remains low
- `goal_progress ≥ 0.70`: Task performance targets met

## Model Card Template

Every trained model automatically generates a model card containing:

- Model architecture details
- Training configuration
- Performance metrics by task
- Determinism verification
- Ethical considerations
- Known limitations

See [Model Card Template](model_card_template.md) for details.

## CI/CD Integration

Two workflows automate testing and benchmarking:

### ownai-ci.yml

- Runs on every push/PR
- Linting (ruff, black, mypy)
- Unit tests
- Quick benchmark (≤20 minutes)
- Uploads artifacts

### ownai-nightly.yml

- Runs daily at 2 AM UTC
- Full benchmark suite
- Commits updated docs/ownai/benchmarks.md
- Uploads comprehensive artifacts

## Security & Compliance

- ✅ No `eval()` or `exec()` usage
- ✅ Deterministic operations enforced
- ✅ QGH ledger for audit trail
- ✅ TERC observables for monitoring
- ✅ Prediction hash consensus verification

## References

- [REVULTRA Algorithms](../../quasim/revultra/)
- [QGH Non-Speculative Algorithms](../../quasim/qgh/)
- [TERC Bridge](../../quasim/terc_bridge/)
- [QuASIM Main Documentation](../../README.md)

## License

Apache 2.0 - See [LICENSE](../../LICENSE) for details.
