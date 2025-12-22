# QuASIM-Own AI

**Deterministic, Auditable AI with Symbolic-Latent Transformer "Modo"**

## Quick Start

```bash
# Install dependencies
pip install -e .[ownai]

# Train a model
quasim-own train --model slt --task text-cls --dataset imdb-mini --seed 1337

# Run benchmarks
quasim-own benchmark --suite quick --repeat 3

# Generate model card
quasim-own modelcard --run runs/slt_imdb --out docs/model_card.md
```

## Features

✅ **Deterministic Execution**: Reproducible predictions with hash verification  
✅ **Symbolic-Latent Transformer**: Fuses REVULTRA features with learned embeddings  
✅ **Multi-Task Support**: Tabular, text, vision, time-series  
✅ **Comprehensive Benchmarking**: Performance, latency, energy metrics  
✅ **TERC Observables**: Stability, consensus, complexity tracking  
✅ **QGH Causal History**: Append-only audit ledger  
✅ **Model Cards**: Auto-generated documentation  

## Architecture

```
quasim/ownai/
├── configs.py              # Configuration management
├── determinism.py          # Seeding, hashing, verification
├── data/                   # Data loaders and preprocessing
│   ├── loaders.py
│   ├── preprocess.py
│   └── schemas.py
├── revultra/               # REVULTRA feature extraction
│   └── feats.py
├── models/                 # Model architectures
│   ├── slt.py             # Symbolic-Latent Transformer
│   ├── mlp.py             # Deterministic MLP
│   ├── cnn_tiny.py        # Tiny CNN
│   └── heads.py           # Task heads
├── train/                  # Training utilities
│   └── metrics.py
├── eval/                   # Evaluation and benchmarking
│   ├── benchmark.py
│   └── reporting.py
└── integration/            # TERC/QGH integration
    ├── terc_observables.py
    ├── qgh_hooks.py
    └── model_card.py
```

## CLI Commands

### Train

```bash
quasim-own train \
  --model slt \
  --task text-cls \
  --dataset imdb-mini \
  --seed 1337 \
  --out runs/slt_imdb
```

### Evaluate

```bash
quasim-own eval --run runs/slt_imdb --metrics all
```

### Benchmark

```bash
# Quick: tabular + text (~10 minutes)
quasim-own benchmark --suite quick --repeat 3

# Standard: all tasks (~30 minutes)
quasim-own benchmark --suite std --repeat 5

# Full: all tasks + more repeats (~60 minutes)
quasim-own benchmark --suite full --repeat 10
```

### Export

```bash
quasim-own export \
  --run runs/slt_imdb \
  --format json \
  --out models/slt_imdb.json
```

### Model Card

```bash
quasim-own modelcard \
  --run runs/slt_imdb \
  --out docs/model_card_slt.md
```

## Python API

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

## Testing

```bash
# Run all tests
pytest tests/ownai/ -v

# Run specific test module
pytest tests/ownai/test_determinism.py -v

# Run with coverage
pytest tests/ownai/ --cov=quasim.ownai --cov-report=html
```

## CI/CD

### ownai-ci.yml

- Triggers on push/PR to ownai code
- Linting (ruff, black, mypy)
- Unit tests
- Quick benchmark (~20 minutes)
- Artifact uploads

### ownai-nightly.yml

- Runs daily at 2 AM UTC
- Full benchmark suite
- Commits docs/ownai/benchmarks.md
- Archives results (30 days)

## Datasets

All datasets are synthetic and deterministically generated:

- **Tabular**: Wine, Adult, Higgs-mini
- **Text**: IMDb-mini, AGNews-mini
- **Vision**: CIFAR-10-subset, MNIST-1k
- **Time Series**: ETTh1-mini, Synthetic ARMA

## Baselines

Included baseline models:

- **sklearn**: LogisticRegression, LinearSVC, RandomForest
- **XGBoost**: Tiny configuration
- **LightGBM**: Tiny configuration
- **Neural**: Deterministic MLP, Tiny CNN

## Performance

Demo results on synthetic data:

- Text classification (IMDb-mini): 99.0% accuracy
- Tabular classification (Wine): 85-90% accuracy
- Vision classification (MNIST-1k): 80-85% accuracy

All results deterministic across runs with same seed.

## Documentation

- [Overview](../../docs/ownai/ownai_overview.md): Comprehensive documentation
- [Model Card Template](../../docs/ownai/model_card_template.md): Auto-generated cards
- [Benchmarks](../../docs/ownai/benchmarks.md): Latest benchmark results (auto-updated)

## Contributing

1. Install dev dependencies: `pip install -e .[dev,ownai]`
2. Run tests: `pytest tests/ownai/`
3. Run linter: `ruff check quasim/ownai`
4. Format code: `black quasim/ownai`
5. Type check: `mypy quasim/ownai`

## License

Apache 2.0 - See [LICENSE](../../LICENSE) for details.
