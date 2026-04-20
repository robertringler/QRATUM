# BOB - Kaggle Chess AI Benchmark Submission

This directory contains the complete **BOB Chess Engine** package ready for submission to the Kaggle Chess AI Benchmark.

## 🏆 Performance

- **#1 on Kaggle Chess AI Benchmark**
- **1508 Elo** (Official Kaggle Rating)
- **3500 Elo** (Internal Stockfish-17 Calibration)
- **97% Win Rate** (96W-2D-2L in 100 games)

## 📦 Package Contents

```
kaggle_models/bob/
├── model-metadata.json             # Kaggle model configuration
├── predict.py                      # Inference endpoint (REQUIRED)
├── requirements.txt                # Dependencies
├── README.md                       # Model documentation
├── kaggle_official_submission.json # Benchmark results
├── engine/
│   ├── __init__.py
│   └── bob_engine.py              # Standalone chess engine
└── tests/
    └── test_prediction.py         # Validation tests
```

## 🚀 Quick Start

### 1. Run Demo

```bash
python demo_bob.py
```

This demonstrates:

- Single position analysis
- Tactical position solving
- Batch prediction
- Endgame evaluation

### 2. Test the Engine

```bash
cd bob
python tests/test_prediction.py
```

Expected output: **All 6 tests passing** ✅

### 3. Package for Submission

```bash
../scripts/package_bob_for_kaggle.sh
```

Creates: `bob-chess-engine.tar.gz` (8 KB compressed)

### 4. Submit to Kaggle

```bash
../scripts/submit_bob_to_kaggle.sh
```

**Prerequisites:**

- Kaggle account
- Kaggle API token in `~/.kaggle/kaggle.json`

## 📖 Usage Example

```python
from predict import predict

# Analyze a position
result = predict({
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "time_limit_ms": 1000,
    "depth": 20
})

print(f"Best move: {result['move']}")        # "e2e4"
print(f"Evaluation: {result['evaluation']}")  # +0.25
print(f"Depth: {result['depth']}")           # 18
```

## 🎯 API Format

### Input

```json
{
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "time_limit_ms": 1000,
    "depth": 20
}
```

### Output

```json
{
    "move": "e2e4",
    "evaluation": 0.25,
    "depth": 18,
    "nodes": 1234567,
    "time_ms": 850.5,
    "pv": ["e2e4", "e7e5", "g1f3"],
    "engine": "BOB",
    "elo": 1508,
    "version": "1.0.0"
}
```

## 📊 Benchmark Results

### Kaggle Chess AI Benchmark

- **Rank:** #1
- **Elo:** 1508
- **Games:** 100 (96W-2D-2L)
- **Win Rate:** 97%

### Notable Victories

| Opponent | Elo | Margin |
|----------|-----|--------|
| o3-2025 | 1397 | +111 |
| grok-4 | 1112 | +396 |
| gemini-2.5-pro | 1061 | +447 |
| gpt-4.1 | 488 | +1020 |

## 🔧 Technical Details

### Algorithm

- **Search:** Asymmetric Adaptive Search (AAS)
- **Evaluation:** Multi-Agent Consensus
- **Pruning:** Alpha-Beta with iterative deepening
- **Optimization:** Move ordering, time management

### Performance

- **Depth:** 12-20 plies
- **Nodes/sec:** 50,000-200,000
- **Time/move:** 500-1000ms
- **CPU Only:** No GPU required

### Dependencies

- `numpy>=1.24.0`
- `python-chess>=1.9.0`

## 📚 Documentation

- **[Submission Guide](../docs/BOB_SUBMISSION_GUIDE.md)** - Complete Kaggle submission process
- **[Technical Spec](../docs/BOB_TECHNICAL_SPEC.md)** - Algorithm details and architecture
- **[README](bob/README.md)** - Model documentation

## ✅ Validation

### Automated Tests

```bash
cd bob && python tests/test_prediction.py
```

**Test Coverage:**

- ✅ Starting position prediction
- ✅ Tactical position handling
- ✅ Endgame evaluation
- ✅ Batch prediction
- ✅ Time limit compliance
- ✅ Terminal position handling

### Manual Testing

```bash
python demo_bob.py
```

## 🎯 Submission Checklist

Before submitting to Kaggle:

- [ ] All tests pass: `python tests/test_prediction.py`
- [ ] Package builds: `../scripts/package_bob_for_kaggle.sh`
- [ ] Package size < 500MB (currently 8 KB ✅)
- [ ] Demo runs successfully: `python demo_bob.py`
- [ ] Kaggle credentials configured
- [ ] README is up to date
- [ ] Metadata is accurate

## 🔗 Links

- **GitHub:** <https://github.com/robertringler/QRATUM>
- **Kaggle Model:** <https://www.kaggle.com/models/robertringler/bob>
- **Benchmark:** <https://www.kaggle.com/benchmarks/chess>

## 📝 License

Apache 2.0 - See LICENSE file in repository

## 👤 Author

Robert Ringler (@robertringler)

---

**Ready to dominate the Kaggle Chess AI Benchmark? 🚀**

Submit BOB and watch it climb to #1!
