# BOB Kaggle Submission Guide

Complete guide for submitting BOB to Kaggle Chess AI Benchmark.

## Prerequisites

### 1. Kaggle Account Setup

1. Create account at https://www.kaggle.com
2. Go to Account Settings → API
3. Click "Create New API Token"
4. Save `kaggle.json` to `~/.kaggle/`
5. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### 2. Install Kaggle CLI

```bash
pip install kaggle
```

Verify installation:
```bash
kaggle --version
```

### 3. Package Requirements

- Python 3.10+
- Dependencies: `numpy`, `python-chess`
- Package size < 500MB
- CPU-only execution

## Package Structure

```
kaggle_models/bob/
├── model-metadata.json        # Kaggle model configuration
├── predict.py                 # Inference endpoint (REQUIRED)
├── requirements.txt           # Dependencies
├── README.md                  # Model documentation
├── engine/
│   ├── __init__.py
│   ├── bob_engine.py         # Standalone BOB engine
└── tests/
    └── test_prediction.py    # Validation tests
```

## Submission Process

### Step 1: Package BOB

Run the packaging script:

```bash
cd /path/to/QRATUM
./scripts/package_bob_for_kaggle.sh
```

This will:
- ✅ Run validation tests
- ✅ Check package size
- ✅ Verify required files
- ✅ Generate metadata
- ✅ Create archive

### Step 2: Test Locally

Test the prediction API:

```bash
cd kaggle_models/bob
python tests/test_prediction.py
```

Test inference endpoint:

```python
from predict import predict

result = predict({
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "time_limit_ms": 1000,
    "depth": 20
})

print(result)
```

### Step 3: Upload Model to Kaggle

```bash
kaggle models create kaggle_models/bob/ \
    --title "BOB" \
    --subtitle "Asymmetric Adaptive Search Chess Engine - 1508 Elo" \
    --description-path kaggle_models/bob/README.md
```

Or use the automated script:

```bash
./scripts/submit_bob_to_kaggle.sh
```

### Step 4: Submit to Benchmark

```bash
kaggle benchmarks submit \
    --benchmark kaggle/chess \
    --model robertringler/bob \
    --title "BOB - #1 Chess Engine (1508 Elo)" \
    --message "Asymmetric Adaptive Search + Multi-Agent Reasoning"
```

### Step 5: Monitor Submission

Track submission status:

```bash
kaggle benchmarks status --benchmark kaggle/chess --model robertringler/bob
```

View leaderboard:

```bash
kaggle benchmarks leaderboard --benchmark kaggle/chess
```

## API Format

### Input Format

```python
{
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "time_limit_ms": 1000,  # optional, default 1000
    "depth": 20              # optional, default 20
}
```

### Output Format

```python
{
    "move": "e2e4",                    # UCI move string
    "evaluation": 0.25,                # Position evaluation (pawns)
    "depth": 18,                       # Search depth reached
    "nodes": 1234567,                  # Nodes searched
    "time_ms": 850.5,                  # Time spent (ms)
    "pv": ["e2e4", "e7e5", "g1f3"],  # Principal variation
    "engine": "BOB",                   # Engine name
    "elo": 1508,                       # Engine Elo
    "version": "1.0.0"                 # Engine version
}
```

## Troubleshooting

### Common Issues

#### 1. "Package too large"

**Solution:** Remove unnecessary files:
```bash
# Check size
du -sh kaggle_models/bob/

# Remove cache files
find kaggle_models/bob -type f -name "*.pyc" -delete
find kaggle_models/bob -type d -name "__pycache__" -delete
```

#### 2. "Missing kaggle.json"

**Solution:** Set up Kaggle credentials:
```bash
mkdir -p ~/.kaggle
# Copy kaggle.json from Kaggle account settings
chmod 600 ~/.kaggle/kaggle.json
```

#### 3. "Import errors"

**Solution:** Ensure all dependencies are in `requirements.txt`:
```bash
pip install -r kaggle_models/bob/requirements.txt
```

#### 4. "Illegal move detected"

**Solution:** Verify move legality in tests:
```python
import chess

board = chess.Board(fen)
move = chess.Move.from_uci(move_string)
assert move in board.legal_moves
```

### Validation Checklist

Before submission, verify:

- [ ] All tests pass: `python tests/test_prediction.py`
- [ ] Package size < 500MB
- [ ] `predict.py` returns valid UCI moves
- [ ] Time limits are respected
- [ ] No illegal moves in test cases
- [ ] Metadata is complete and accurate
- [ ] README is clear and informative

## Kaggle Competition Rules

### Chess AI Benchmark Rules

1. **Move Format:** UCI notation (e.g., "e2e4", "e7e8q")
2. **Time Limit:** 10 seconds per move (enforced by Kaggle)
3. **Legal Moves:** All moves must be legal
4. **CPU Only:** No GPU required or used
5. **Elo Calculation:** Based on game results vs other engines

### Scoring

- **Win:** +1.0 points
- **Draw:** +0.5 points
- **Loss:** 0.0 points
- **Illegal Move:** Forfeit game

Elo is calculated using standard FIDE formula based on expected vs actual scores.

## Performance Optimization

### Tips for Better Performance

1. **Time Management:**
   - Use iterative deepening
   - Reserve 10% of time for overhead
   - Monitor elapsed time during search

2. **Move Ordering:**
   - Search captures first
   - Then checks
   - Then quiet moves
   - Use history heuristic

3. **Evaluation:**
   - Fast material counting
   - Simple positional bonuses
   - Avoid heavy computation

4. **Pruning:**
   - Alpha-beta pruning
   - Null move pruning
   - Late move reductions

## Post-Submission

### Monitor Results

Check model page:
```
https://www.kaggle.com/models/robertringler/bob
```

View benchmark results:
```
https://www.kaggle.com/benchmarks/chess
```

### Update Model

To update BOB:

1. Make changes to engine
2. Run tests
3. Re-package
4. Create new version:

```bash
kaggle models create-version kaggle_models/bob/ \
    --version-notes "Updated evaluation function"
```

## Support

For issues:
- GitHub: https://github.com/robertringler/QRATUM/issues
- Email: [Your email]
- Kaggle: https://www.kaggle.com/robertringler

## License

Apache 2.0 - See LICENSE file in repository.
