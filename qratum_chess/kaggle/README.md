# QRATUM-Chess Kaggle Competition Integration

**Objective: Sustained Leaderboard Supremacy**

This is the fully automated pipeline for dominating Kaggle chess competitions. Not just participation — **supremacy**.

## Overview

Complete end-to-end automation:
1. Downloads real Kaggle competition data
2. Runs actual QRATUM engine (no mocks, no stubs)
3. Generates compliant Kaggle submissions
4. Submits automatically
5. Tracks ranking, performance drift, and novel motifs
6. Optimizes hyperparameters based on leaderboard feedback

## System Constraints

- ✓ **No mock data** - All benchmarks on real Kaggle datasets
- ✓ **No heuristic shortcuts** - All moves from QRATUM core search + tri-modal cortex fusion
- ✓ **Reproducibility enforced** - Every submission hash-tracked with config snapshot
- ✓ **Security hardened** - Kaggle API keys read only from `~/.kaggle/kaggle.json`

## Quick Start

### 1. Setup Kaggle Credentials

```bash
# Create Kaggle API credentials
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Get your API key from: https://www.kaggle.com/settings/account

### 2. Install Kaggle CLI

```bash
pip install kaggle
```

### 3. Run Competition Pipeline

```bash
# Full automated pipeline
./scripts/run_kaggle_benchmark.sh \
  --competition chess-positions \
  --submit \
  --optimize \
  --track \
  --no-mock

# Or use Python directly
python3 run_qratum_chess_kaggle.py \
  --competition chess-positions \
  --enable-trimodal \
  --enable-novelty-pressure \
  --disable-randomness \
  --submit \
  --optimize \
  --track
```

## Architecture

### Module Structure

```
qratum_chess/kaggle/
├── __init__.py          # Main exports
├── client.py            # Kaggle API client with authentication
├── submission.py        # Submission formatting and validation
├── leaderboard.py       # Leaderboard polling and rank tracking
└── config.py            # Configuration management

run_qratum_chess_kaggle.py  # Main pipeline execution
scripts/run_kaggle_benchmark.sh  # CLI wrapper
```

### Pipeline Flow

```
1. Authentication
   ↓
2. Download Competition Data
   ↓
3. Load Test Positions (FEN)
   ↓
4. Initialize QRATUM Engine
   ├─ AsymmetricAdaptiveSearch
   ├─ Tri-modal Cortex Fusion
   └─ Novelty Pressure Enabled
   ↓
5. Run Engine on All Positions
   ├─ Generate Best Moves
   ├─ Calculate Evaluations
   └─ Track Node Counts
   ↓
6. Format Submission
   ├─ Convert to CSV
   ├─ Validate Schema
   ├─ Check Move Legality
   └─ Inject Metadata
   ↓
7. Submit to Kaggle
   ↓
8. Poll Leaderboard
   ├─ Wait for Score
   ├─ Track Rank Changes
   └─ Update History
   ↓
9. Optimize Parameters
   ├─ Analyze Rank Delta
   ├─ Adjust Novelty Pressure
   ├─ Tune Search Depth
   └─ Save Feedback
```

## Usage

### CLI Script

```bash
# Basic submission
./scripts/run_kaggle_benchmark.sh --competition chess-comp --submit

# Full optimization pipeline
./scripts/run_kaggle_benchmark.sh \
  --competition chess-comp \
  --depth 20 \
  --enable-novelty-pressure \
  --novelty-pressure 0.6 \
  --submit \
  --optimize \
  --track

# Quick test (no submission)
./scripts/run_kaggle_benchmark.sh --quick
```

### Python API

```python
from qratum_chess.kaggle import KaggleClient, SubmissionFormatter, LeaderboardTracker
from qratum_chess.search.aas import AsymmetricAdaptiveSearch

# Initialize client
client = KaggleClient()

# Download competition
dataset = client.download_competition_data(
    "chess-competition",
    "data/kaggle/current"
)

# Run engine and format submission
# (see run_qratum_chess_kaggle.py for full example)

# Submit
client.submit_competition(
    "chess-competition",
    "submission.csv",
    message="QRATUM v1.0 | Hash: abc123"
)

# Track rank
tracker = LeaderboardTracker("chess-competition", "username")
entry = tracker.wait_for_score(max_wait=300)
tracker.print_status()
```

## Output Format

### Submission CSV

```csv
# QRATUM_HASH=abc123...
# ENGINE_VERSION=AsymmetricAdaptiveSearch
# NOVELTY_PRESSURE=0.5
# CORTEX_WEIGHTS=tactical=0.33,strategic=0.33,conceptual=0.34
# SEARCH_DEPTH=15
# TIMESTAMP=1704067200.0
id,move
pos_001,e2e4
pos_002,d2d4
...
```

### Leaderboard Status

```
══════════════════════════════════════════════════════════
🏆 Kaggle Leaderboard Status - chess-competition
══════════════════════════════════════════════════════════
Current Rank: #17
Best Rank: #12
Δ Rank: +5 ⬆
Current Score: 0.8524
Δ Score: +0.0042
Total Submissions: 8
Rank Volatility: 3.2
Score Progression: +0.0053/submission
══════════════════════════════════════════════════════════
```

## Hyperparameter Optimization

Feedback-driven parameter tuning:

### Parameters Optimized
- **Novelty Pressure Ω(a)**: Adjusted based on rank delta
- **Search Depth**: Increased for rank improvements, decreased for regressions  
- **Cortex Weights**: Gradient-based optimization (future enhancement)

### Optimization Algorithm

```python
# Rank improvement → increase novelty
if delta_rank > 0:
    novelty_pressure *= 1.05
    if search_depth < 20:
        search_depth += 1

# Rank regression → decrease novelty
elif delta_rank < 0:
    novelty_pressure *= 0.95
    if search_depth > 10:
        search_depth -= 1
```

### Feedback Storage

Parameters and results saved to `benchmarks/kaggle_feedback.json`:

```json
{
  "submissions": [
    {
      "timestamp": 1704067200.0,
      "rank": 17,
      "score": 0.8524,
      "parameters": {
        "novelty_pressure": 0.5,
        "search_depth": 15
      }
    }
  ],
  "best_rank": 12,
  "best_score": 0.8566,
  "best_parameters": {...}
}
```

## Anti-Overfitting Safeguards

- **Cross-competition validation**: Test on multiple competitions
- **Torture suite correlation**: Penalize leaderboard-optimized strategies that reduce torture accuracy
- **Parameter drift limits**: Block changes exceeding recursive depth κ(Mₜ)
- **Illegal move blocking**: Zero-tolerance validation before submission

## Certification Gates

Submissions blocked unless:
- `hash_hit_rate ≥ 0.90`
- `illegal_moves = 0`
- `novelty_score_mean ≥ 0.4`
- `benchmark_drift ≤ 3%`

## Success Definition

QRATUM-Chess achieves **leaderboard domination** when:

1. ✓ Rank stabilizes within **top-1%** for 10 consecutive submissions
2. ✓ Novel motif emergence rate **> Stockfish baseline**
3. ✓ ELO confidence interval **< ±12**

## Real-Time Dashboard

Live CLI output during execution:

```
🏆 Kaggle Rank: #17
📈 Δ Rank: +5
🔥 Novel Motifs This Run: 12
⚙ Cortex Drift: Stable
⏱ Submission Latency: 8.2s
```

## Security

- API keys read from `~/.kaggle/kaggle.json` (600 permissions required)
- No keys in environment variables (unless explicitly set)
- Subprocess calls use list format (no shell injection)
- Timeouts on all network operations
- SSL verification enabled by default

## Troubleshooting

### Kaggle CLI Not Found

```bash
pip install kaggle
```

### Credentials Not Found

Check `~/.kaggle/kaggle.json` exists and has correct format:
```json
{"username":"your_username","key":"your_api_key"}
```

Permissions should be 600:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Competition Download Fails

- Verify competition ID is correct
- Check you've accepted competition rules on Kaggle website
- Ensure API key has proper permissions

### Submission Fails Validation

- Check for illegal moves in output
- Verify CSV format matches competition requirements
- Ensure all required columns are present

## Advanced Usage

### Custom Engine Configuration

```python
from qratum_chess.search.aas import AsymmetricAdaptiveSearch, AASConfig

config = AASConfig(
    opening_width=2.5,
    middlegame_focus=1.8,
    endgame_precision=2.5,
    entropy_gradient_learning_rate=0.15
)

engine = AsymmetricAdaptiveSearch(config=config)
```

### Batch Competition Processing

```bash
for comp in chess-comp-1 chess-comp-2 chess-comp-3; do
    ./scripts/run_kaggle_benchmark.sh \
        --competition $comp \
        --submit \
        --optimize \
        --track
done
```

## Performance Metrics

Expected performance on modern hardware:
- **Nodes/sec**: 50,000 - 100,000
- **Time per position**: 2-5 seconds (depth 15)
- **Submission latency**: 5-10 seconds
- **Leaderboard poll**: 10-30 seconds

## Future Enhancements

Planned improvements:
- Multi-engine ensemble submissions
- Real-time cortex weight optimization
- Automated competition discovery
- Performance prediction models
- Advanced motif extraction and tracking

## License

Part of QRATUM-Chess. Follows the same license as the main project.

---

**Remember**: The objective is not participation — it is **sustained leaderboard supremacy**.
