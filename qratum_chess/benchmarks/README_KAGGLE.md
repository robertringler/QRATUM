# Kaggle Chess Leaderboard Integration

Complete guide for running QRATUM Chess engine benchmarks against Kaggle chess positions and submitting results to Kaggle leaderboards.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
- [API Configuration](#api-configuration)
- [Submission Workflow](#submission-workflow)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Overview

The Kaggle integration enables QRATUM Chess to:

1. **Load benchmark positions** from Kaggle chess competitions
2. **Run engine analysis** using AsymmetricAdaptiveSearch (AAS)
3. **Format results** according to Kaggle submission requirements
4. **Submit results** to Kaggle leaderboards
5. **Track performance** and leaderboard rankings

### Features

✅ Kaggle API authentication and credential management  
✅ Download benchmark positions from competitions  
✅ Run QRATUM engine against test positions  
✅ Format results for CSV/JSON submission  
✅ Automatic submission to Kaggle leaderboards  
✅ Leaderboard ranking retrieval  
✅ Comprehensive error handling  
✅ Dry-run mode for testing  
✅ Integration with existing benchmark suite

## Setup

### 1. Install Dependencies

```bash
pip install kaggle>=1.6.0
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

### 2. Obtain Kaggle API Credentials

#### Option A: Web Interface

1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
2. Scroll to "API" section
3. Click "Create New API Token"
4. Download `kaggle.json` to `~/.kaggle/kaggle.json`

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### Option B: Manual Creation

Create `~/.kaggle/kaggle.json`:

```json
{
  "username": "your_kaggle_username",
  "key": "your_api_key_here"
}
```

Then set permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

#### Option C: Environment Variables

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_api_key_here"
```

### 3. Verify Setup

```bash
# Test Kaggle API access
python3 -c "from kaggle.api.kaggle_api_extended import KaggleApi; api = KaggleApi(); api.authenticate(); print('✓ Authentication successful')"
```

## Usage

### Command Line Interface

#### Basic Benchmark (No Submission)

```bash
# Using Python script
python3 qratum_chess/benchmarks/benchmark_kaggle.py

# Using shell wrapper
./scripts/run_kaggle_benchmark.sh
```

#### Benchmark with Submission

```bash
# Submit to Kaggle
python3 qratum_chess/benchmarks/benchmark_kaggle.py --submit

# Submit with custom message
python3 qratum_chess/benchmarks/benchmark_kaggle.py --submit --message "QRATUM v1.0 - AAS engine"

# Using shell wrapper
./scripts/run_kaggle_benchmark.sh --submit --message "QRATUM v1.0"
```

#### Advanced Options

```bash
# Download fresh data from Kaggle
python3 qratum_chess/benchmarks/benchmark_kaggle.py --download --submit

# Use local data file
python3 qratum_chess/benchmarks/benchmark_kaggle.py --input kaggle_data.json

# Dry run (validate without submitting)
python3 qratum_chess/benchmarks/benchmark_kaggle.py --submit --dry-run

# Custom search parameters
python3 qratum_chess/benchmarks/benchmark_kaggle.py --depth 20 --time-limit 10000

# Use sample positions for testing
python3 qratum_chess/benchmarks/benchmark_kaggle.py --use-sample
# Kaggle Chess Benchmark Integration for QRATUM-Chess

This document describes how to use the Kaggle Chess Leaderboard API integration with QRATUM-Chess benchmarking infrastructure.

## Overview

The Kaggle integration allows you to:
- Download and parse Kaggle chess leaderboard data
- Run QRATUM engine against benchmark positions from Kaggle
- Compare QRATUM's performance with other engines on the leaderboard
- Integrate Kaggle benchmarks into the adversarial gauntlet

## Setup

### Prerequisites

1. **Python 3.8+** with the following packages:
   - All standard QRATUM-Chess dependencies
   - `requests` (optional, for API access)

2. **curl** - For downloading leaderboard data

3. **Kaggle API Credentials** (Optional):
   If you want to use authenticated Kaggle API access:
   
   ```bash
   # Create Kaggle API credentials file
   mkdir -p ~/.kaggle
   echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
   
   Get your API key from: https://www.kaggle.com/settings/account

### Installation

No additional installation is required. The Kaggle integration modules are part of the QRATUM-Chess benchmarking suite.

## Usage

### Quick Start - Shell Script

The easiest way to run a Kaggle benchmark:

```bash
# Run with default settings
./scripts/run_kaggle_benchmark.sh

# Quick mode (reduced depth and positions)
./scripts/run_kaggle_benchmark.sh --quick

# Custom depth
./scripts/run_kaggle_benchmark.sh --depth 20

# Limit number of positions
./scripts/run_kaggle_benchmark.sh --max-positions 50

# Custom output directory
./scripts/run_kaggle_benchmark.sh --output-dir ~/my_results
```

### Python API

#### Basic Usage

```python
from qratum_chess.benchmarks import (
    KaggleConfig,
    KaggleIntegration,
    KaggleSubmission,
)

# Load configuration
config = KaggleConfig()

# Load benchmark positions
integration = KaggleIntegration(config)
positions = integration.create_sample_positions()

# Run benchmark (implement your benchmark logic)
results = run_my_benchmark(positions)

# Submit to Kaggle
submission = KaggleSubmission(config)
result = submission.submit_to_kaggle(
    results,
    message="QRATUM Chess v1.0"
)

print(f"Success: {result.success}")
print(f"Submission ID: {result.submission_id}")
```

#### Download Leaderboard Data

```python
from qratum_chess.benchmarks import KaggleIntegration, KaggleConfig

config = KaggleConfig()
integration = KaggleIntegration(config)

# Download leaderboard
leaderboard = integration.download_leaderboard_data(
    competition_id="chess-engine-leaderboard",
    save_path="leaderboard.json"
)

print(f"Positions: {len(leaderboard.positions)}")
print(f"Top engines: {leaderboard.get_top_engines(n=5)}")
```

#### Load from File

```python
from qratum_chess.benchmarks import KaggleIntegration

integration = KaggleIntegration()
leaderboard = integration.load_leaderboard_from_file("kaggle_data.json")

for position in leaderboard.positions:
    print(f"{position.position_id}: {position.fen}")
```

## API Configuration

### Competition Configuration

```python
from qratum_chess.benchmarks.kaggle_config import KaggleCompetitionConfig

config = KaggleCompetitionConfig(
    competition_id="chess-engine-leaderboard",
    competition_name="Chess Engine Leaderboard",
    submission_format="csv",
    required_fields=["position_id", "best_move", "evaluation"]
)
```

### Submission Format

```python
from qratum_chess.benchmarks.kaggle_config import SubmissionFormat

format_spec = SubmissionFormat(
    format_type="csv",
    headers=["position_id", "best_move", "evaluation", "nodes_searched", "time_ms"]
)

# Validate submission data
is_valid, error = format_spec.validate_submission_data(results)
```

## Submission Workflow

### 1. Prepare Results

Results must be a list of dictionaries with required fields:

```python
results = [
    {
        "position_id": "pos_1",
        "best_move": "e2e4",
        "evaluation": 0.25,
        "nodes_searched": 1000000,
        "time_ms": 150.5
    },
    {
        "position_id": "pos_2",
        "best_move": "d2d4",
        "evaluation": 0.15,
        "nodes_searched": 950000,
        "time_ms": 145.2
    },
    # ... more results
]
```

### 2. Validate Format

```python
from qratum_chess.benchmarks import KaggleSubmission, KaggleConfig

config = KaggleConfig()
submission = KaggleSubmission(config)

is_valid, error = submission.validate_submission(results)
if not is_valid:
    print(f"Validation error: {error}")
```

### 3. Submit

```python
# Submit to Kaggle
result = submission.submit_to_kaggle(
    results,
    message="QRATUM Chess Engine v1.0",
    dry_run=False  # Set to True for testing
)

if result.success:
    print(f"✓ Submitted: {result.submission_id}")
else:
    print(f"✗ Failed: {result.error}")
```

### 4. Check Status

```python
# Wait for scoring
if result.submission_id:
    status = submission.get_submission_status(
        result.submission_id,
        timeout=300  # 5 minutes
    )
    
    print(f"Score: {status.score}")
    print(f"Position: {status.leaderboard_position}")
```

### 5. Display Summary

```python
submission.display_submission_summary(result)
```

Output:
```
Kaggle Submission Summary
✓ Status: SUCCESS
  Submission ID: 12345678
  Message: Submission successful
  Score: 0.8542
  
  Current Leaderboard Standing:
    Position: #15
    Score: 0.8542
```

## Examples

### Example 1: Simple Benchmark

```bash
# Run on sample positions (no Kaggle API needed)
python3 qratum_chess/benchmarks/benchmark_kaggle.py --use-sample
```

### Example 2: Download and Benchmark

```bash
# Download fresh data and run benchmark
python3 qratum_chess/benchmarks/benchmark_kaggle.py --download
```

### Example 3: Benchmark and Submit

```bash
# Complete workflow: download, benchmark, submit
python3 qratum_chess/benchmarks/benchmark_kaggle.py \
    --download \
    --submit \
    --message "QRATUM AAS v1.0 - depth 20" \
    --depth 20 \
    --time-limit 10000
```

### Example 4: Shell Script

```bash
# Simple submission
./scripts/run_kaggle_benchmark.sh --submit

# With custom message
./scripts/run_kaggle_benchmark.sh --submit --message "QRATUM v1.0"

# Dry run test
./scripts/run_kaggle_benchmark.sh --submit --dry-run
```

### Example 5: Integration with Gauntlet

```python
from qratum_chess.benchmarks import AdversarialGauntlet, KaggleIntegration
from qratum_chess.search.aas import AsymmetricAdaptiveSearch

# Initialize engine
engine = AsymmetricAdaptiveSearch()

# Load Kaggle positions
integration = KaggleIntegration()
positions = integration.create_sample_positions()

# Run gauntlet benchmark
gauntlet = AdversarialGauntlet()
match_rate, results = gauntlet.run_kaggle_benchmark(
    engine,
    positions,
    depth=15,
    time_limit_ms=5000
)

print(f"Match rate: {match_rate * 100:.1f}%")
#### Load Kaggle Leaderboard Data

```python
from qratum_chess.benchmarks.kaggle_integration import KaggleLeaderboardLoader

# Load from file
loader = KaggleLeaderboardLoader()
leaderboard = loader.load_from_file("kaggle_chess_leaderboard.json")

# Access leaderboard data
print(f"Benchmark: {leaderboard.benchmark_name}")
print(f"Version: {leaderboard.version}")
print(f"Test positions: {len(leaderboard.test_positions)}")
print(f"Submissions: {len(leaderboard.submissions)}")

# Get top submissions
top_5 = loader.get_top_submissions(n=5, leaderboard=leaderboard)
for sub in top_5:
    print(f"#{sub.rank}: {sub.team_name} - {sub.score}")
```

#### Run Benchmark

```python
from qratum_chess.benchmarks.kaggle_integration import KaggleLeaderboardLoader
from qratum_chess.benchmarks.benchmark_kaggle import KaggleBenchmarkRunner
from qratum_chess.search.aas import AsymmetricAdaptiveSearch

# Initialize components
loader = KaggleLeaderboardLoader()
runner = KaggleBenchmarkRunner()
engine = AsymmetricAdaptiveSearch()

# Load leaderboard
leaderboard = loader.load_from_file("kaggle_chess_leaderboard.json")

# Run benchmark
results = runner.run_benchmark(
    engine,
    leaderboard,
    depth=15,                # Search depth
    time_limit_ms=5000,     # Optional time limit per position
    max_positions=100        # Optional position limit
)

# Generate and print summary
summary = runner.generate_summary(results)
runner.print_summary(summary)

# Save results
output_path = runner.save_results("benchmarks/kaggle_results/", results)
print(f"Results saved to: {output_path}")
```

#### Compare with Leaderboard

```python
# Compare QRATUM with leaderboard
comparison = runner.compare_with_leaderboard(leaderboard, summary)

print(f"QRATUM score: {comparison['qratum_score']*100:.1f}%")
print(f"Estimated rank: #{comparison['estimated_rank']}")
print("\nTop 5 submissions:")
for sub in comparison['top_submissions']:
    print(f"  #{sub['rank']}: {sub['team_name']} - {sub['score']:.4f}")
```

#### Integration with Adversarial Gauntlet

```python
from qratum_chess.benchmarks.gauntlet import AdversarialGauntlet, AdversaryType

# Initialize gauntlet
gauntlet = AdversarialGauntlet(engine_name="QRATUM-Chess")

# Run Kaggle benchmark as adversary
kaggle_score = gauntlet.run_kaggle_benchmark_adversary(
    engine,
    leaderboard_file="kaggle_chess_leaderboard.json",
    depth=15,
    max_positions=50
)

print(f"Kaggle adversary score: {kaggle_score*100:.1f}%")
```

### Download Kaggle Data Manually

```bash
# Download leaderboard data
curl -L -o ~/Downloads/kaggle_chess_leaderboard.json \
  https://www.kaggle.com/api/v1/benchmarks/kaggle/chess/versions/1/leaderboard

# Or use Python
python3 -c "
from qratum_chess.benchmarks.kaggle_integration import download_kaggle_leaderboard
download_kaggle_leaderboard('~/Downloads/kaggle_chess_leaderboard.json')
"
```

## Output Format

### Results JSON Structure

```json
{
  "results": [
    {
      "test_id": "test_001",
      "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "best_move": "e2e4",
      "evaluation": 0.25,
      "depth_reached": 15,
      "nodes_searched": 125000,
      "time_ms": 450.2,
      "expected_move": "e2e4",
      "expected_eval": 0.20,
      "move_matches": true,
      "eval_diff": 0.05,
      "category": "opening",
      "difficulty": "easy"
    }
  ],
  "summary": {
    "total_positions": 50,
    "move_accuracy": 0.82,
    "avg_eval_diff": 0.15,
    "avg_depth": 14.8,
    "avg_nodes": 120000,
    "avg_time_ms": 425.5,
    "total_time_ms": 21275.0,
    "results_by_category": {
      "opening": {"count": 10, "move_accuracy": 0.90},
      "middlegame": {"count": 25, "move_accuracy": 0.80},
      "endgame": {"count": 15, "move_accuracy": 0.75}
    },
    "results_by_difficulty": {
      "easy": {"count": 15, "move_accuracy": 0.95},
      "medium": {"count": 20, "move_accuracy": 0.85},
      "hard": {"count": 15, "move_accuracy": 0.70}
    }
  }
}
```

## Metrics Explanation

### Move Accuracy
Percentage of positions where QRATUM's best move matches the expected/reference move from the Kaggle benchmark. This is the primary metric for comparison.

### Evaluation Difference
Average absolute difference between QRATUM's position evaluation and the expected evaluation. Lower is better.

### Nodes Per Second (NPS)
Calculated as `avg_nodes / (avg_time_ms / 1000)`. Indicates search efficiency.

### Search Depth
Average depth reached during search. Higher depth generally indicates more thorough analysis.

### Category Breakdown
- **Opening**: Standard opening positions
- **Middlegame**: Complex tactical and strategic positions
- **Endgame**: Endgame positions requiring precise calculation

### Difficulty Breakdown
- **Easy**: Straightforward positions with clear best moves
- **Medium**: Positions requiring tactical awareness
- **Hard**: Complex positions with non-obvious solutions

## Integration with Existing Benchmarks

The Kaggle integration is designed to work seamlessly with QRATUM's existing benchmark infrastructure:

```python
from qratum_chess.benchmarks.runner import BenchmarkRunner, BenchmarkConfig

# Create benchmark config with Kaggle gauntlet enabled
config = BenchmarkConfig(
    run_gauntlet=True,
    gauntlet_games=100
)

# Run full benchmark suite
runner = BenchmarkRunner(config)
summary = runner.run(engine, evaluator=None)

# Kaggle adversary will be included if registered in gauntlet
```

## Troubleshooting

### Authentication Errors

**Problem:** `Kaggle credentials not found`

**Solutions:**
1. Verify `~/.kaggle/kaggle.json` exists and has correct format
2. Check file permissions: `chmod 600 ~/.kaggle/kaggle.json`
3. Try using environment variables instead
4. Re-download credentials from Kaggle website

### API Request Failures

**Problem:** `HTTP 401: Unauthorized`

**Solutions:**
1. Check API key is valid and not expired
2. Verify username matches your Kaggle account
3. Ensure API access is enabled in account settings

**Problem:** `HTTP 404: Not Found`

**Solutions:**
1. Verify competition ID is correct
2. Check you have accepted competition rules
3. Ensure competition is still active

**Problem:** `HTTP 429: Too Many Requests`

**Solutions:**
1. Wait a few minutes before retrying
2. Reduce submission frequency
3. Check Kaggle API rate limits

### Submission Errors

**Problem:** `Validation failed: Missing required field`

**Solutions:**
1. Check all required fields are present in results
2. Verify field names match submission format
3. Ensure no null/missing values

**Problem:** `Invalid submission format`

**Solutions:**
1. Verify CSV headers match required format
2. Check for correct data types
3. Ensure no duplicate position IDs

### Data Loading Errors

**Problem:** `No positions found in leaderboard data`

**Solutions:**
1. Use `--use-sample` flag for testing
2. Check JSON file structure
3. Try downloading fresh data
4. Verify competition has test data available

## API Reference

### KaggleConfig

Configuration management for Kaggle API.

```python
config = KaggleConfig(
    credentials_path=None,  # Path to kaggle.json
    use_env_credentials=False,  # Use env vars
    competition_config=None,  # Competition settings
    submission_format=None  # Format specification
)
```

### KaggleIntegration

Load and parse Kaggle leaderboard data.

```python
integration = KaggleIntegration(config=None)

# Load from file
leaderboard = integration.load_leaderboard_from_file("data.json")

# Download from API
leaderboard = integration.download_leaderboard_data(
    competition_id="chess-engine-leaderboard",
    save_path="leaderboard.json"
)

# Create sample positions
positions = integration.create_sample_positions()
```

### KaggleSubmission

Submit results to Kaggle competitions.

```python
submission = KaggleSubmission(config)

# Submit results
result = submission.submit_to_kaggle(
    results,
    message="Submission message",
    dry_run=False
)

# Check status
status = submission.get_submission_status(submission_id, timeout=300)

# Get leaderboard position
position, score = submission.get_leaderboard_position(username=None)

# Display summary
submission.display_submission_summary(result)
```

### KaggleBenchmarkPosition

Represents a single benchmark position.

```python
position = KaggleBenchmarkPosition(
    position_id="pos_1",
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    description="Starting position",
    expected_move="e2e4",
    difficulty=1.0
)

# Access QRATUM Position
qratum_pos = position.position
```

## Security Considerations

⚠️ **Important Security Notes:**

1. **Never commit credentials** to version control
2. `.kaggle/` and `kaggle.json` are in `.gitignore`
3. Set proper file permissions: `chmod 600 ~/.kaggle/kaggle.json`
4. Use environment variables in CI/CD pipelines
5. Rotate API keys regularly
6. Validate all API responses before processing
7. Handle rate limiting gracefully

## Performance Tips

1. **Cache downloaded data** to avoid repeated API calls
2. **Use appropriate depth** for time constraints
3. **Batch positions** for efficient processing
4. **Monitor rate limits** to avoid API throttling
5. **Use dry-run** mode for testing submissions

## Support

For issues or questions:

1. Check this documentation
2. Review troubleshooting section
3. Check Kaggle API documentation: https://github.com/Kaggle/kaggle-api
4. Review QRATUM Chess benchmark docs: `qratum_chess/benchmarks/README_AUTOMATION.md`

## License

QRATUM Chess Kaggle Integration is part of the QRATUM project.
See LICENSE file for details.
### Issue: Download fails

**Problem**: `curl` fails to download leaderboard data

**Solution**:
- Check internet connection
- Verify the Kaggle API endpoint is accessible
- Try downloading manually and specifying the file path
- Use fallback mode (script will use standard positions)

### Issue: No test positions in leaderboard

**Problem**: Leaderboard JSON doesn't contain test positions

**Solution**: The loader will automatically fall back to standard benchmark positions. This is normal for some Kaggle benchmark formats.

### Issue: Engine search fails

**Problem**: Engine throws errors during search

**Solution**:
- Reduce search depth with `--depth 10`
- Add time limit with `--time-limit 3000`
- Check that the engine is properly initialized
- Verify Position objects are valid

### Issue: Slow performance

**Problem**: Benchmark takes too long to run

**Solution**:
- Use `--quick` mode for faster testing
- Reduce `--max-positions` to test fewer positions
- Lower `--depth` to reduce search time
- Add `--time-limit` to cap search time per position

## Advanced Usage

### Custom Position Sets

```python
# Create custom benchmark positions
from qratum_chess.benchmarks.kaggle_integration import (
    KaggleBenchmarkPosition,
    KaggleLeaderboard
)
from qratum_chess.core.position import Position

positions = [
    KaggleBenchmarkPosition(
        fen="your_fen_here",
        position=Position.from_fen("your_fen_here"),
        test_id="custom_001",
        category="custom",
        difficulty="hard"
    )
]

leaderboard = KaggleLeaderboard(
    benchmark_name="custom_benchmark",
    version="1.0",
    test_positions=positions
)

# Run benchmark with custom positions
runner = KaggleBenchmarkRunner()
results = runner.run_benchmark(engine, leaderboard, depth=15)
```

### Batch Processing

```python
# Process multiple leaderboard files
import glob

for leaderboard_file in glob.glob("data/kaggle_*.json"):
    print(f"Processing {leaderboard_file}...")
    leaderboard = loader.load_from_file(leaderboard_file)
    results = runner.run_benchmark(engine, leaderboard, depth=12)
    runner.save_results(f"results/{Path(leaderboard_file).stem}/")
```

## API Reference

### KaggleLeaderboardLoader

- `load_from_file(filepath)`: Load leaderboard from JSON file
- `load_from_dict(data)`: Load leaderboard from dictionary
- `parse_leaderboard(data)`: Parse raw Kaggle API data
- `extract_positions(leaderboard)`: Extract benchmark positions
- `get_top_submissions(n, leaderboard)`: Get top N submissions
- `export_positions_to_fen_file(filepath)`: Export positions as FEN

### KaggleBenchmarkRunner

- `run_benchmark(engine, leaderboard, depth, time_limit_ms, max_positions)`: Run full benchmark
- `generate_summary(results)`: Generate summary statistics
- `save_results(output_dir, results)`: Save results to files
- `print_summary(summary)`: Print summary to console
- `compare_with_leaderboard(leaderboard, summary)`: Compare with leaderboard

### AdversarialGauntlet

- `run_kaggle_benchmark_adversary(engine, leaderboard_file, depth, max_positions)`: Run as adversary test

## Support

For issues, questions, or contributions related to the Kaggle integration:

1. Check the main QRATUM-Chess documentation
2. Review the benchmark runner documentation in `qratum_chess/benchmarks/`
3. Consult the existing test files for usage examples

## Future Enhancements

Planned improvements:
- Direct API authentication for automated downloads
- Support for submitting results to Kaggle (if API allows)
- More sophisticated position selection algorithms
- Multi-engine comparison mode
- Visualization dashboard for results

## License

This integration is part of QRATUM-Chess and follows the same license terms as the main project.
