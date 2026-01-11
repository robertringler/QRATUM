#!/usr/bin/env python3
"""Demo script showing complete Kaggle integration workflow.

This script demonstrates:
1. Loading sample benchmark positions
2. Running QRATUM engine analysis
3. Formatting results for submission
4. Validating submission format
5. Simulating submission (dry-run)
"""

import sys
from pathlib import Path

# Repository path setup for standalone execution
# Note: For development, consider using: pip install -e .
# This allows proper imports without sys.path manipulation
if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from qratum_chess.benchmarks.kaggle_integration import KaggleIntegration
from qratum_chess.benchmarks.kaggle_submission import KaggleSubmission
from qratum_chess.benchmarks.kaggle_config import KaggleConfig, KaggleCredentials
from qratum_chess.search.aas import AsymmetricAdaptiveSearch, AASConfig
import time


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    """Run the demo."""
    print_header("QRATUM Chess - Kaggle Integration Demo")
    
    # Step 1: Create sample positions
    print_header("Step 1: Load Sample Benchmark Positions")
    integration = KaggleIntegration()
    positions = integration.create_sample_positions()
    print(f"✓ Loaded {len(positions)} sample positions:")
    for pos in positions:
        print(f"  • {pos.position_id}: {pos.description}")
    
    # Step 2: Run engine analysis
    print_header("Step 2: Run QRATUM Engine Analysis")
    config = AASConfig(opening_width=2.0, middlegame_focus=1.5, endgame_precision=2.0)
    engine = AsymmetricAdaptiveSearch(config=config)
    
    results = []
    for i, pos in enumerate(positions, 1):
        print(f"Analyzing position {i}/{len(positions)}: {pos.position_id}... ", end="", flush=True)
        
        start = time.perf_counter()
        best_move, evaluation, stats = engine.search(pos.position, depth=5, time_limit_ms=500)
        elapsed = (time.perf_counter() - start) * 1000
        
        result = {
            "position_id": pos.position_id,
            "best_move": best_move.to_uci() if best_move else "",
            "evaluation": float(evaluation) if evaluation is not None else 0.0,
            "nodes_searched": stats.nodes_searched if stats else 0,
            "time_ms": elapsed,
        }
        results.append(result)
        
        print(f"✓ {result['best_move']} (eval: {result['evaluation']:+.2f})")
    
    # Step 3: Format results
    print_header("Step 3: Format Results for Kaggle Submission")
    
    # Create config without loading credentials
    try:
        mock_config = KaggleConfig(use_env_credentials=False)
    except FileNotFoundError:
        # Mock credentials for demo when real credentials don't exist
        from qratum_chess.benchmarks.kaggle_config import KaggleCompetitionConfig, SubmissionFormat
        mock_config = object.__new__(KaggleConfig)
        mock_config.credentials = KaggleCredentials(username="demo_user", key="demo_key")
        mock_config.competition = KaggleCompetitionConfig()
        mock_config.submission_format = SubmissionFormat()
    
    submission = KaggleSubmission(mock_config)
    
    print("✓ Formatting results as CSV...")
    csv_output = submission._format_as_csv(results)
    print("\nCSV Preview:")
    print(csv_output[:300] + "...")
    
    # Step 4: Validate submission
    print_header("Step 4: Validate Submission Format")
    is_valid, error = submission.validate_submission(results)
    
    if is_valid:
        print("✓ Submission format is valid!")
        print(f"  • Total positions: {len(results)}")
        print(f"  • All required fields present")
        print(f"  • No duplicate position IDs")
    else:
        print(f"✗ Validation failed: {error}")
        return 1
    
    # Step 5: Simulate submission (dry-run)
    print_header("Step 5: Simulate Submission (Dry Run)")
    print("Running dry-run submission (no actual API call)...")
    
    result = submission.submit_to_kaggle(results, message="Demo submission", dry_run=True)
    
    if result.success:
        print("✓ Dry run successful!")
        print(f"  • Message: {result.message}")
        print(f"  • Timestamp: {result.timestamp}")
    else:
        print(f"✗ Dry run failed: {result.error}")
    
    # Summary
    print_header("Summary")
    print("✅ Complete Kaggle integration workflow demonstrated!")
    print("\nKey features:")
    print("  ✓ Position loading and parsing")
    print("  ✓ Engine analysis with AsymmetricAdaptiveSearch")
    print("  ✓ Result formatting for Kaggle submission")
    print("  ✓ Submission validation")
    print("  ✓ Dry-run testing capability")
    
    print("\nTo run with real Kaggle API:")
    print("  1. Set up Kaggle credentials (~/.kaggle/kaggle.json)")
    print("  2. Run: python3 qratum_chess/benchmarks/benchmark_kaggle.py --submit")
    print("  3. Or use: ./scripts/run_kaggle_benchmark.sh --submit")
    
    print("\n" + "=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
