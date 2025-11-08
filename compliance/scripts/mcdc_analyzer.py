#!/usr/bin/env python3
"""MC/DC (Modified Condition/Decision Coverage) Analyzer for DO-178C Level A"""

import json
from pathlib import Path


def analyze_mcdc_coverage(coverage_file):
    """Analyze MC/DC coverage from test results"""

    print("Analyzing MC/DC Coverage...")

    # Load coverage data
    if Path(coverage_file).exists():
        with open(coverage_file) as f:
            coverage = json.load(f)
    else:
        coverage = {"mcdc": 0, "statement": 0, "branch": 0}

    # DO-178C Level A Requirements
    requirements = {"mcdc": 100, "statement": 100, "branch": 100}

    results = {"compliant": True, "details": {}}

    for metric, required in requirements.items():
        actual = coverage.get(metric, 0)
        results["details"][metric] = {
            "required": required,
            "actual": actual,
            "compliant": actual >= required,
        }
        if actual < required:
            results["compliant"] = False

    # Print results
    print("\nDO-178C Level A Coverage Results:")
    print("=" * 50)
    for metric, data in results["details"].items():
        status = "✓" if data["compliant"] else "✗"
        print(f"{status} {metric.upper()}: {data['actual']}% (required: {data['required']}%)")

    print("=" * 50)
    print(f"Overall: {'✓ COMPLIANT' if results['compliant'] else '✗ NON-COMPLIANT'}")

    return results


if __name__ == "__main__":
    import sys

    coverage_file = sys.argv[1] if len(sys.argv) > 1 else "coverage.json"
    analyze_mcdc_coverage(coverage_file)
