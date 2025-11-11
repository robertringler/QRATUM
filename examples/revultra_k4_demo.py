#!/usr/bin/env python3
"""REVULTRA Kryptos K4 Demo

Demonstrates complete REVULTRA analysis on the famous Kryptos K4 cipher.
"""

import json

import numpy as np

from quasim.revultra.algorithms import REVULTRAAlgorithms
from quasim.revultra.metrics import find_peaks

# Kryptos K4 ciphertext (97 characters)
KRYPTOS_K4 = (
    "OBKRUOXOGHULBSOLIFBBWFLRVQQPRNGKSSOTWTQSJQSSEKZZWATJKLUDIAWINFBNYPVTTMZFPK"
    "WGDKZXTJCDIGKUHUAUEKCAR"
)


def main():
    """Run K4 analysis demo."""
    print("=" * 70)
    print("REVULTRA Analysis: Kryptos K4")
    print("=" * 70)
    print(f"\nCiphertext ({len(KRYPTOS_K4)} chars):")
    print(KRYPTOS_K4)
    print()

    # Initialize with deterministic RNG
    rev = REVULTRAAlgorithms(rng=np.random.default_rng(42))

    # 1. Frequency Analysis
    print("1. Frequency Analysis")
    print("-" * 40)
    freqs = rev.frequency_analysis(KRYPTOS_K4)
    top_chars = sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:5]
    for char, freq in top_chars:
        print(f"  {char}: {freq:5.2f}%")
    print()

    # 2. Complexity Score
    print("2. Emergent Complexity")
    print("-" * 40)
    complexity = rev.emergent_complexity_score(KRYPTOS_K4)
    print(f"  Overall Score: {complexity['score']:.2f}")
    print(f"  Entropy: {complexity['entropy']:.4f}")
    print(f"  Pattern Density: {complexity['pattern_density']:.4f}")
    print(f"  IoC Variance: {complexity['ioc_variance']:.4f}")
    print()

    # 3. Period Detection (IoC)
    print("3. Index of Coincidence Analysis")
    print("-" * 40)
    ioc = rev.index_of_coincidence_tensor(KRYPTOS_K4, max_period=20)
    ioc_peaks = find_peaks(ioc, threshold=0.2)
    print(f"  IoC Peaks at periods: {[p+1 for p in ioc_peaks]}")
    print("  Top 3 IoC values:")
    ioc_sorted = sorted(enumerate(ioc), key=lambda x: x[1], reverse=True)[:3]
    for period, value in ioc_sorted:
        print(f"    Period {period+1}: {value:.4f}")
    print()

    # 4. Spectral Autocorrelation
    print("4. Spectral Autocorrelation")
    print("-" * 40)
    autocorr = rev.spectral_autocorrelation(KRYPTOS_K4, max_lag=30)
    autocorr_peaks = find_peaks(autocorr[1:], threshold=0.3)  # Skip lag 0
    print(f"  Autocorrelation peaks at lags: {[p+1 for p in autocorr_peaks]}")
    print()

    # 5. Pattern Mining
    print("5. Recurring Patterns")
    print("-" * 40)
    patterns = rev.memory_recursive_pattern_mining(KRYPTOS_K4, min_pattern_len=3)
    if patterns:
        top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        for pattern, count in top_patterns:
            print(f"  '{pattern}': {count} occurrences")
    else:
        print("  No recurring patterns found")
    print()

    # 6. Holographic Entropy
    print("6. Holographic Entropy Transform")
    print("-" * 40)
    entropy, surface = rev.holographic_entropy_transform(KRYPTOS_K4)
    print(f"  Holographic Entropy: {entropy:.4f}")
    print(f"  Surface Shape: {surface.shape}")
    print()

    # 7. Chi-Squared Test
    print("7. Chi-Squared Test (vs English)")
    print("-" * 40)
    chi2 = rev.chi_squared_test(KRYPTOS_K4)
    print(f"  Chi-squared: {chi2:.2f}")
    print("  (Lower values indicate closer to English)")
    print()

    # Export complete analysis
    results = {
        "ciphertext": KRYPTOS_K4,
        "length": len(KRYPTOS_K4),
        "frequency": freqs,
        "complexity": complexity,
        "ioc_peaks": [p + 1 for p in ioc_peaks],
        "autocorr_peaks": [p + 1 for p in autocorr_peaks],
        "patterns": patterns,
        "holographic_entropy": entropy,
        "chi_squared": chi2,
    }

    output_file = "kryptos_k4_analysis.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)

    print(f"✓ Complete analysis exported to {output_file}")
    print()
    print("=" * 70)
    print("Key Findings:")
    print("=" * 70)
    print(f"• Complexity score: {complexity['score']:.2f}/100")
    print(f"• {len(patterns)} recurring patterns detected")
    print(f"• Potential periods: {[p+1 for p in ioc_peaks] if ioc_peaks else 'None detected'}")
    print(f"• Chi-squared: {chi2:.2f} (compare to ~26 for random)")
    print()


if __name__ == "__main__":
    main()
