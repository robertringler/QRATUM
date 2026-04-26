"""Demo examples for REVULTRA algorithms."""

from __future__ import annotations

import logging
from typing import Any

from quasim.revultra.algorithms import REVULTRAAlgorithms
from quasim.revultra.metrics import find_peaks

logger = logging.getLogger(__name__)

# Kryptos K4 ciphertext (97 characters)
KRYPTOS_K4 = (
    "OBKRUOXOGHULBSOLIFBBWFLRVQQPRNGKSSOTWTQSJQSSEKZZWATJKLUDIAWINFBNYPVTTMZFPK"
    "WGDKZXTJCDIGKUHUAUEKCAR"
)


def demo_kryptos_k4_analysis() -> dict[str, Any]:
    """Run complete REVULTRA analysis on Kryptos K4.

    Returns
    -------
    dict[str, Any]
        Complete analysis results

    Examples
    --------
    >>> results = demo_kryptos_k4_analysis()
    >>> 'complexity' in results
    True
    """

    logger.info("Running REVULTRA analysis on Kryptos K4...")

    rev = REVULTRAAlgorithms()
    results = rev.kryptos_k4_analysis(KRYPTOS_K4)

    # Add peak detection for IoC
    ioc_peaks = find_peaks(results["ioc_tensor"], threshold=0.2)
    results["ioc_peaks"] = ioc_peaks

    logger.info(f"Analysis complete. Complexity score: {results['complexity']['score']:.2f}")
    logger.info(f"Detected {len(ioc_peaks)} IoC peaks at periods: {ioc_peaks}")

    return results


def demo_simple_cipher_analysis(ciphertext: str) -> dict[str, Any]:
    """Run basic REVULTRA analysis on a simple ciphertext.

    Parameters
    ----------
    ciphertext : str
        Ciphertext to analyze

    Returns
    -------
    dict[str, Any]
        Analysis results

    Examples
    --------
    >>> results = demo_simple_cipher_analysis("HELLO" * 10)
    >>> results['frequency']['L'] > results['frequency']['H']
    True
    """

    rev = REVULTRAAlgorithms()

    return {
        "frequency": rev.frequency_analysis(ciphertext),
        "complexity": rev.emergent_complexity_score(ciphertext),
        "chi_squared": rev.chi_squared_test(ciphertext),
        "patterns": rev.memory_recursive_pattern_mining(ciphertext),
    }


def demo_period_detection(ciphertext: str, max_period: int = 20) -> dict[str, Any]:
    """Demonstrate period detection using IoC and autocorrelation.

    Parameters
    ----------
    ciphertext : str
        Ciphertext to analyze
    max_period : int, optional
        Maximum period to test (default: 20)

    Returns
    -------
    dict[str, Any]
        Period detection results
    """

    rev = REVULTRAAlgorithms()

    ioc = rev.index_of_coincidence_tensor(ciphertext, max_period=max_period)
    autocorr = rev.spectral_autocorrelation(ciphertext, max_lag=max_period)

    ioc_peaks = find_peaks(ioc, threshold=0.2)
    autocorr_peaks = find_peaks(autocorr, threshold=0.3)

    return {
        "ioc_tensor": ioc,
        "ioc_peaks": ioc_peaks,
        "autocorrelation": autocorr,
        "autocorr_peaks": autocorr_peaks,
        "likely_periods": sorted(set(ioc_peaks) & set(autocorr_peaks)),
    }


def demo_holographic_analysis(ciphertext: str) -> dict[str, Any]:
    """Demonstrate holographic entropy transform.

    Parameters
    ----------
    ciphertext : str
        Ciphertext to analyze

    Returns
    -------
    dict[str, Any]
        Holographic analysis results
    """

    rev = REVULTRAAlgorithms()

    entropy, surface = rev.holographic_entropy_transform(ciphertext)
    topology = rev.quantum_information_topology(ciphertext)

    return {
        "holographic_entropy": entropy,
        "surface_shape": surface.shape,
        "topology_chars": len(topology),
        "max_amplitude": max(abs(v) for v in topology.values()) if topology else 0.0,
    }


if __name__ == "__main__":
    # Run demo analyses
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("REVULTRA Kryptos K4 Analysis Demo")
    print("=" * 70)
    k4_results = demo_kryptos_k4_analysis()
    print(f"\nComplexity Score: {k4_results['complexity']['score']:.2f}")
    print(f"IoC Peaks at periods: {k4_results['ioc_peaks']}")
    print(f"Pattern count: {len(k4_results['patterns'])}")

    print("\n" + "=" * 70)
    print("Simple Cipher Analysis Demo")
    print("=" * 70)
    simple_results = demo_simple_cipher_analysis("ATTACKATDAWN" * 5)
    print(f"Complexity Score: {simple_results['complexity']['score']:.2f}")
    print(f"Chi-squared: {simple_results['chi_squared']:.2f}")
    print(f"Top 3 patterns: {list(simple_results['patterns'].items())[:3]}")
