"""REVULTRA cryptanalytic algorithms implementation.

This module implements quantum-inspired cryptanalytic methods for analyzing
ciphertexts using holographic entropy, temporal embeddings, and spectral analysis.
"""

from __future__ import annotations

import logging
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from quasim.revultra.typing import NpArray

logger = logging.getLogger(__name__)

# Numerical stability epsilon
EPS = 1e-10


@dataclass(slots=True)
class REVULTRAConfig:
    """Configuration for REVULTRA algorithms.

    Parameters
    ----------
    hsrq : float
        Holographic surface resonance quantum parameter (default: 0.15)
    teq : float
        Temporal embedding quantum parameter (default: 1.0)
    etq : float
        Entropy transform quantum parameter (default: 1.3)
    itq : float
        Information topology quantum parameter (default: 0.9)
    qcorr : float
        Quantum correlation parameter (default: 0.73)
    memq : float
        Memory quantum parameter (default: 0.88)
    """

    hsrq: float = 0.15
    teq: float = 1.0
    etq: float = 1.3
    itq: float = 0.9
    qcorr: float = 0.73
    memq: float = 0.88


class REVULTRAAlgorithms:
    """REVULTRA cryptanalytic algorithms suite.

    This class provides quantum-inspired methods for analyzing ciphertexts,
    including holographic entropy transforms, temporal embeddings, and
    spectral autocorrelation analysis.

    Parameters
    ----------
    cfg : REVULTRAConfig, optional
        Configuration parameters for the algorithms
    rng : np.random.Generator, optional
        Random number generator for reproducibility (default: seeded with 42)

    Examples
    --------
    >>> rev = REVULTRAAlgorithms()
    >>> text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    >>> ioc = rev.index_of_coincidence_tensor(text, max_period=5)
    >>> complexity = rev.emergent_complexity_score(text)
    """

    def __init__(
        self, cfg: REVULTRAConfig | None = None, rng: np.random.Generator | None = None
    ) -> None:
        self.cfg = cfg or REVULTRAConfig()
        self.rng = rng or np.random.default_rng(42)

    def _normalize_text(self, text: str) -> str:
        """Normalize text to uppercase letters only.

        Parameters
        ----------
        text : str
            Input text to normalize

        Returns
        -------
        str
            Normalized text containing only A-Z uppercase letters
        """

        return "".join(c.upper() for c in text if c.upper() in string.ascii_uppercase)

    def quantum_information_topology(self, text: str) -> dict[str, complex]:
        """Compute quantum information topology of ciphertext.

        Maps character probabilities to complex quantum amplitudes using
        holographic encoding principles.

        Parameters
        ----------
        text : str
            Ciphertext to analyze

        Returns
        -------
        dict[str, complex]
            Dictionary mapping characters to complex quantum amplitudes

        Examples
        --------
        >>> rev = REVULTRAAlgorithms()
        >>> topology = rev.quantum_information_topology("HELLO")
        >>> abs(topology['L']) > abs(topology['H'])  # L appears more frequently
        True
        """

        normalized = self._normalize_text(text)
        if not normalized:
            return {}

        counts = Counter(normalized)
        total = len(normalized)

        topology = {}
        for char, count in counts.items():
            prob = count / total
            # Compute amplitude with quantum phase
            amplitude = np.sqrt(prob)
            phase = np.exp(1j * self.cfg.qcorr * np.log(prob + EPS))
            topology[char] = amplitude * phase

        return topology

    def holographic_entropy_transform(self, text: str) -> tuple[float, NpArray]:
        """Apply holographic entropy transform using 2D FFT.

        Transforms text into a 2D grid and computes holographic entropy
        via Fourier analysis of the character distribution surface.

        Parameters
        ----------
        text : str
            Ciphertext to analyze

        Returns
        -------
        tuple[float, NDArray[np.float64]]
            Entropy value and 2D surface array representing holographic structure

        Examples
        --------
        >>> rev = REVULTRAAlgorithms()
        >>> entropy, surface = rev.holographic_entropy_transform("ABCD" * 16)
        >>> 0.0 <= entropy <= 10.0
        True
        >>> surface.shape[0] > 0 and surface.shape[1] > 0
        True
        """

        normalized = self._normalize_text(text)
        if len(normalized) < 4:
            # Return minimal surface for short texts
            return 0.0, np.zeros((2, 2), dtype=np.float64)

        # Create 2D grid representation
        side = int(np.ceil(np.sqrt(len(normalized))))
        padded = normalized + "A" * (side * side - len(normalized))

        # Convert to numeric grid
        grid = np.array([ord(c) - ord("A") for c in padded], dtype=np.float64)
        grid = grid.reshape(side, side)

        # Apply 2D FFT
        fft_surface = np.fft.fft2(grid)
        magnitude = np.abs(fft_surface)

        # Compute entropy from FFT magnitudes
        magnitude_flat = magnitude.flatten()
        magnitude_norm = magnitude_flat / (np.sum(magnitude_flat) + EPS)
        entropy = -np.sum(magnitude_norm * np.log(magnitude_norm + EPS))

        return float(entropy), magnitude

    def temporal_embedding_sequence(self, text: str, window_size: int = 5) -> NpArray:
        """Generate temporal embedding sequence with sliding window.

        Creates time-series embeddings by analyzing character transitions
        within a sliding window over the ciphertext.

        Parameters
        ----------
        text : str
            Ciphertext to analyze
        window_size : int, optional
            Size of sliding window (default: 5)

        Returns
        -------
        NDArray[np.float64]
            Array of temporal embeddings, shape (num_windows, window_size)

        Examples
        --------
        >>> rev = REVULTRAAlgorithms()
        >>> embeddings = rev.temporal_embedding_sequence("ABCDEFGH", window_size=3)
        >>> embeddings.shape[1] == 3
        True
        """

        normalized = self._normalize_text(text)
        if len(normalized) < window_size:
            return np.array([[]], dtype=np.float64)

        # Create sliding windows
        embeddings = []
        for i in range(len(normalized) - window_size + 1):
            window = normalized[i : i + window_size]
            # Convert to numeric embedding
            embed = np.array([ord(c) - ord("A") for c in window], dtype=np.float64)
            # Apply temporal quantum weighting
            weights = np.exp(-self.cfg.teq * np.arange(window_size) / window_size)
            embeddings.append(embed * weights)

        return np.array(embeddings, dtype=np.float64)

    def memory_recursive_pattern_mining(
        self, text: str, min_pattern_len: int = 3, max_pattern_len: int = 6
    ) -> dict[str, int]:
        """Mine recurring patterns using recursive memory structures.

        Identifies repeating patterns of various lengths in the ciphertext
        using recursive search with memory optimization.

        Parameters
        ----------
        text : str
            Ciphertext to analyze
        min_pattern_len : int, optional
            Minimum pattern length to search (default: 3)
        max_pattern_len : int, optional
            Maximum pattern length to search (default: 6)

        Returns
        -------
        dict[str, int]
            Dictionary mapping patterns to occurrence counts

        Examples
        --------
        >>> rev = REVULTRAAlgorithms()
        >>> patterns = rev.memory_recursive_pattern_mining("ABCABC")
        >>> patterns.get("ABC", 0) >= 2
        True
        """

        normalized = self._normalize_text(text)
        if len(normalized) < min_pattern_len:
            return {}

        patterns: dict[str, int] = {}

        for length in range(min_pattern_len, min(max_pattern_len + 1, len(normalized))):
            for i in range(len(normalized) - length + 1):
                pattern = normalized[i : i + length]
                patterns[pattern] = patterns.get(pattern, 0) + 1

        # Filter to only recurring patterns (count >= 2)
        return {p: c for p, c in patterns.items() if c >= 2}

    def index_of_coincidence_tensor(self, text: str, max_period: int = 20) -> NpArray:
        """Compute Index of Coincidence (IoC) tensor for period detection.

        Calculates IoC for different period lengths to identify potential
        polyalphabetic cipher periodicities.

        Parameters
        ----------
        text : str
            Ciphertext to analyze
        max_period : int, optional
            Maximum period length to test (default: 20)

        Returns
        -------
        NDArray[np.float64]
            Array of IoC values for periods 1 through max_period

        Examples
        --------
        >>> rev = REVULTRAAlgorithms()
        >>> ioc = rev.index_of_coincidence_tensor("ABCD" * 20, max_period=10)
        >>> np.all((ioc >= 0.0) & (ioc <= 1.0))
        True
        """

        normalized = self._normalize_text(text)
        if len(normalized) < 2:
            return np.zeros(max_period, dtype=np.float64)

        ioc_values = []

        for period in range(1, max_period + 1):
            # Split text into cosets
            cosets = [normalized[i::period] for i in range(period)]

            # Compute IoC for each coset and average
            ioc_sum = 0.0
            for coset in cosets:
                if len(coset) < 2:
                    continue
                counts = Counter(coset)
                n = len(coset)
                ioc = sum(c * (c - 1) for c in counts.values()) / (n * (n - 1) + EPS)
                ioc_sum += ioc

            avg_ioc = ioc_sum / period if period > 0 else 0.0
            ioc_values.append(avg_ioc)

        return np.array(ioc_values, dtype=np.float64)

    def spectral_autocorrelation(self, text: str, max_lag: int = 50) -> NpArray:
        """Compute spectral autocorrelation for periodicity detection.

        Uses FFT-based autocorrelation to identify periodic structures
        in the frequency domain.

        Parameters
        ----------
        text : str
            Ciphertext to analyze
        max_lag : int, optional
            Maximum lag for autocorrelation (default: 50)

        Returns
        -------
        NDArray[np.float64]
            Autocorrelation coefficients for lags 0 through max_lag

        Examples
        --------
        >>> rev = REVULTRAAlgorithms()
        >>> autocorr = rev.spectral_autocorrelation("ABCABC" * 10)
        >>> autocorr[0] == 1.0  # Perfect correlation at lag 0
        True
        """

        normalized = self._normalize_text(text)
        if len(normalized) < 2:
            return np.zeros(max_lag + 1, dtype=np.float64)

        # Convert to numeric sequence
        numeric = np.array([ord(c) - ord("A") for c in normalized], dtype=np.float64)

        # Normalize
        numeric = numeric - np.mean(numeric)

        # Compute autocorrelation via FFT
        n = len(numeric)
        fft = np.fft.fft(numeric, n=2 * n)
        power = fft * np.conj(fft)
        autocorr = np.fft.ifft(power).real[:n]

        # Normalize by lag 0
        autocorr = autocorr / (autocorr[0] + EPS)

        return autocorr[: max_lag + 1]

    def emergent_complexity_score(self, text: str) -> dict[str, float]:
        """Compute emergent complexity metrics for ciphertext.

        Combines multiple complexity measures including entropy, pattern density,
        and topological features to produce an overall complexity score.

        Parameters
        ----------
        text : str
            Ciphertext to analyze

        Returns
        -------
        dict[str, float]
            Dictionary with 'score' and component metrics

        Examples
        --------
        >>> rev = REVULTRAAlgorithms()
        >>> complexity = rev.emergent_complexity_score("ABCD" * 20)
        >>> 'score' in complexity and 'entropy' in complexity
        True
        >>> 0.0 <= complexity['score'] <= 100.0
        True
        """

        normalized = self._normalize_text(text)
        if not normalized:
            return {"score": 0.0, "entropy": 0.0, "pattern_density": 0.0, "ioc_variance": 0.0}

        # Component 1: Character entropy
        counts = Counter(normalized)
        probs = np.array([c / len(normalized) for c in counts.values()], dtype=np.float64)
        char_entropy = -np.sum(probs * np.log(probs + EPS))

        # Component 2: Pattern density
        patterns = self.memory_recursive_pattern_mining(normalized)
        pattern_density = len(patterns) / max(len(normalized), 1)

        # Component 3: IoC variance (periodicity indicator)
        ioc = self.index_of_coincidence_tensor(normalized, max_period=10)
        ioc_variance = float(np.var(ioc))

        # Combine into overall score (0-100 scale)
        score = (char_entropy * 10 + pattern_density * 50 + ioc_variance * 100) / 3
        score = min(100.0, max(0.0, score))

        return {
            "score": float(score),
            "entropy": float(char_entropy),
            "pattern_density": float(pattern_density),
            "ioc_variance": float(ioc_variance),
        }

    def kryptos_k4_analysis(self, ciphertext: str) -> dict[str, Any]:
        """Specialized analysis for Kryptos K4 or similar ciphertexts.

        Combines all REVULTRA methods for comprehensive cryptanalysis.

        Parameters
        ----------
        ciphertext : str
            Ciphertext to analyze (e.g., Kryptos K4)

        Returns
        -------
        dict[str, Any]
            Complete analysis results including all metrics

        Examples
        --------
        >>> rev = REVULTRAAlgorithms()
        >>> k4 = "OBKRUOXOGHULBSOLIFBBWFLRVQQPRNGKSSOTWTQSJQSSEKZZWATJKLUDIAWINFBNYPVTTMZFPKWGDKZXTJCDIGKUHUAUEKCAR"
        >>> results = rev.kryptos_k4_analysis(k4)
        >>> 'complexity' in results and 'ioc_tensor' in results
        True
        """

        normalized = self._normalize_text(ciphertext)

        return {
            "length": len(normalized),
            "topology": self.quantum_information_topology(normalized),
            "complexity": self.emergent_complexity_score(normalized),
            "ioc_tensor": self.index_of_coincidence_tensor(normalized, max_period=20),
            "patterns": self.memory_recursive_pattern_mining(normalized),
            "autocorrelation": self.spectral_autocorrelation(normalized, max_lag=30),
        }

    def frequency_analysis(self, text: str) -> dict[str, float]:
        """Perform basic frequency analysis on ciphertext.

        Parameters
        ----------
        text : str
            Ciphertext to analyze

        Returns
        -------
        dict[str, float]
            Character frequencies as percentages

        Examples
        --------
        >>> rev = REVULTRAAlgorithms()
        >>> freqs = rev.frequency_analysis("AAB")
        >>> freqs['A'] > freqs['B']
        True
        """

        normalized = self._normalize_text(text)
        if not normalized:
            return {}

        counts = Counter(normalized)
        total = len(normalized)
        return {char: (count / total) * 100 for char, count in counts.items()}

    def chi_squared_test(self, text: str, expected_freqs: dict[str, float] | None = None) -> float:
        """Perform chi-squared test against expected frequency distribution.

        Parameters
        ----------
        text : str
            Ciphertext to test
        expected_freqs : dict[str, float], optional
            Expected frequencies (defaults to English letter frequencies)

        Returns
        -------
        float
            Chi-squared statistic

        Examples
        --------
        >>> rev = REVULTRAAlgorithms()
        >>> chi2 = rev.chi_squared_test("AAAAA")
        >>> chi2 > 0.0
        True
        """

        # Default English letter frequencies (percentages)
        if expected_freqs is None:
            expected_freqs = {
                "E": 12.7,
                "T": 9.1,
                "A": 8.2,
                "O": 7.5,
                "I": 7.0,
                "N": 6.7,
                "S": 6.3,
                "H": 6.1,
                "R": 6.0,
                "D": 4.3,
                "L": 4.0,
                "C": 2.8,
                "U": 2.8,
                "M": 2.4,
                "W": 2.4,
                "F": 2.2,
                "G": 2.0,
                "Y": 2.0,
                "P": 1.9,
                "B": 1.5,
                "V": 1.0,
                "K": 0.8,
                "J": 0.15,
                "X": 0.15,
                "Q": 0.10,
                "Z": 0.07,
            }

        normalized = self._normalize_text(text)
        if not normalized:
            return 0.0

        observed = self.frequency_analysis(normalized)

        chi2 = 0.0
        for char in string.ascii_uppercase:
            obs = observed.get(char, 0.0)
            exp = expected_freqs.get(char, 1.0)  # Uniform fallback
            chi2 += ((obs - exp) ** 2) / (exp + EPS)

        return float(chi2)
