"""Tests for REVULTRA algorithms."""

import numpy as np
import pytest

from quasim.revultra.algorithms import REVULTRAAlgorithms, REVULTRAConfig


@pytest.fixture
def revultra():
    """Create REVULTRAAlgorithms instance with deterministic RNG."""

    return REVULTRAAlgorithms(rng=np.random.default_rng(42))


class TestQuantumInformationTopology:
    """Tests for quantum_information_topology method."""

    def test_topology_basic(self, revultra):
        """Test basic topology computation."""

        text = "HELLO"
        topology = revultra.quantum_information_topology(text)

        assert isinstance(topology, dict)
        assert len(topology) > 0
        assert all(isinstance(v, complex) for v in topology.values())

    def test_topology_frequency(self, revultra):
        """Test that more frequent characters have higher amplitudes."""

        text = "AAABBC"
        topology = revultra.quantum_information_topology(text)

        # A appears 3 times, B appears 2 times, C appears 1 time
        assert abs(topology["A"]) > abs(topology["B"])
        assert abs(topology["B"]) > abs(topology["C"])

    def test_topology_empty(self, revultra):
        """Test with empty text."""

        topology = revultra.quantum_information_topology("")
        assert len(topology) == 0


class TestHolographicEntropyTransform:
    """Tests for holographic_entropy_transform method."""

    def test_entropy_bounds(self, revultra):
        """Test that entropy is non-negative."""

        text = "ABCDEFGH" * 10
        entropy, surface = revultra.holographic_entropy_transform(text)

        assert entropy >= 0.0
        assert isinstance(surface, np.ndarray)
        assert surface.shape[0] > 0
        assert surface.shape[1] > 0

    def test_entropy_short_text(self, revultra):
        """Test with very short text."""

        text = "AB"
        entropy, surface = revultra.holographic_entropy_transform(text)

        assert entropy >= 0.0
        assert surface.shape == (2, 2)


class TestTemporalEmbeddingSequence:
    """Tests for temporal_embedding_sequence method."""

    def test_embedding_shape(self, revultra):
        """Test embedding shape."""

        text = "ABCDEFGH"
        window_size = 3
        embeddings = revultra.temporal_embedding_sequence(text, window_size=window_size)

        assert embeddings.shape[1] == window_size
        assert embeddings.shape[0] == len(text) - window_size + 1

    def test_embedding_short_text(self, revultra):
        """Test with text shorter than window."""

        text = "AB"
        embeddings = revultra.temporal_embedding_sequence(text, window_size=5)

        # Should return empty 2D array when text is too short
        assert embeddings.shape == (1, 0) or embeddings.shape == (0, 0)


class TestMemoryRecursivePatternMining:
    """Tests for memory_recursive_pattern_mining method."""

    def test_pattern_detection(self, revultra):
        """Test basic pattern detection."""

        text = "ABCABC"
        patterns = revultra.memory_recursive_pattern_mining(text, min_pattern_len=3)

        assert "ABC" in patterns
        assert patterns["ABC"] >= 2

    def test_no_patterns(self, revultra):
        """Test with no repeating patterns."""

        text = "ABCDEFGH"
        patterns = revultra.memory_recursive_pattern_mining(text)

        # Should be empty or only have patterns that appear < 2 times
        assert all(count >= 2 for count in patterns.values())


class TestIndexOfCoincidenceTensor:
    """Tests for index_of_coincidence_tensor method."""

    def test_ioc_bounds(self, revultra):
        """Test that IoC values are in [0, 1]."""

        text = "ABCD" * 50
        ioc = revultra.index_of_coincidence_tensor(text, max_period=8)

        assert np.all(ioc >= 0.0)
        assert np.all(ioc <= 1.0)
        assert len(ioc) == 8

    def test_ioc_periodic_text(self, revultra):
        """Test IoC on periodic text."""

        # Text with period 3
        text = "ABC" * 30
        ioc = revultra.index_of_coincidence_tensor(text, max_period=10)

        # IoC at period 3 should be relatively high
        assert ioc[2] > 0.5  # 0-indexed, so period 3 is index 2


class TestSpectralAutocorrelation:
    """Tests for spectral_autocorrelation method."""

    def test_autocorr_lag_zero(self, revultra):
        """Test that autocorrelation at lag 0 is 1.0."""

        text = "ABCABC" * 10
        autocorr = revultra.spectral_autocorrelation(text)

        assert abs(autocorr[0] - 1.0) < 1e-6

    def test_autocorr_shape(self, revultra):
        """Test autocorrelation output shape."""

        text = "ABCDEF" * 10
        max_lag = 20
        autocorr = revultra.spectral_autocorrelation(text, max_lag=max_lag)

        assert len(autocorr) == max_lag + 1


class TestEmergentComplexityScore:
    """Tests for emergent_complexity_score method."""

    def test_complexity_bounds(self, revultra):
        """Test that complexity score is in valid range."""

        text = "ABCDEFGH" * 20
        complexity = revultra.emergent_complexity_score(text)

        assert 0.0 <= complexity["score"] <= 100.0
        assert "entropy" in complexity
        assert "pattern_density" in complexity
        assert "ioc_variance" in complexity

    def test_complexity_components(self, revultra):
        """Test that all complexity components are present."""

        text = "ATTACKATDAWN" * 5
        complexity = revultra.emergent_complexity_score(text)

        assert complexity["entropy"] >= 0.0
        assert complexity["pattern_density"] >= 0.0
        assert complexity["ioc_variance"] >= 0.0


class TestFrequencyAnalysis:
    """Tests for frequency_analysis method."""

    def test_frequency_sum(self, revultra):
        """Test that frequencies sum to 100%."""

        text = "ABCABC"
        freqs = revultra.frequency_analysis(text)

        total = sum(freqs.values())
        assert abs(total - 100.0) < 1e-6

    def test_frequency_values(self, revultra):
        """Test frequency calculation."""

        text = "AAB"
        freqs = revultra.frequency_analysis(text)

        assert abs(freqs["A"] - 66.67) < 0.1
        assert abs(freqs["B"] - 33.33) < 0.1


class TestChiSquaredTest:
    """Tests for chi_squared_test method."""

    def test_chi_squared_non_negative(self, revultra):
        """Test that chi-squared is non-negative."""

        text = "AAAAA"
        chi2 = revultra.chi_squared_test(text)

        assert chi2 >= 0.0

    def test_chi_squared_with_custom_freqs(self, revultra):
        """Test with custom expected frequencies."""

        text = "ABCD"
        expected = {"A": 25.0, "B": 25.0, "C": 25.0, "D": 25.0}
        chi2 = revultra.chi_squared_test(text, expected_freqs=expected)

        assert chi2 >= 0.0


class TestKryptosK4Analysis:
    """Tests for kryptos_k4_analysis method."""

    def test_k4_analysis_complete(self, revultra):
        """Test complete K4 analysis."""

        k4 = "OBKRUOXOGHULBSOLIFBBWFLRVQQPRNGKSSOTWTQSJQSSEKZZWATJKLUDIAWINFBNYPVTTMZFPK"
        results = revultra.kryptos_k4_analysis(k4)

        assert "length" in results
        assert "topology" in results
        assert "complexity" in results
        assert "ioc_tensor" in results
        assert "patterns" in results
        assert "autocorrelation" in results

    def test_k4_length(self, revultra):
        """Test K4 analysis reports correct length."""

        text = "ABCDEFGH" * 10
        results = revultra.kryptos_k4_analysis(text)

        # After normalization (removing non-letters)
        assert results["length"] == 80


class TestREVULTRAConfig:
    """Tests for REVULTRAConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""

        config = REVULTRAConfig()

        assert config.hsrq == 0.15
        assert config.teq == 1.0
        assert config.etq == 1.3
        assert config.itq == 0.9
        assert config.qcorr == 0.73
        assert config.memq == 0.88

    def test_config_custom(self):
        """Test custom configuration values."""

        config = REVULTRAConfig(hsrq=0.2, teq=1.5)

        assert config.hsrq == 0.2
        assert config.teq == 1.5


class TestDeterministicBehavior:
    """Tests for deterministic behavior with seeded RNG."""

    def test_deterministic_topology(self):
        """Test that topology is deterministic."""

        rev1 = REVULTRAAlgorithms(rng=np.random.default_rng(123))
        rev2 = REVULTRAAlgorithms(rng=np.random.default_rng(123))

        text = "HELLOWORLD"
        topo1 = rev1.quantum_information_topology(text)
        topo2 = rev2.quantum_information_topology(text)

        assert topo1 == topo2

    def test_deterministic_entropy(self):
        """Test that entropy transform is deterministic."""

        rev1 = REVULTRAAlgorithms(rng=np.random.default_rng(123))
        rev2 = REVULTRAAlgorithms(rng=np.random.default_rng(123))

        text = "TESTCIPHERTEXT" * 5
        ent1, surf1 = rev1.holographic_entropy_transform(text)
        ent2, surf2 = rev2.holographic_entropy_transform(text)

        assert abs(ent1 - ent2) < 1e-10
        assert np.allclose(surf1, surf2)
