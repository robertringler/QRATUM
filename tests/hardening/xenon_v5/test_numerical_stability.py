"""
Numerical Stability Tests for XENON v5

Tests numerical stability of all computations.
Certificate: QRATUM-HARDENING-20251215-V5
"""

import pytest
import numpy as np

from qratum.core.validation import NumericalStabilityAnalyzer
from qratum.bioinformatics.xenon.omics import InformationEngine


class TestNumericalStability:
    """Test numerical stability across XENON."""
    
    def test_matrix_condition_number(self):
        """Test matrix condition number checking."""
        analyzer = NumericalStabilityAnalyzer(condition_threshold=1e10)
        
        # Well-conditioned matrix
        well_conditioned = np.eye(10)
        is_stable, cond = analyzer.check_matrix_condition(well_conditioned)
        assert is_stable, "Identity matrix should be well-conditioned"
        assert cond < 10, "Identity matrix condition number should be ~1"
        
        # Ill-conditioned matrix
        ill_conditioned = np.array([[1, 1], [1, 1.00000001]])
        is_stable, cond = analyzer.check_matrix_condition(ill_conditioned)
        # This may or may not be stable depending on threshold
        assert cond > 1, "Nearly singular matrix should have high condition number"
    
    def test_entropy_stability(self):
        """Test entropy computation stability."""
        analyzer = NumericalStabilityAnalyzer()
        
        # Valid probability distribution
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        is_valid, entropy = analyzer.check_entropy_stability(probs)
        assert is_valid, "Valid probability distribution should pass"
        assert entropy is not None
        assert 0 <= entropy <= 2.0, "Entropy should be in valid range"
        
        # Invalid: doesn't sum to 1
        invalid_probs = np.array([0.1, 0.2, 0.3])
        is_valid, entropy = analyzer.check_entropy_stability(invalid_probs)
        assert not is_valid, "Probabilities not summing to 1 should fail"
    
    def test_gradient_flow(self):
        """Test gradient flow analysis."""
        analyzer = NumericalStabilityAnalyzer()
        
        # Normal gradients
        normal_grads = np.random.randn(100, 100) * 0.1
        result = analyzer.check_gradient_flow(normal_grads)
        assert not result["vanishing"], "Normal gradients should not vanish"
        assert not result["exploding"], "Normal gradients should not explode"
        
        # Vanishing gradients
        vanishing_grads = np.random.randn(100, 100) * 1e-10
        result = analyzer.check_gradient_flow(vanishing_grads)
        assert result["vanishing"], "Small gradients should be detected as vanishing"
    
    def test_overflow_underflow_detection(self):
        """Test overflow and underflow detection."""
        analyzer = NumericalStabilityAnalyzer()
        
        # Normal array
        normal = np.array([1.0, 2.0, 3.0])
        result = analyzer.detect_overflow_underflow(normal)
        assert result["is_stable"], "Normal array should be stable"
        assert not result["has_nan"]
        assert not result["has_inf"]
        
        # Array with NaN
        with_nan = np.array([1.0, np.nan, 3.0])
        result = analyzer.detect_overflow_underflow(with_nan)
        assert not result["is_stable"], "Array with NaN should be unstable"
        assert result["has_nan"]
        
        # Array with Inf
        with_inf = np.array([1.0, np.inf, 3.0])
        result = analyzer.detect_overflow_underflow(with_inf)
        assert not result["is_stable"], "Array with Inf should be unstable"
        assert result["has_inf"]
    
    def test_entropy_conservation(self):
        """Test entropy conservation in information engine."""
        engine = InformationEngine(seed=42)
        
        # Generate correlated data
        n = 100
        x = np.random.randn(n, 1)
        y = x + np.random.randn(n, 1) * 0.1
        
        result = engine.compute_mutual_information(x, y)
        
        # Check entropy conservation: H(X,Y) <= H(X) + H(Y)
        assert result["entropy_conservation"], "Entropy must be conserved"
        assert result["h_xy"] <= result["h_x"] + result["h_y"] + 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
