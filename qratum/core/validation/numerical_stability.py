"""
Numerical Stability Analyzer for QRATUM

Monitors condition numbers, entropy stability, gradient flow, and overflow detection.
Certificate: QRATUM-HARDENING-20251215-V5
"""

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np


class NumericalStabilityAnalyzer:
    """
    Analyzes numerical stability of computations.
    
    Monitors:
    - Matrix condition numbers
    - Entropy computation stability
    - Gradient flow analysis
    - Overflow/underflow detection
    """
    
    def __init__(self, condition_threshold: float = 1e10):
        """
        Initialize stability analyzer.
        
        Args:
            condition_threshold: Maximum acceptable condition number
        """
        self.condition_threshold = condition_threshold
        self.warnings_issued = []
    
    def check_matrix_condition(self, matrix: np.ndarray, name: str = "matrix") -> Tuple[bool, float]:
        """
        Check matrix condition number.
        
        Args:
            matrix: Matrix to analyze
            name: Name for reporting
            
        Returns:
            Tuple of (is_stable, condition_number)
        """
        try:
            cond = np.linalg.cond(matrix)
            is_stable = cond < self.condition_threshold
            
            if not is_stable:
                warning_msg = f"{name} has high condition number: {cond:.2e} (threshold: {self.condition_threshold:.2e})"
                warnings.warn(warning_msg)
                self.warnings_issued.append(warning_msg)
            
            return is_stable, float(cond)
        except np.linalg.LinAlgError as e:
            warning_msg = f"{name} condition number computation failed: {e}"
            warnings.warn(warning_msg)
            self.warnings_issued.append(warning_msg)
            return False, float('inf')
    
    def check_entropy_stability(self, probabilities: np.ndarray, epsilon: float = 1e-10) -> Tuple[bool, Optional[float]]:
        """
        Check entropy computation stability.
        
        Args:
            probabilities: Probability distribution
            epsilon: Small value for numerical stability
            
        Returns:
            Tuple of (is_valid, entropy_value)
        """
        # Check if probabilities sum to approximately 1
        prob_sum = np.sum(probabilities)
        if not np.isclose(prob_sum, 1.0, atol=1e-6):
            warning_msg = f"Probabilities sum to {prob_sum:.6f}, not 1.0"
            warnings.warn(warning_msg)
            self.warnings_issued.append(warning_msg)
            return False, None
        
        # Check for negative probabilities
        if np.any(probabilities < 0):
            warning_msg = "Negative probabilities detected"
            warnings.warn(warning_msg)
            self.warnings_issued.append(warning_msg)
            return False, None
        
        # Compute entropy with stability
        prob_safe = np.maximum(probabilities, epsilon)
        entropy = -np.sum(probabilities * np.log2(prob_safe))
        
        return True, float(entropy)
    
    def check_gradient_flow(self, gradients: np.ndarray, threshold: float = 1e-7) -> Dict[str, Any]:
        """
        Analyze gradient flow in neural networks.
        
        Args:
            gradients: Gradient tensor
            threshold: Threshold for vanishing gradient detection
            
        Returns:
            Dictionary with gradient statistics
        """
        grad_norm = np.linalg.norm(gradients)
        grad_max = np.max(np.abs(gradients))
        grad_min = np.min(np.abs(gradients[gradients != 0])) if np.any(gradients != 0) else 0.0
        
        vanishing = grad_max < threshold
        exploding = grad_max > 1e3
        
        if vanishing:
            warning_msg = f"Vanishing gradients detected: max={grad_max:.2e}"
            warnings.warn(warning_msg)
            self.warnings_issued.append(warning_msg)
        
        if exploding:
            warning_msg = f"Exploding gradients detected: max={grad_max:.2e}"
            warnings.warn(warning_msg)
            self.warnings_issued.append(warning_msg)
        
        return {
            "norm": float(grad_norm),
            "max": float(grad_max),
            "min": float(grad_min),
            "vanishing": vanishing,
            "exploding": exploding
        }
    
    def detect_overflow_underflow(self, array: np.ndarray, name: str = "array") -> Dict[str, Any]:
        """
        Detect overflow and underflow in numerical arrays.
        
        Args:
            array: Array to check
            name: Name for reporting
            
        Returns:
            Dictionary with detection results
        """
        has_nan = np.any(np.isnan(array))
        has_inf = np.any(np.isinf(array))
        
        # Check for values close to machine epsilon (underflow)
        if array.dtype in [np.float32, np.float64]:
            eps = np.finfo(array.dtype).eps
            has_underflow = np.any(np.abs(array[array != 0]) < eps * 10)
        else:
            has_underflow = False
        
        if has_nan:
            warning_msg = f"{name} contains NaN values"
            warnings.warn(warning_msg)
            self.warnings_issued.append(warning_msg)
        
        if has_inf:
            warning_msg = f"{name} contains Inf values"
            warnings.warn(warning_msg)
            self.warnings_issued.append(warning_msg)
        
        if has_underflow:
            warning_msg = f"{name} contains values close to underflow"
            warnings.warn(warning_msg)
            self.warnings_issued.append(warning_msg)
        
        return {
            "has_nan": bool(has_nan),
            "has_inf": bool(has_inf),
            "has_underflow": bool(has_underflow),
            "is_stable": not (has_nan or has_inf)
        }
    
    def get_warnings(self) -> list:
        """Get all warnings issued during analysis."""
        return self.warnings_issued.copy()
    
    def reset_warnings(self) -> None:
        """Clear all warnings."""
        self.warnings_issued.clear()
