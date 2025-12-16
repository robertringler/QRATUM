"""
Security Tests for XENON v5

Tests input validation and security hardening.
Certificate: QRATUM-HARDENING-20251215-V5
"""

import pytest
import numpy as np

from qratum.core.security import SecurityValidator
from qratum.bioinformatics.xenon.alignment import QuantumAlignmentEngine


class TestSecurity:
    """Test security validation across XENON."""
    
    def test_sequence_validation_dna(self):
        """Test DNA sequence validation."""
        validator = SecurityValidator()
        
        # Valid DNA
        result = validator.validate_sequence("ACGTACGT", alphabet="DNA")
        assert result["valid"], "Valid DNA should pass"
        
        # Invalid characters
        result = validator.validate_sequence("ACGTXYZ", alphabet="DNA")
        assert not result["valid"], "Invalid DNA characters should fail"
        assert "invalid_chars" in result
    
    def test_sequence_validation_protein(self):
        """Test protein sequence validation."""
        validator = SecurityValidator()
        
        # Valid protein
        result = validator.validate_sequence("ACDEFGHIKLMNPQRSTVWY", alphabet="PROTEIN")
        assert result["valid"], "Valid protein should pass"
        
        # Invalid characters
        result = validator.validate_sequence("ACDE123", alphabet="PROTEIN")
        assert not result["valid"], "Invalid protein characters should fail"
    
    def test_matrix_validation(self):
        """Test matrix validation."""
        validator = SecurityValidator()
        
        # Valid matrix
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = validator.validate_matrix(matrix, check_finite=True)
        assert result["valid"], "Valid matrix should pass"
        
        # Matrix with NaN
        nan_matrix = np.array([[1.0, np.nan], [3.0, 4.0]])
        result = validator.validate_matrix(nan_matrix, check_finite=True)
        assert not result["valid"], "Matrix with NaN should fail"
        assert result["has_nan"]
        
        # Matrix with Inf
        inf_matrix = np.array([[1.0, 2.0], [np.inf, 4.0]])
        result = validator.validate_matrix(inf_matrix, check_finite=True)
        assert not result["valid"], "Matrix with Inf should fail"
        assert result["has_inf"]
    
    def test_file_path_validation(self):
        """Test file path sanitization."""
        validator = SecurityValidator()
        
        # Normal path
        result = validator.validate_file_path("/tmp/test.txt")
        assert result["valid"], "Normal path should pass"
        
        # Directory traversal attempt
        result = validator.validate_file_path("../../../etc/passwd")
        assert not result["valid"], "Directory traversal should be blocked"
    
    def test_bounds_validation(self):
        """Test bounds checking."""
        validator = SecurityValidator()
        
        # Within bounds
        result = validator.validate_bounds(5.0, min_value=0.0, max_value=10.0)
        assert result["valid"], "Value within bounds should pass"
        
        # Below minimum
        result = validator.validate_bounds(-5.0, min_value=0.0, max_value=10.0)
        assert not result["valid"], "Value below minimum should fail"
        
        # Above maximum
        result = validator.validate_bounds(15.0, min_value=0.0, max_value=10.0)
        assert not result["valid"], "Value above maximum should fail"
    
    def test_alignment_input_validation(self):
        """Test alignment engine validates inputs."""
        engine = QuantumAlignmentEngine(seed=42)
        
        # Valid sequences
        result = engine.align("ACGT", "ACGT")
        assert "score" in result
        
        # Invalid sequence should raise error
        with pytest.raises(ValueError):
            engine.align("ACGT123", "ACGT")
    
    def test_information_engine_input_validation(self):
        """Test information engine validates inputs."""
        from qratum.bioinformatics.xenon.omics import InformationEngine
        
        engine = InformationEngine(seed=42)
        
        # Valid data
        data_x = np.random.randn(100, 1)
        data_y = np.random.randn(100, 1)
        result = engine.compute_mutual_information(data_x, data_y)
        assert "mutual_information" in result
        
        # Data with NaN should raise error
        nan_data = np.array([[1.0], [np.nan], [3.0]])
        with pytest.raises(ValueError):
            engine.compute_mutual_information(nan_data, data_y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
