"""

Security Input Validator for QRATUM

Enforces input validation for sequences, matrices, file paths, and bounds.
Certificate: QRATUM-HARDENING-20251215-V5
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class SecurityValidator:
    """

    Validates inputs for security and correctness.

    Enforces:
    - Sequence alphabet validation
    - Matrix NaN/Inf detection
    - File path sanitization
    - Bounds checking
    """

    # Valid biological sequence alphabets
    DNA_ALPHABET = set("ACGT")
    RNA_ALPHABET = set("ACGU")
    PROTEIN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")

    def __init__(self):
        """Initialize security validator."""

        self.validation_errors = []

    def validate_sequence(
        self, sequence: str, alphabet: str = "DNA", allow_ambiguous: bool = False
    ) -> Dict[str, Any]:
        """

        Validate biological sequence.

        Args:
            sequence: Sequence string
            alphabet: One of "DNA", "RNA", "PROTEIN"
            allow_ambiguous: Allow ambiguous codes (e.g., N for DNA)

        Returns:
            Dictionary with validation results
        """

        if not sequence:
            return {"valid": False, "reason": "Empty sequence"}

        # Select alphabet
        if alphabet == "DNA":
            valid_chars = self.DNA_ALPHABET
            if allow_ambiguous:
                valid_chars = valid_chars | set("NRYSWKMBDHV")
        elif alphabet == "RNA":
            valid_chars = self.RNA_ALPHABET
            if allow_ambiguous:
                valid_chars = valid_chars | set("NRYSWKMBDHV")
        elif alphabet == "PROTEIN":
            valid_chars = self.PROTEIN_ALPHABET
            if allow_ambiguous:
                valid_chars = valid_chars | set("XBZJ")
        else:
            return {"valid": False, "reason": f"Unknown alphabet: {alphabet}"}

        # Check sequence
        sequence_upper = sequence.upper()
        invalid_chars = set(sequence_upper) - valid_chars

        if invalid_chars:
            error_msg = f"Invalid characters in {alphabet} sequence: {invalid_chars}"
            self.validation_errors.append(error_msg)
            return {"valid": False, "reason": error_msg, "invalid_chars": list(invalid_chars)}

        return {"valid": True, "length": len(sequence), "alphabet": alphabet}

    def validate_matrix(
        self,
        matrix: np.ndarray,
        name: str = "matrix",
        check_finite: bool = True,
        check_positive: bool = False,
        check_normalized: bool = False,
    ) -> Dict[str, Any]:
        """

        Validate numerical matrix.

        Args:
            matrix: Matrix to validate
            name: Name for error reporting
            check_finite: Check for NaN/Inf
            check_positive: Check all values are positive
            check_normalized: Check rows/columns sum to 1

        Returns:
            Dictionary with validation results
        """

        if not isinstance(matrix, np.ndarray):
            return {"valid": False, "reason": f"{name} is not a numpy array"}

        # Check for NaN/Inf
        if check_finite:
            has_nan = np.any(np.isnan(matrix))
            has_inf = np.any(np.isinf(matrix))

            if has_nan or has_inf:
                error_msg = f"{name} contains "
                if has_nan:
                    error_msg += "NaN"
                if has_inf:
                    error_msg += " Inf" if has_nan else "Inf"
                error_msg += " values"
                self.validation_errors.append(error_msg)
                return {
                    "valid": False,
                    "reason": error_msg,
                    "has_nan": bool(has_nan),
                    "has_inf": bool(has_inf),
                }

        # Check positivity
        if check_positive:
            if np.any(matrix < 0):
                error_msg = f"{name} contains negative values"
                self.validation_errors.append(error_msg)
                return {"valid": False, "reason": error_msg}

        # Check normalization
        if check_normalized:
            if matrix.ndim == 2:
                row_sums = np.sum(matrix, axis=1)
                if not np.allclose(row_sums, 1.0, atol=1e-6):
                    error_msg = f"{name} rows do not sum to 1"
                    self.validation_errors.append(error_msg)
                    return {"valid": False, "reason": error_msg}

        return {"valid": True, "shape": matrix.shape, "dtype": str(matrix.dtype)}

    def validate_file_path(
        self,
        path: str,
        must_exist: bool = False,
        allowed_extensions: Optional[list] = None,
        base_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """

        Validate and sanitize file path.

        Args:
            path: File path to validate
            must_exist: Check if file exists
            allowed_extensions: List of allowed extensions (e.g., ['.fasta', '.txt'])
            base_dir: Base directory to restrict access

        Returns:
            Dictionary with validation results
        """

        try:
            # Resolve path
            path_obj = Path(path).resolve()

            # Check for directory traversal
            if ".." in path:
                error_msg = "Directory traversal attempt detected"
                self.validation_errors.append(error_msg)
                return {"valid": False, "reason": error_msg}

            # Check base directory restriction
            if base_dir is not None:
                base_path = Path(base_dir).resolve()
                try:
                    path_obj.relative_to(base_path)
                except ValueError:
                    error_msg = f"Path {path} is outside allowed base directory {base_dir}"
                    self.validation_errors.append(error_msg)
                    return {"valid": False, "reason": error_msg}

            # Check existence
            if must_exist and not path_obj.exists():
                error_msg = f"Path does not exist: {path}"
                self.validation_errors.append(error_msg)
                return {"valid": False, "reason": error_msg}

            # Check extension
            if allowed_extensions is not None:
                if path_obj.suffix not in allowed_extensions:
                    error_msg = f"File extension {path_obj.suffix} not in allowed list: {allowed_extensions}"
                    self.validation_errors.append(error_msg)
                    return {"valid": False, "reason": error_msg}

            return {"valid": True, "resolved_path": str(path_obj), "exists": path_obj.exists()}

        except Exception as e:
            error_msg = f"Path validation failed: {e}"
            self.validation_errors.append(error_msg)
            return {"valid": False, "reason": error_msg}

    def validate_bounds(
        self,
        value: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        name: str = "value",
    ) -> Dict[str, Any]:
        """

        Validate value is within bounds.

        Args:
            value: Value to check
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            name: Name for error reporting

        Returns:
            Dictionary with validation results
        """

        if min_value is not None and value < min_value:
            error_msg = f"{name} ({value}) is below minimum ({min_value})"
            self.validation_errors.append(error_msg)
            return {"valid": False, "reason": error_msg}

        if max_value is not None and value > max_value:
            error_msg = f"{name} ({value}) is above maximum ({max_value})"
            self.validation_errors.append(error_msg)
            return {"valid": False, "reason": error_msg}

        return {"valid": True, "value": value}

    def get_errors(self) -> list:
        """Get all validation errors."""

        return self.validation_errors.copy()

    def reset_errors(self) -> None:
        """Clear all validation errors."""

        self.validation_errors.clear()
