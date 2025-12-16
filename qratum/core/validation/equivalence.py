"""
Equivalence Validator for QRATUM

Validates equivalence between quantum and classical backends, and optimization passes.
Certificate: QRATUM-HARDENING-20251215-V5
"""

from typing import Any, Dict, Optional

import numpy as np


class EquivalenceValidator:
    """
    Validates equivalence between different computational paths.

    Used to ensure:
    - Quantum-classical equivalence
    - Optimization preserves results
    - Cross-platform consistency
    """

    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize equivalence validator.

        Args:
            tolerance: Maximum acceptable difference for equivalence
        """
        self.tolerance = tolerance

    def validate_array_equivalence(
        self, array1: np.ndarray, array2: np.ndarray, name1: str = "array1", name2: str = "array2"
    ) -> Dict[str, Any]:
        """
        Validate equivalence between two arrays.

        Args:
            array1: First array
            array2: Second array
            name1: Name of first array
            name2: Name of second array

        Returns:
            Dictionary with validation results
        """
        # Shape check
        if array1.shape != array2.shape:
            return {
                "equivalent": False,
                "reason": f"Shape mismatch: {array1.shape} vs {array2.shape}",
                "max_diff": None,
                "mean_diff": None,
            }

        # Compute differences
        diff = np.abs(array1 - array2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        equivalent = max_diff <= self.tolerance

        return {
            "equivalent": bool(equivalent),
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff),
            "tolerance": self.tolerance,
            "reason": (
                None
                if equivalent
                else f"Max difference {max_diff:.2e} exceeds tolerance {self.tolerance:.2e}"
            ),
        }

    def validate_scalar_equivalence(
        self, value1: float, value2: float, name1: str = "value1", name2: str = "value2"
    ) -> Dict[str, Any]:
        """
        Validate equivalence between two scalar values.

        Args:
            value1: First value
            value2: Second value
            name1: Name of first value
            name2: Name of second value

        Returns:
            Dictionary with validation results
        """
        diff = abs(value1 - value2)
        equivalent = diff <= self.tolerance

        return {
            "equivalent": bool(equivalent),
            "value1": float(value1),
            "value2": float(value2),
            "diff": float(diff),
            "tolerance": self.tolerance,
            "reason": (
                None
                if equivalent
                else f"Difference {diff:.2e} exceeds tolerance {self.tolerance:.2e}"
            ),
        }

    def validate_dict_equivalence(
        self, dict1: Dict[str, Any], dict2: Dict[str, Any], check_keys: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Validate equivalence between dictionaries.

        Args:
            dict1: First dictionary
            dict2: Second dictionary
            check_keys: Optional list of keys to check (checks all if None)

        Returns:
            Dictionary with validation results
        """
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())

        if keys1 != keys2:
            return {
                "equivalent": False,
                "reason": f"Key mismatch: {keys1.symmetric_difference(keys2)}",
                "key_results": {},
            }

        keys_to_check = check_keys if check_keys is not None else list(keys1)
        key_results = {}
        all_equivalent = True

        for key in keys_to_check:
            if key not in dict1 or key not in dict2:
                key_results[key] = {
                    "equivalent": False,
                    "reason": f"Key '{key}' missing in one dictionary",
                }
                all_equivalent = False
                continue

            val1, val2 = dict1[key], dict2[key]

            # Handle different types
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                result = self.validate_array_equivalence(
                    val1, val2, f"{key}[dict1]", f"{key}[dict2]"
                )
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                result = self.validate_scalar_equivalence(
                    val1, val2, f"{key}[dict1]", f"{key}[dict2]"
                )
            else:
                # For other types, use equality
                result = {
                    "equivalent": val1 == val2,
                    "reason": None if val1 == val2 else f"Values differ: {val1} vs {val2}",
                }

            key_results[key] = result
            if not result["equivalent"]:
                all_equivalent = False

        return {
            "equivalent": all_equivalent,
            "key_results": key_results,
            "reason": None if all_equivalent else "One or more keys have non-equivalent values",
        }
