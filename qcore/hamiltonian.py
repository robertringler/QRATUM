"""Hamiltonian abstraction for quantum problem encoding.

This module provides structural contracts for Hamiltonian representation.
Full encoding implementation will be completed in PR-003 and PR-004.

Version: 1.0.0
Status: Production (Stub)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class PauliTerm:
    """Represents a single term in a Pauli decomposition.

    Attributes:
        coefficient: Complex coefficient for this term
        operators: List of (qubit_index, pauli_operator) tuples
                   where pauli_operator is one of 'I', 'X', 'Y', 'Z'
    """

    coefficient: complex
    operators: list[tuple[int, str]]

    def __post_init__(self) -> None:
        """Validate Pauli operators after initialization."""
        valid_operators = {"I", "X", "Y", "Z"}
        for _qubit_idx, operator in self.operators:
            if operator not in valid_operators:
                raise ValueError(
                    f"Invalid Pauli operator '{operator}'. " f"Must be one of {valid_operators}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with coefficient (split into real/imag) and operators
        """
        return {
            "coefficient": {
                "real": self.coefficient.real,
                "imag": self.coefficient.imag,
            },
            "operators": self.operators,
        }

    def __str__(self) -> str:
        """Human-readable representation.

        Returns:
            String representation like '(2.5+1.5j) * X0 Y1 Z2'
        """
        coeff_str = f"{self.coefficient}"
        if not self.operators:
            return coeff_str
        ops_str = " ".join(f"{op}{idx}" for idx, op in self.operators)
        return f"{coeff_str} * {ops_str}"


class Hamiltonian:
    """Represents a quantum Hamiltonian as a sum of Pauli terms.

    This is a structural contract establishing the interface.
    Full implementation of encoding methods will be completed in:
    - PR-003: energy() method for computing expectation values
    - PR-004: from_semantic_state() method for problem encoding
    """

    def __init__(self, terms: list[PauliTerm]) -> None:
        """Initialize Hamiltonian with a list of Pauli terms.

        Args:
            terms: List of PauliTerm objects
        """
        self.terms = terms

    def encode(self) -> list[dict[str, Any]]:
        """Return list of serialized terms.

        Returns:
            List of dictionaries representing each term
        """
        return [term.to_dict() for term in self.terms]

    def energy(self, state: Any) -> float:
        """Compute energy expectation value for a given state.

        This method will be implemented in PR-003.

        Args:
            state: Quantum state (format TBD)

        Returns:
            Energy expectation value

        Raises:
            NotImplementedError: Placeholder for PR-003
        """
        raise NotImplementedError(
            "energy() computation will be implemented in PR-003. "
            "This is a structural contract establishing the interface."
        )

    @classmethod
    def from_semantic_state(cls, semantic_state: Any) -> Hamiltonian:
        """Construct Hamiltonian from semantic problem definition.

        This method will be implemented in PR-004.

        Args:
            semantic_state: SemanticState instance

        Returns:
            Hamiltonian instance

        Raises:
            NotImplementedError: Placeholder for PR-004
        """
        raise NotImplementedError(
            "from_semantic_state() encoding will be implemented in PR-004. "
            "This is a structural contract establishing the interface."
        )

    def num_qubits(self) -> int:
        """Return number of qubits required for this Hamiltonian.

        Returns:
            Maximum qubit index + 1 across all terms
        """
        if not self.terms:
            return 0
        max_qubit = -1
        for term in self.terms:
            for qubit_idx, _ in term.operators:
                max_qubit = max(max_qubit, qubit_idx)
        return max_qubit + 1

    def __str__(self) -> str:
        """Human-readable representation.

        Returns:
            String like 'H = term1 + term2 + ...'
        """
        if not self.terms:
            return "H = 0"
        terms_str = " + ".join(str(term) for term in self.terms)
        return f"H = {terms_str}"

    def to_json(self) -> str:
        """JSON serialization of Hamiltonian.

        Returns:
            JSON string with all terms
        """
        return json.dumps(
            {
                "num_qubits": self.num_qubits(),
                "terms": self.encode(),
            },
            sort_keys=True,
        )
