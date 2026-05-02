"""QCORE - Canonical semantic problem definitions and Hamiltonian abstractions.

This module provides the foundational abstractions for quantum computing problems,
including semantic state representation and Hamiltonian encoding.

QRATUM v1.0 Extension: Sovereign Control Plane (Authorization, Policy, Resolution, Contracts)

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from qcore.authority import (AuthorizationEngine, AuthorizationError,
                             AuthorizationResult, check_authority_constraints)
from qcore.hamiltonian import Hamiltonian, PauliTerm
from qcore.issuer import ContractBundle, ContractIssuer
from qcore.policy import (PolicyEngine, PolicyEvaluationResult, PolicyRule,
                          create_custom_policy_rule, evaluate_hardware_policy)
from qcore.resolver import (CapabilityResolutionError, CapabilityResolver,
                            ResolvedCapability)
from qcore.semantic_state import (ChemistryValidator, DomainValidator,
                                  FinanceValidator, OptimizationValidator,
                                  SemanticState)

__all__ = [
    # Semantic State & Hamiltonian
    "SemanticState",
    "DomainValidator",
    "ChemistryValidator",
    "OptimizationValidator",
    "FinanceValidator",
    "Hamiltonian",
    "PauliTerm",
    # Authority & Authorization
    "AuthorizationEngine",
    "AuthorizationError",
    "AuthorizationResult",
    "check_authority_constraints",
    # Policy Evaluation
    "PolicyEngine",
    "PolicyEvaluationResult",
    "PolicyRule",
    "create_custom_policy_rule",
    "evaluate_hardware_policy",
    # Capability Resolution
    "CapabilityResolver",
    "CapabilityResolutionError",
    "ResolvedCapability",
    # Contract Issuing
    "ContractIssuer",
    "ContractBundle",
]

__version__ = "1.0.0"
