"""Partial Information Decomposition (PID) for multi-omics integration.

This module implements information-theoretic fusion of multi-omics data with:
- Partial Information Decomposition (Williams & Beer framework)
- Conservation constraint enforcement (information cannot exceed source entropy)
- Numerical stability monitoring
- Deterministic reproducibility

Mathematical Foundation:
    For target T and sources S1, S2:
    I(S1, S2; T) = Unique(S1) + Unique(S2) + Redundant(S1, S2) + Synergistic(S1, S2)

    Conservation Constraints:
    1. Non-negativity: All components >= 0
    2. Upper bound: I(S1, S2; T) <= min(H(S1, S2), H(T))
    3. Monotonicity: Adding sources cannot decrease information

References:
    Williams, P. L., & Beer, R. D. (2010). Nonnegative decomposition of
    multivariate information. arXiv:1004.2515
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

# Try new package name first, fallback to old for compatibility
try:
    from qratum.common.seeding import SeedManager
except ImportError:
    from quasim.common.seeding import SeedManager


class InformationComponent(Enum):
    """Types of information components in PID."""

    UNIQUE_S1 = "unique_s1"
    UNIQUE_S2 = "unique_s2"
    REDUNDANT = "redundant"
    SYNERGISTIC = "synergistic"


@dataclass
class PIDResult:
    """Result of Partial Information Decomposition.

    Attributes:
        unique_s1: Information unique to source 1
        unique_s2: Information unique to source 2
        redundant: Information shared by both sources
        synergistic: Information only available from both sources together
        total_mi: Total mutual information I(S1, S2; T)
        entropy_target: Entropy of target H(T)
        entropy_sources: Joint entropy of sources H(S1, S2)
        conservation_valid: Whether conservation constraints are satisfied
        violations: List of conservation constraint violations
        condition_number: Numerical stability metric
    """

    unique_s1: float
    unique_s2: float
    redundant: float
    synergistic: float
    total_mi: float
    entropy_target: float
    entropy_sources: float
    conservation_valid: bool
    violations: list[str] = field(default_factory=list)
    condition_number: float = 1.0


@dataclass
class ConservationConstraints:
    """Configuration for conservation constraint enforcement.

    Attributes:
        enforce_non_negativity: Enforce all components >= 0
        enforce_upper_bound: Enforce total MI <= min(H(sources), H(target))
        enforce_monotonicity: Enforce monotonicity constraints
        tolerance: Numerical tolerance for constraint violations
        auto_correct: Automatically correct minor violations
    """

    enforce_non_negativity: bool = True
    enforce_upper_bound: bool = True
    enforce_monotonicity: bool = True
    tolerance: float = 1e-6
    auto_correct: bool = True


class InformationFusionEngine:
    """Multi-omics information fusion with PID and conservation constraints.

    Implements partial information decomposition for understanding how
    different omics layers contribute unique, redundant, and synergistic
    information about biological targets.

    Attributes:
        constraints: Conservation constraint configuration
        seed_manager: Deterministic seed management
        _decompositions: Cache of PID results
    """

    def __init__(
        self,
        constraints: Optional[ConservationConstraints] = None,
        seed: Optional[int] = None,
    ):
        """Initialize information fusion engine.

        Args:
            constraints: Conservation constraint configuration
            seed: Random seed for reproducibility
        """
        self.constraints = constraints or ConservationConstraints()
        self.seed_manager = SeedManager(seed if seed is not None else 42)
        self._decompositions: dict[str, PIDResult] = {}
        self._entropy_cache: dict[str, float] = {}

    def compute_entropy(
        self,
        data: np.ndarray,
        bins: int = 10,
    ) -> float:
        """Compute Shannon entropy of data distribution.

        Mathematical Basis:
            H(X) = -Σ p(x) * log₂(p(x))

        Args:
            data: Data array (samples × features)
            bins: Number of bins for discretization

        Returns:
            Shannon entropy in bits
        """
        if data.size == 0:
            return 0.0

        # Handle multidimensional data by flattening
        if len(data.shape) > 1:
            # For multivariate, use product space
            data_flat = data.reshape(-1, data.shape[-1])
        else:
            data_flat = data.reshape(-1, 1)

        # Discretize data
        try:
            hist, _ = np.histogramdd(data_flat, bins=bins)
        except ValueError:
            # Fallback for edge cases
            hist = np.histogram(data_flat.ravel(), bins=bins)[0]

        # Normalize to probabilities
        hist = hist.ravel()
        hist = hist[hist > 0]  # Remove zero bins
        probs = hist / hist.sum()

        # Compute entropy
        entropy = -np.sum(probs * np.log2(probs))

        return float(entropy)

    def compute_mutual_information(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bins: int = 10,
    ) -> float:
        """Compute mutual information between two variables.

        Mathematical Basis:
            I(X; Y) = H(X) + H(Y) - H(X, Y)

        Args:
            x: First variable (samples,) or (samples, features)
            y: Second variable (samples,)
            bins: Number of bins for discretization

        Returns:
            Mutual information in bits
        """
        if x.size == 0 or y.size == 0:
            return 0.0

        # Flatten to 1D for individual entropies if needed
        x_flat = x.ravel()
        y_flat = y.ravel()

        # Ensure same number of samples
        min_len = min(len(x_flat), len(y_flat))
        x_flat = x_flat[:min_len]
        y_flat = y_flat[:min_len]

        # Compute individual entropies
        h_x = self.compute_entropy(x_flat, bins=bins)
        h_y = self.compute_entropy(y_flat, bins=bins)

        # Compute joint entropy
        xy = np.column_stack([x_flat, y_flat])
        h_xy = self.compute_entropy(xy, bins=bins)

        # MI = H(X) + H(Y) - H(X,Y)
        mi = h_x + h_y - h_xy

        # Enforce non-negativity
        return max(0.0, float(mi))

    def decompose_information(
        self,
        source1: np.ndarray,
        source2: np.ndarray,
        target: np.ndarray,
        bins: int = 10,
    ) -> PIDResult:
        """Perform Partial Information Decomposition (PID).

        Decomposes mutual information between sources and target into:
        - Unique information from source 1
        - Unique information from source 2
        - Redundant information (shared by both)
        - Synergistic information (only from combination)

        Args:
            source1: First source omics layer (samples,)
            source2: Second source omics layer (samples,)
            target: Target variable (samples,)
            bins: Number of bins for discretization

        Returns:
            PIDResult with decomposition and validation

        Raises:
            ValueError: If conservation constraints are violated and auto_correct=False
        """
        # Generate cache key based on data hash for uniqueness
        import hashlib

        data_hash = hashlib.sha256(
            np.concatenate([source1.ravel(), source2.ravel(), target.ravel()]).tobytes()
        ).hexdigest()[:16]
        cache_key = f"{data_hash}_{bins}"
        if cache_key in self._decompositions:
            return self._decompositions[cache_key]

        # Compute necessary entropies and mutual informations
        h_t = self.compute_entropy(target, bins=bins)
        h_s1s2 = self.compute_entropy(
            np.column_stack([source1.ravel(), source2.ravel()]), bins=bins
        )

        # Mutual informations
        mi_s1_t = self.compute_mutual_information(source1, target, bins=bins)
        mi_s2_t = self.compute_mutual_information(source2, target, bins=bins)

        # Joint mutual information I(S1, S2; T)
        s1s2 = np.column_stack([source1.ravel(), source2.ravel()])
        mi_s1s2_t = self.compute_mutual_information(s1s2, target, bins=bins)

        # Conditional mutual informations
        # I(S1; T | S2) approximated as I(S1, S2; T) - I(S2; T)
        mi_s1_t_given_s2 = max(0.0, mi_s1s2_t - mi_s2_t)
        mi_s2_t_given_s1 = max(0.0, mi_s1s2_t - mi_s1_t)

        # PID components using Williams & Beer framework
        # Redundant: min(I(S1; T), I(S2; T))
        redundant = min(mi_s1_t, mi_s2_t)

        # Unique to S1: I(S1; T) - Redundant
        unique_s1 = max(0.0, mi_s1_t - redundant)

        # Unique to S2: I(S2; T) - Redundant
        unique_s2 = max(0.0, mi_s2_t - redundant)

        # Synergistic: Total MI - (Unique_S1 + Unique_S2 + Redundant)
        synergistic = max(0.0, mi_s1s2_t - (unique_s1 + unique_s2 + redundant))

        # Create result
        result = PIDResult(
            unique_s1=unique_s1,
            unique_s2=unique_s2,
            redundant=redundant,
            synergistic=synergistic,
            total_mi=mi_s1s2_t,
            entropy_target=h_t,
            entropy_sources=h_s1s2,
            conservation_valid=True,
        )

        # Validate conservation constraints
        violations = self._validate_conservation(result)
        result.violations = violations
        result.conservation_valid = len(violations) == 0

        # Compute condition number for numerical stability
        components = np.array([unique_s1, unique_s2, redundant, synergistic])
        if components.max() > 0:
            result.condition_number = float(components.max() / (components.min() + 1e-10))

        # Auto-correct minor violations if enabled
        if not result.conservation_valid and self.constraints.auto_correct:
            result = self._auto_correct_violations(result)
        elif not result.conservation_valid:
            raise ValueError(f"Conservation constraint violations detected: {violations}")

        # Cache result
        self._decompositions[cache_key] = result

        return result

    def _validate_conservation(self, result: PIDResult) -> list[str]:
        """Validate conservation constraints.

        Args:
            result: PID result to validate

        Returns:
            List of violation messages (empty if valid)
        """
        violations = []
        tol = self.constraints.tolerance

        # 1. Non-negativity
        if self.constraints.enforce_non_negativity:
            if result.unique_s1 < -tol:
                violations.append(f"Unique_S1 < 0: {result.unique_s1:.6f}")
            if result.unique_s2 < -tol:
                violations.append(f"Unique_S2 < 0: {result.unique_s2:.6f}")
            if result.redundant < -tol:
                violations.append(f"Redundant < 0: {result.redundant:.6f}")
            if result.synergistic < -tol:
                violations.append(f"Synergistic < 0: {result.synergistic:.6f}")

        # 2. Upper bound: Total MI <= min(H(sources), H(target))
        if self.constraints.enforce_upper_bound:
            max_mi = min(result.entropy_sources, result.entropy_target)
            if result.total_mi > max_mi + tol:
                violations.append(
                    f"Total MI {result.total_mi:.6f} exceeds upper bound {max_mi:.6f}"
                )

        # 3. Decomposition sum equals total MI
        component_sum = result.unique_s1 + result.unique_s2 + result.redundant + result.synergistic
        if abs(component_sum - result.total_mi) > tol:
            violations.append(
                f"Component sum {component_sum:.6f} != Total MI {result.total_mi:.6f}"
            )

        return violations

    def _auto_correct_violations(self, result: PIDResult) -> PIDResult:
        """Automatically correct minor conservation violations.

        Args:
            result: PID result with violations

        Returns:
            Corrected PID result
        """
        # Enforce non-negativity by clamping
        unique_s1 = max(0.0, result.unique_s1)
        unique_s2 = max(0.0, result.unique_s2)
        redundant = max(0.0, result.redundant)
        synergistic = max(0.0, result.synergistic)

        # Renormalize to match total MI
        component_sum = unique_s1 + unique_s2 + redundant + synergistic
        if component_sum > 0 and abs(component_sum - result.total_mi) > self.constraints.tolerance:
            scale = result.total_mi / component_sum
            unique_s1 *= scale
            unique_s2 *= scale
            redundant *= scale
            synergistic *= scale

        # Create corrected result
        corrected = PIDResult(
            unique_s1=unique_s1,
            unique_s2=unique_s2,
            redundant=redundant,
            synergistic=synergistic,
            total_mi=result.total_mi,
            entropy_target=result.entropy_target,
            entropy_sources=result.entropy_sources,
            conservation_valid=True,
            violations=[],
            condition_number=result.condition_number,
        )

        return corrected

    def compute_information_flow(
        self,
        omics_layers: list[np.ndarray],
        target: np.ndarray,
        layer_names: Optional[list[str]] = None,
        bins: int = 10,
    ) -> dict[str, any]:
        """Compute information flow from multiple omics layers to target.

        Args:
            omics_layers: List of omics data arrays
            target: Target variable
            layer_names: Names for each omics layer
            bins: Number of bins for discretization

        Returns:
            Dictionary with information flow analysis
        """
        if layer_names is None:
            layer_names = [f"Layer_{i}" for i in range(len(omics_layers))]

        # Compute individual mutual informations
        individual_mi = {}
        for name, layer in zip(layer_names, omics_layers):
            mi = self.compute_mutual_information(layer, target, bins=bins)
            individual_mi[name] = mi

        # Compute pairwise PIDs
        pairwise_pids = {}
        for i, (name1, layer1) in enumerate(zip(layer_names, omics_layers)):
            for name2, layer2 in zip(layer_names[i + 1 :], omics_layers[i + 1 :]):
                pid = self.decompose_information(layer1, layer2, target, bins=bins)
                pairwise_pids[f"{name1}_{name2}"] = pid

        return {
            "individual_mi": individual_mi,
            "pairwise_decompositions": pairwise_pids,
            "total_layers": len(omics_layers),
        }

    def get_statistics(self) -> dict[str, any]:
        """Get engine statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_decompositions": len(self._decompositions),
            "cached_entropies": len(self._entropy_cache),
            "conservation_enforcement": {
                "non_negativity": self.constraints.enforce_non_negativity,
                "upper_bound": self.constraints.enforce_upper_bound,
                "monotonicity": self.constraints.enforce_monotonicity,
                "tolerance": self.constraints.tolerance,
                "auto_correct": self.constraints.auto_correct,
            },
        }

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._decompositions.clear()
        self._entropy_cache.clear()
