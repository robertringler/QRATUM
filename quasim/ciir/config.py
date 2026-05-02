"""CIIR configuration schema and YAML support."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CIIRConfig:
    """Configuration for a CIIR evolution run.

    Attributes
    ----------
    rank : int
        Ontic dimension (R in D3.1).
    rep_dim : int
        Cognitive representation dimension (D in D3.11).
    batch_size : int
        Number of parallel states to evolve.
    n_constraints : int
        Number of constraint operators.
    n_observers : int
        Number of observer operators.
    n_steps : int
        Evolution steps.
    lr : float
        Learning rate / time step.
    method : str
        Integration method: gradient | projected_gradient | strang_splitting.
    entropy_weight : float
        Regularisation coefficient γ.
    constraint_weights : list[float] | None
        Per-constraint λ_i (defaults to all-ones).
    observer_weights : list[float] | None
        Per-observer μ_j (defaults to all-ones).
    observer_targets : list[float] | None
        Target values y_j (defaults to all-zeros).
    kappa_min : float
        Curvature lower bound K_min (D3.8).
    diameter : float
        Constraint manifold diameter D.
    seed : int
        Random seed.
    precision : str
        Numerical precision: fp32 | fp64.
    output_path : str
        Output directory for results.
    record_every : int
        Record state every N steps.
    tol : float
        Convergence tolerance.
    """

    rank: int = 4
    rep_dim: int = 8
    batch_size: int = 4
    n_constraints: int = 3
    n_observers: int = 2
    n_steps: int = 200
    lr: float = 0.01
    method: str = "projected_gradient"
    entropy_weight: float = 0.01
    constraint_weights: list[float] | None = None
    observer_weights: list[float] | None = None
    observer_targets: list[float] | None = None
    kappa_min: float = 1.0
    diameter: float = 1.0
    seed: int = 42
    precision: str = "fp64"
    output_path: str = "ciir_output"
    record_every: int = 10
    tol: float = 1e-6

    def to_yaml(self, path: str | Path) -> None:
        """Write configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CIIRConfig:
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
