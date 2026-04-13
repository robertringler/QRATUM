r"""CIIR Module: State manifolds, observers, and constraint visualization.

Implements the CIIR layer of the simulation pipeline:

    CIIRState — State representation with manifold geometry
    CIIRObserver — Observer interface with sequential measurement tracking
    CIIRVisualizer — 3D rendering of CIIR manifolds and constraints

Mathematical Objects (from Ch3–Ch5)
------------------------------------
State manifold M ⊆ R^n as tensor space T^{r,d}:
    X ∈ R^{B × R × D}, ρ = X^T X / Tr(X^T X)

Constraint surface:
    S_C = {ρ ∈ D(H_C) : C_i(ρ) = 0 ∀i}

Observer representation:
    O_j ∈ A(H_C), spectral decomposition O_j = Σ_o a_o P_o
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quasim.ciir.observers import PVM, born_distribution, measurement_update
from quasim.ciir.theory import (
    CIIRTheory,
    RealTensor,
)

# ================================================================
# CIIRState — State representation with manifold geometry
# ================================================================


@dataclass
class CIIRState:
    r"""CIIR state representation with manifold embedding.

    Tracks the cognitive state ρ ∈ D(H_C) along with derived
    geometric quantities used for 3D visualization:

    - Eigenvalues (spectrum of ρ) for colour mapping
    - Constraint evaluations C_i(ρ) for surface rendering
    - Purity Tr(ρ²) for opacity control
    - Von Neumann entropy S(ρ) for radial scaling

    Parameters
    ----------
    rho : array, shape (B, D, D)
        Batch of density matrices.
    theory : CIIRTheory
        The underlying CIIR theory tuple.
    """

    rho: RealTensor
    theory: CIIRTheory
    step: int = 0

    # Cached derived quantities
    _eigenvalues: RealTensor | None = field(default=None, repr=False)
    _constraint_values: RealTensor | None = field(default=None, repr=False)
    _purity: RealTensor | None = field(default=None, repr=False)
    _entropy: RealTensor | None = field(default=None, repr=False)

    def invalidate_cache(self) -> None:
        """Clear cached quantities after state update."""
        self._eigenvalues = None
        self._constraint_values = None
        self._purity = None
        self._entropy = None

    @property
    def batch_size(self) -> int:
        return self.rho.shape[0]

    @property
    def dim(self) -> int:
        return self.rho.shape[1]

    @property
    def eigenvalues(self) -> RealTensor:
        """Spectrum of ρ for each batch element. Shape (B, D)."""
        if self._eigenvalues is None:
            self._eigenvalues = np.linalg.eigvalsh(self.rho)
        return self._eigenvalues

    @property
    def constraint_values(self) -> RealTensor:
        """C_i(ρ) for each constraint and batch element. Shape (B, n_c)."""
        if self._constraint_values is None:
            vals = []
            for c in self.theory.constraints:
                vals.append(c.evaluate(self.rho))
            if vals:
                self._constraint_values = np.stack(vals, axis=1)
            else:
                self._constraint_values = np.empty((self.batch_size, 0))
        return self._constraint_values

    @property
    def constraint_violations(self) -> RealTensor:
        """C_i(ρ)² for each constraint and batch element. Shape (B, n_c)."""
        return self.constraint_values**2

    @property
    def purity(self) -> RealTensor:
        """Tr(ρ²) ∈ [1/D, 1]. Shape (B,)."""
        if self._purity is None:
            self._purity = np.einsum("bde,bed->b", self.rho, self.rho)
        return self._purity

    @property
    def entropy(self) -> RealTensor:
        """S(ρ) = −Tr(ρ log ρ). Shape (B,)."""
        if self._entropy is None:
            eigs = np.clip(self.eigenvalues, 1e-12, None)
            self._entropy = -np.sum(eigs * np.log(eigs), axis=1)
        return self._entropy

    def manifold_embedding_3d(self) -> RealTensor:
        r"""Project state onto 3D coordinates for visualization.

        Uses the first three eigenvalues of ρ as manifold coordinates.
        For B states, returns (B, 3) array.

        This corresponds to the spectral embedding of D(H_C) into R³:
            φ(ρ) = (λ₁, λ₂, λ₃)
        where λ_i are the largest eigenvalues of ρ.
        """
        eigs = self.eigenvalues  # (B, D)
        # Take the 3 largest eigenvalues
        if eigs.shape[1] >= 3:
            return eigs[:, -3:]
        # Pad with zeros if fewer than 3 dims
        pad = np.zeros((eigs.shape[0], 3 - eigs.shape[1]))
        return np.concatenate([pad, eigs], axis=1)

    def constraint_surface_samples(self, n_samples: int = 100) -> RealTensor:
        r"""Sample points near the constraint surface S_C for rendering.

        Generates perturbed versions of the current state and evaluates
        constraint values to map the constraint landscape.

        Returns
        -------
        points : array, shape (n_samples, 3)
            3D coordinates of surface samples.
        values : array, shape (n_samples,)
            Total constraint violation at each point.
        """
        rng = np.random.default_rng(self.step)
        D = self.dim

        # Perturb around first batch element
        base = self.rho[0]
        points_3d = []
        violations = []

        for _ in range(n_samples):
            # Random perturbation in density operator space
            delta = rng.standard_normal((D, D)) * 0.1
            delta = 0.5 * (delta + delta.T)
            perturbed = base + delta
            # Project to valid density matrix
            eigvals, eigvecs = np.linalg.eigh(perturbed)
            eigvals = np.clip(eigvals, 0, None)
            total = eigvals.sum()
            if total > 1e-12:
                eigvals /= total
            perturbed = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # 3D embedding
            eigs = np.sort(eigvals)[-3:]
            points_3d.append(eigs)

            # Constraint violation
            rho_batch = perturbed[None, :, :]
            cv = sum(float(c.violation(rho_batch).mean()) for c in self.theory.constraints)
            violations.append(cv)

        return np.array(points_3d), np.array(violations)

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for export."""
        return {
            "step": self.step,
            "batch_size": self.batch_size,
            "dim": self.dim,
            "purity": self.purity.tolist(),
            "entropy": self.entropy.tolist(),
            "constraint_values": self.constraint_values.tolist(),
            "eigenvalues": self.eigenvalues.tolist(),
        }


# ================================================================
# CIIRObserver — Observer interface with measurement tracking
# ================================================================


@dataclass
class CIIRObserver:
    """Observer interface with sequential measurement tracking.

    Wraps CIIR observer operators with visualization-friendly methods
    for tracking measurement outcomes, Born probabilities, and
    state collapse trajectories.

    Parameters
    ----------
    theory : CIIRTheory
        The CIIR theory tuple.
    """

    theory: CIIRTheory
    measurement_history: list[dict[str, Any]] = field(default_factory=list)

    def observe(self, state: CIIRState, observer_index: int) -> dict[str, Any]:
        """Perform observation and record results.

        Parameters
        ----------
        state : CIIRState
            Current state.
        observer_index : int
            Index of the observer operator to apply.

        Returns
        -------
        result : dict
            Measurement result with probabilities and post-measurement state.
        """
        obs = self.theory.observers[observer_index]
        pvm = PVM.from_observable(obs)

        # Born probabilities
        probs = born_distribution(state.rho[0], pvm)

        # Most probable outcome
        best = int(np.argmax(probs))
        rho_post = measurement_update(state.rho[0], pvm.projectors[best])

        result = {
            "observer": obs.name or f"O{observer_index}",
            "step": state.step,
            "probabilities": probs.tolist(),
            "outcome": best,
            "outcome_value": float(pvm.eigenvalues[best]),
            "pre_entropy": float(state.entropy[0]),
            "post_entropy": float(
                -np.sum(
                    np.clip(np.linalg.eigvalsh(rho_post), 1e-12, None)
                    * np.log(np.clip(np.linalg.eigvalsh(rho_post), 1e-12, None))
                )
            ),
        }
        self.measurement_history.append(result)
        return result

    def commutator_matrix(self) -> RealTensor:
        """Compute the commutator norm matrix ‖[O_i, O_j]‖ for all pairs.

        Returns
        -------
        comm_norms : array, shape (n_obs, n_obs)
        """
        n = len(self.theory.observers)
        norms = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                comm = self.theory.observers[i].commutator(self.theory.observers[j])
                norm = float(np.linalg.norm(comm))
                norms[i, j] = norm
                norms[j, i] = norm
        return norms

    def expectation_values(self, state: CIIRState) -> RealTensor:
        """Compute ⟨O_j⟩_ρ for all observers and batch elements.

        Returns
        -------
        expectations : array, shape (B, n_obs)
        """
        exps = []
        for obs in self.theory.observers:
            exps.append(obs.expectation(state.rho))
        if exps:
            return np.stack(exps, axis=1)
        return np.empty((state.batch_size, 0))


# ================================================================
# CIIRVisualizer — 3D rendering of CIIR manifolds and constraints
# ================================================================


class CIIRVisualizer:
    """3D visualization of CIIR manifolds, constraints, and observer effects.

    Generates matplotlib 3D figures for:
    - State manifold with eigenvalue embedding
    - Constraint surface heatmap
    - Observer measurement trajectories
    - Commutator structure visualization
    """

    def __init__(self, figsize: tuple[int, int] = (12, 8), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi

    def plot_manifold_trajectory(
        self,
        states: list[CIIRState],
        save_path: str | None = None,
    ) -> Any:
        """Plot 3D trajectory of state evolution on the manifold.

        Parameters
        ----------
        states : list of CIIRState
            Evolution trajectory.
        save_path : str, optional
            Path to save figure.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection="3d")

        for b in range(states[0].batch_size):
            coords = np.array([s.manifold_embedding_3d()[b] for s in states])
            colors = np.linspace(0, 1, len(coords))
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=colors,
                cmap="viridis",
                s=10,
                alpha=0.7,
            )
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                alpha=0.3,
                linewidth=0.5,
            )

        ax.set_xlabel("λ₁ (eigenvalue)")
        ax.set_ylabel("λ₂")
        ax.set_zlabel("λ₃")
        ax.set_title("CIIR State Manifold Trajectory")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_constraint_surface(
        self,
        state: CIIRState,
        n_samples: int = 200,
        save_path: str | None = None,
    ) -> Any:
        """Plot 3D constraint surface heatmap.

        Parameters
        ----------
        state : CIIRState
            Current state (used as center for sampling).
        n_samples : int
            Number of surface samples.
        save_path : str, optional
            Path to save figure.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection="3d")

        points, violations = state.constraint_surface_samples(n_samples)
        # Normalise violation for color
        v_norm = violations / max(violations.max(), 1e-12)

        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=v_norm,
            cmap="hot_r",
            s=20,
            alpha=0.6,
        )
        fig.colorbar(scatter, ax=ax, label="Constraint Violation", shrink=0.6)

        # Mark current state
        current = state.manifold_embedding_3d()
        ax.scatter(
            current[:, 0],
            current[:, 1],
            current[:, 2],
            c="blue",
            s=100,
            marker="*",
            label="Current State",
        )

        ax.set_xlabel("λ₁")
        ax.set_ylabel("λ₂")
        ax.set_zlabel("λ₃")
        ax.set_title("CIIR Constraint Surface")
        ax.legend()
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_observer_commutators(
        self,
        observer: CIIRObserver,
        save_path: str | None = None,
    ) -> Any:
        """Plot commutator norm matrix as heatmap.

        Parameters
        ----------
        observer : CIIRObserver
        save_path : str, optional
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        norms = observer.commutator_matrix()
        im = ax.imshow(norms, cmap="Reds", interpolation="nearest")
        fig.colorbar(im, ax=ax, label="‖[Oᵢ, Oⱼ]‖")

        n = norms.shape[0]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        labels = [o.name or f"O{i}" for i, o in enumerate(observer.theory.observers)]
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_title("Observer Commutator Norms [Oᵢ, Oⱼ]")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_eigenvalue_evolution(
        self,
        states: list[CIIRState],
        batch_idx: int = 0,
        save_path: str | None = None,
    ) -> Any:
        """Plot eigenvalue evolution over time.

        Parameters
        ----------
        states : list of CIIRState
        batch_idx : int
            Which batch element to plot.
        save_path : str, optional
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        steps = [s.step for s in states]
        eigs = np.array([s.eigenvalues[batch_idx] for s in states])

        for d in range(eigs.shape[1]):
            ax.plot(steps, eigs[:, d], label=f"λ_{d}", alpha=0.7)

        ax.set_xlabel("Evolution Step")
        ax.set_ylabel("Eigenvalue")
        ax.set_title("Density Matrix Spectrum Evolution")
        ax.legend(fontsize=8, ncol=min(eigs.shape[1], 4))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return fig
