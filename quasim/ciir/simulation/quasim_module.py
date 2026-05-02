r"""QuASIM Tensor Runtime Module: Tensor operations and gradient visualization.

Implements the QuASIM layer of the simulation pipeline:

    QuASIMTensor — Tensor representation with precision modes
    TensorRuntime — Gradient evolution, contraction logic, einsum visualization
    TensorVisualizer — 3D visualization of tensor transformations and loss landscapes

Mathematical Objects (from §4–§5 of formalization)
---------------------------------------------------
State tensor:     X ∈ R^{B × R × D}
Density tensor:   ρ ∈ R^{B × D × D}
Constraint:       C_i(ρ) = Tr(K_i ρ), einsum 'ide,bde->bi'
Observer:         ⟨O_j⟩ = Tr(O_j ρ), einsum 'jde,bde->bj'

Loss functional:
    L(ρ) = Σ λ_i C_i(ρ)² + Σ μ_j |⟨O_j⟩ - y_j|² + γ · Ω(ρ)

Gradient kernel:
    ∇L = Σ 2λ_i C_i K_i + Σ 2μ_j (⟨O_j⟩ - y_j) O_j - γ(log ρ + I)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from quasim.ciir.loss import CIIRLoss
from quasim.ciir.tensors import (batch_constraint_eval, batch_observer_eval,
                                 stack_operators)
from quasim.ciir.theory import CIIRTheory, RealTensor

# ================================================================
# Precision modes
# ================================================================


class PrecisionMode(Enum):
    """Supported numerical precision modes."""
    FP32 = "fp32"
    FP64 = "fp64"


DTYPE_MAP = {
    PrecisionMode.FP32: np.float32,
    PrecisionMode.FP64: np.float64,
}


# ================================================================
# QuASIMTensor — Tensor representation
# ================================================================


@dataclass
class QuASIMTensor:
    r"""Tensor representation with precision control and metadata.

    Wraps NumPy arrays with QuASIM-compatible semantics:
    - Named dimensions for einsum clarity
    - Precision casting
    - Norm/trace/spectrum introspection

    Parameters
    ----------
    data : RealTensor
        The raw tensor data.
    name : str
        Human-readable name (e.g., "state", "constraint_kernel").
    dims : tuple of str
        Named dimensions (e.g., ("batch", "rep", "rep")).
    precision : PrecisionMode
        Numerical precision.
    """

    data: RealTensor
    name: str = ""
    dims: tuple[str, ...] = ()
    precision: PrecisionMode = PrecisionMode.FP64

    def __post_init__(self) -> None:
        self.data = self.data.astype(DTYPE_MAP[self.precision])

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def frobenius_norm(self) -> float:
        """‖T‖_F = √(Σ T²)."""
        return float(np.linalg.norm(self.data))

    @property
    def spectral_norm(self) -> float:
        """Largest singular value (for 2D or batched 2D tensors)."""
        if self.data.ndim == 2:
            return float(np.linalg.norm(self.data, ord=2))
        elif self.data.ndim == 3:
            return float(max(
                np.linalg.norm(self.data[b], ord=2)
                for b in range(self.data.shape[0])
            ))
        return self.frobenius_norm

    def trace(self) -> RealTensor:
        """Trace over last two dimensions."""
        return np.einsum("...dd->...", self.data)

    def spectrum(self) -> RealTensor:
        """Eigenvalues of last two dimensions (must be square)."""
        if self.data.ndim == 2:
            return np.linalg.eigvalsh(self.data)
        elif self.data.ndim == 3:
            return np.array([np.linalg.eigvalsh(self.data[b]) for b in range(self.data.shape[0])])
        raise ValueError(f"Cannot compute spectrum for {self.data.ndim}D tensor")

    def to_dict(self) -> dict[str, Any]:
        """Serialize tensor metadata for export."""
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dims": list(self.dims),
            "precision": self.precision.value,
            "frobenius_norm": self.frobenius_norm,
            "spectral_norm": self.spectral_norm,
        }


# ================================================================
# TensorRuntime — Gradient evolution and contraction logic
# ================================================================


@dataclass
class TensorRuntime:
    r"""QuASIM tensor runtime for CIIR evolution.

    Manages the tensorized CIIR computation:
    - State ↔ density matrix conversion
    - Batched loss and gradient computation
    - Einsum contraction execution with profiling
    - Learning rate scheduling

    Parameters
    ----------
    theory : CIIRTheory
        The CIIR theory tuple.
    loss_fn : CIIRLoss
        The CIIR loss functional.
    precision : PrecisionMode
        Numerical precision.
    """

    theory: CIIRTheory
    loss_fn: CIIRLoss
    precision: PrecisionMode = PrecisionMode.FP64

    # Profiling
    _contraction_log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.K, self.O = stack_operators(self.theory)
        self._contraction_log = []

    def wrap_tensor(self, data: RealTensor, name: str, dims: tuple[str, ...] = ()) -> QuASIMTensor:
        """Wrap raw array as QuASIMTensor."""
        return QuASIMTensor(data=data, name=name, dims=dims, precision=self.precision)

    def compute_loss_and_gradient(
        self, rho: RealTensor
    ) -> tuple[float, RealTensor, dict[str, float]]:
        """Compute full CIIR loss and gradient.

        Returns
        -------
        loss : float
        gradient : array, shape (B, D, D)
        components : dict with 'constraint', 'observer', 'entropy' keys
        """
        loss, grad = self.loss_fn(rho)
        components = self.loss_fn.loss_components(rho)

        # Log contraction
        self._contraction_log.append({
            "loss": loss,
            "grad_norm": float(np.linalg.norm(grad)),
            **components,
        })

        return loss, grad, components

    def constraint_eval(self, rho: RealTensor) -> RealTensor:
        """Batch constraint evaluation. Returns (B, n_c)."""
        if self.K.size == 0:
            return np.empty((rho.shape[0], 0))
        return batch_constraint_eval(rho, self.K)

    def observer_eval(self, rho: RealTensor) -> RealTensor:
        """Batch observer evaluation. Returns (B, n_o)."""
        if self.O.size == 0:
            return np.empty((rho.shape[0], 0))
        return batch_observer_eval(rho, self.O)

    def gradient_step(self, rho: RealTensor, lr: float) -> tuple[RealTensor, float]:
        """Perform one gradient descent step.

        Returns (new_rho, loss).
        """
        loss, grad, _ = self.compute_loss_and_gradient(rho)
        rho_new = rho - lr * grad
        return rho_new, loss

    def loss_landscape_2d(
        self,
        rho_center: RealTensor,
        direction1: RealTensor,
        direction2: RealTensor,
        n_grid: int = 25,
        extent: float = 0.5,
    ) -> tuple[RealTensor, RealTensor, RealTensor]:
        r"""Sample loss landscape on a 2D slice through density operator space.

        Parameters
        ----------
        rho_center : array, shape (D, D)
            Center point.
        direction1, direction2 : array, shape (D, D)
            Orthogonal perturbation directions (symmetric).
        n_grid : int
            Grid resolution per dimension.
        extent : float
            Range along each direction.

        Returns
        -------
        alphas : array, shape (n_grid,)
        betas : array, shape (n_grid,)
        losses : array, shape (n_grid, n_grid)
        """
        alphas = np.linspace(-extent, extent, n_grid)
        betas = np.linspace(-extent, extent, n_grid)
        losses = np.zeros((n_grid, n_grid))

        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                perturbed = rho_center + a * direction1 + b * direction2
                # Quick positivity fix
                eigvals, eigvecs = np.linalg.eigh(perturbed)
                eigvals = np.clip(eigvals, 0, None)
                total = eigvals.sum()
                if total > 1e-12:
                    eigvals /= total
                rho_valid = eigvecs @ np.diag(eigvals) @ eigvecs.T
                loss, _ = self.loss_fn(rho_valid[None, :, :])
                losses[i, j] = loss

        return alphas, betas, losses

    @property
    def contraction_log(self) -> list[dict[str, Any]]:
        """Access contraction profiling log."""
        return self._contraction_log


# ================================================================
# TensorVisualizer — 3D visualization of tensor transformations
# ================================================================


class TensorVisualizer:
    """3D visualization of QuASIM tensor transformations.

    Generates matplotlib figures for:
    - Loss landscape surface/contour plots
    - Gradient field visualization
    - Tensor spectrum evolution
    - Contraction profiling
    """

    def __init__(self, figsize: tuple[int, int] = (12, 8), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi

    def plot_loss_landscape(
        self,
        runtime: TensorRuntime,
        rho_center: RealTensor,
        save_path: str | None = None,
    ) -> Any:
        """Plot 3D loss landscape surface.

        Parameters
        ----------
        runtime : TensorRuntime
        rho_center : array, shape (D, D)
        save_path : str, optional
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        D = rho_center.shape[0]
        rng = np.random.default_rng(0)

        # Generate two random symmetric directions
        d1 = rng.standard_normal((D, D))
        d1 = 0.5 * (d1 + d1.T)
        d1 /= np.linalg.norm(d1)
        d2 = rng.standard_normal((D, D))
        d2 = 0.5 * (d2 + d2.T)
        # Orthogonalise
        d2 -= d1 * np.sum(d1 * d2) / np.sum(d1 * d1)
        d2 /= np.linalg.norm(d2)

        alphas, betas, losses = runtime.loss_landscape_2d(
            rho_center, d1, d2, n_grid=20, extent=0.3,
        )

        A, B = np.meshgrid(alphas, betas, indexing="ij")

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(A, B, losses, cmap="coolwarm", alpha=0.8, edgecolor="none")
        ax.set_xlabel("Direction 1")
        ax.set_ylabel("Direction 2")
        ax.set_zlabel("Loss L(ρ)")
        ax.set_title("CIIR Loss Landscape")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_gradient_norms(
        self,
        contraction_log: list[dict[str, Any]],
        save_path: str | None = None,
    ) -> Any:
        """Plot gradient norm evolution.

        Parameters
        ----------
        contraction_log : list of dicts from TensorRuntime
        save_path : str, optional
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)

        steps = range(len(contraction_log))
        losses = [e["loss"] for e in contraction_log]
        grad_norms = [e["grad_norm"] for e in contraction_log]

        ax1.plot(steps, losses, "b-", linewidth=1)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Convergence")
        ax1.grid(True, alpha=0.3)

        ax2.semilogy(steps, grad_norms, "r-", linewidth=1)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("‖∇L‖")
        ax2.set_title("Gradient Norm")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_loss_components(
        self,
        contraction_log: list[dict[str, Any]],
        save_path: str | None = None,
    ) -> Any:
        """Plot individual loss components over time.

        Parameters
        ----------
        contraction_log : list of dicts
        save_path : str, optional
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        steps = range(len(contraction_log))
        for key in ["constraint", "observer", "entropy"]:
            values = [e.get(key, 0) for e in contraction_log]
            ax.plot(steps, values, label=key, linewidth=1)

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss Component")
        ax.set_title("CIIR Loss Components")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_tensor_spectrum(
        self,
        tensors: list[QuASIMTensor],
        save_path: str | None = None,
    ) -> Any:
        """Plot spectra of a sequence of tensors (e.g., density matrices over time).

        Parameters
        ----------
        tensors : list of QuASIMTensor
        save_path : str, optional
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        for i, t in enumerate(tensors):
            try:
                spec = t.spectrum()
                if spec.ndim == 1:
                    ax.plot(spec, "o-", markersize=3, label=f"t={i}", alpha=0.7)
                elif spec.ndim == 2:
                    ax.plot(spec[0], "o-", markersize=3, label=f"t={i}", alpha=0.7)
            except ValueError:
                continue

        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Eigenvalue")
        ax.set_title("Tensor Spectrum Evolution")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return fig
