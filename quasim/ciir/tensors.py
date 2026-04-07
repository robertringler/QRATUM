r"""Tensorization: map all CIIR objects into batch-compatible tensor algebra.

This module provides the critical §4 tensorization layer, expressing all
CIIR operators using einsum-style tensor contractions suitable for
QuASIM runtime execution.

Tensor Layout Convention
------------------------
State tensor:     X ∈ R^{B × R × D}  (batch × rank × representation)
Density tensor:   ρ ∈ R^{B × D × D}  (batch × rep × rep)
Constraint tensor: K_i ∈ R^{D × D}
Observer tensor:  O_j ∈ R^{D × D}  (or C^{D × D} for complex)
Interface tensor: W ∈ R^{D_out × D_in}

Precision modes: fp32, fp64 (fp8/fp16 via truncation on GPU backends).

All operations are expressed as np.einsum calls for direct translation
to GPU tensor libraries (torch.einsum, jax.numpy.einsum, tf.einsum).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quasim.ciir.theory import CIIRTheory, RealTensor, ComplexTensor


# ================================================================
# Batch density matrix construction
# ================================================================


def state_to_density(X: RealTensor) -> RealTensor:
    r"""ρ_b = X_b^T X_b / Tr(X_b^T X_b).

    einsum: 'brd,bre->bde' then normalise.

    Parameters
    ----------
    X : array, shape (B, R, D)

    Returns
    -------
    rho : array, shape (B, D, D)
    """
    rho = np.einsum("brd,bre->bde", X, X)
    traces = np.einsum("bdd->b", rho)
    return rho / np.maximum(traces[:, None, None], 1e-12)


# ================================================================
# Constraint evaluation (batch)
# ================================================================


def batch_constraint_eval(rho: RealTensor, K: RealTensor) -> RealTensor:
    r"""C_i(ρ) = Tr(K_i ρ) for all constraints simultaneously.

    einsum: 'ide,bde->bi'

    Parameters
    ----------
    rho : array, shape (B, D, D)
    K : array, shape (n_constraints, D, D)
        Stacked constraint kernels.

    Returns
    -------
    violations : array, shape (B, n_constraints)
    """
    return np.einsum("ide,bde->bi", K, rho)


def batch_constraint_violation(rho: RealTensor, K: RealTensor) -> RealTensor:
    r"""C_i(ρ)² for all constraints.  Shape (B, n_constraints)."""
    c = batch_constraint_eval(rho, K)
    return c ** 2


def batch_constraint_gradient(
    rho: RealTensor, K: RealTensor, weights: RealTensor
) -> RealTensor:
    r"""∂(Σ λ_i C_i²)/∂ρ = Σ 2λ_i C_i(ρ) K_i.

    einsum: 'bi,i,ide->bde'

    Parameters
    ----------
    rho : array, shape (B, D, D)
    K : array, shape (n_constraints, D, D)
    weights : array, shape (n_constraints,)

    Returns
    -------
    grad : array, shape (B, D, D)
    """
    c = batch_constraint_eval(rho, K)  # (B, n_c)
    weighted_c = 2.0 * c * weights[None, :]  # (B, n_c)
    return np.einsum("bi,ide->bde", weighted_c, K)


# ================================================================
# Observer evaluation (batch)
# ================================================================


def batch_observer_eval(rho: RealTensor, O: RealTensor | ComplexTensor) -> RealTensor:
    r"""⟨O_j⟩_ρ = Tr(O_j ρ) for all observers.

    einsum: 'jde,bde->bj'

    Parameters
    ----------
    rho : array, shape (B, D, D)
    O : array, shape (n_observers, D, D)

    Returns
    -------
    expectations : array, shape (B, n_observers)
    """
    return np.real(np.einsum("jde,bde->bj", O, rho))


def batch_observer_gradient(
    rho: RealTensor,
    O: RealTensor | ComplexTensor,
    targets: RealTensor,
    weights: RealTensor,
) -> RealTensor:
    r"""∂L_O/∂ρ = Σ_j 2μ_j (⟨O_j⟩ − y_j) O_j.

    Parameters
    ----------
    rho : array, shape (B, D, D)
    O : array, shape (n_observers, D, D)
    targets : array, shape (n_observers,)
    weights : array, shape (n_observers,)

    Returns
    -------
    grad : array, shape (B, D, D)
    """
    exp = batch_observer_eval(rho, O)  # (B, n_obs)
    residual = exp - targets[None, :]  # (B, n_obs)
    weighted_res = 2.0 * residual * weights[None, :]  # (B, n_obs)
    return np.einsum("bj,jde->bde", weighted_res, np.real(O))


# ================================================================
# Commutator tensor
# ================================================================


def commutator_tensor(A: RealTensor, B_mat: RealTensor) -> RealTensor:
    r"""[A, B] = A B − B A.

    For stacked operators:  result[i,j] = [A_i, B_j].

    Parameters
    ----------
    A : array, shape (nA, D, D)
    B_mat : array, shape (nB, D, D)

    Returns
    -------
    comm : array, shape (nA, nB, D, D)
    """
    # AB_{ij} = A_i B_j
    AB = np.einsum("ide,jef->ijdf", A, B_mat)
    BA = np.einsum("jde,ief->ijdf", B_mat, A)  # reorder for BA
    # Fix: BA_{ij} = B_j A_i
    BA_correct = np.einsum("jde,ief->jidf", B_mat, A)
    return AB - BA_correct


# ================================================================
# Interface map (batch tensor)
# ================================================================


def batch_interface_map(X: RealTensor, W: RealTensor) -> RealTensor:
    r"""Π(X) = normalise(W · X).

    einsum: 'od,brd->bro'

    Parameters
    ----------
    X : array, shape (B, R, D_in)
    W : array, shape (D_out, D_in)

    Returns
    -------
    Y : array, shape (B, R, D_out)
    """
    Y = np.einsum("od,brd->bro", W, X)
    norms = np.linalg.norm(Y, axis=(1, 2), keepdims=True)
    return Y / np.maximum(norms, 1e-12)


# ================================================================
# Tensorized loss (full batch)
# ================================================================


def tensorized_ciir_loss(
    rho: RealTensor,
    K: RealTensor,
    O: RealTensor | ComplexTensor,
    constraint_weights: RealTensor,
    observer_weights: RealTensor,
    observer_targets: RealTensor,
    entropy_weight: float = 0.01,
) -> tuple[float, RealTensor]:
    r"""Full CIIR loss in pure tensor form.

    L = Σ λ_i C_i² + Σ μ_j |⟨O_j⟩ − y_j|² − γ Σ S(ρ_b)

    Parameters
    ----------
    rho : (B, D, D)
    K : (n_c, D, D)
    O : (n_o, D, D)
    constraint_weights : (n_c,)
    observer_weights : (n_o,)
    observer_targets : (n_o,)
    entropy_weight : float

    Returns
    -------
    loss : float
    grad : (B, D, D)
    """
    # Constraint term
    cv = batch_constraint_violation(rho, K)  # (B, n_c)
    lc = float(np.einsum("bi,i->", cv, constraint_weights))
    gc = batch_constraint_gradient(rho, K, constraint_weights)

    # Observer term
    exp = batch_observer_eval(rho, O)  # (B, n_o)
    residual = exp - observer_targets[None, :]
    lo = float(np.einsum("bj,bj,j->", residual, residual, observer_weights))
    go = batch_observer_gradient(rho, O, observer_targets, observer_weights)

    # Entropy term (vectorised via eigendecomposition)
    B = rho.shape[0]
    le = 0.0
    ge = np.zeros_like(rho)
    for b in range(B):
        eigvals, eigvecs = np.linalg.eigh(rho[b])
        eigvals = np.clip(eigvals, 1e-12, None)
        S = -np.sum(eigvals * np.log(eigvals))
        le -= entropy_weight * S
        log_diag = -(np.log(eigvals) + 1.0)
        ge[b] = -entropy_weight * (eigvecs @ np.diag(log_diag) @ eigvecs.T)

    return lc + lo + le, gc + go + ge


def stack_operators(theory: CIIRTheory) -> tuple[RealTensor, RealTensor | ComplexTensor]:
    """Stack constraint and observer matrices into batch tensors.

    Returns
    -------
    K : array, shape (n_constraints, D, D)
    O : array, shape (n_observers, D, D)
    """
    K = np.stack([c.matrix for c in theory.constraints], axis=0) if theory.constraints else np.empty((0, 0, 0))
    O = np.stack([o.matrix for o in theory.observers], axis=0) if theory.observers else np.empty((0, 0, 0))
    return K, O
