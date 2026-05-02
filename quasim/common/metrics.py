"""Metrics for simulation quality and fidelity.

Provides standard metrics for evaluating simulation results:
- RMSE: Root mean squared error
- Wasserstein distance
- Bures fidelity for quantum states
- PR-AUC for classification tasks
"""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        RMSE value
    """

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MAE value
    """

    return float(np.mean(np.abs(y_true - y_pred)))


def wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
    """Compute 1D Wasserstein distance between two distributions.

    Args:
        u: First distribution samples
        v: Second distribution samples

    Returns:
        Wasserstein-1 distance
    """

    u_sorted = np.sort(u)
    v_sorted = np.sort(v)

    # Ensure same length for comparison
    n = min(len(u_sorted), len(v_sorted))
    u_sorted = u_sorted[:n]
    v_sorted = v_sorted[:n]

    return float(np.mean(np.abs(u_sorted - v_sorted)))


def bures_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute Bures fidelity between two density matrices.

    Fidelity F(ρ,σ) = (Tr[√(√ρ σ √ρ)])²

    Args:
        rho: First density matrix
        sigma: Second density matrix

    Returns:
        Bures fidelity (0 to 1)
    """

    # Compute sqrt(rho)
    eigvals_rho, eigvecs_rho = np.linalg.eigh(rho)
    eigvals_rho = np.maximum(eigvals_rho, 0)  # Ensure non-negative
    sqrt_rho = eigvecs_rho @ np.diag(np.sqrt(eigvals_rho)) @ eigvecs_rho.T.conj()

    # Compute sqrt(rho) @ sigma @ sqrt(rho)
    M = sqrt_rho @ sigma @ sqrt_rho

    # Compute eigenvalues of M
    eigvals_M = np.linalg.eigvalsh(M)
    eigvals_M = np.maximum(eigvals_M, 0)

    # Fidelity is (sum of sqrt of eigenvalues)^2
    fidelity = np.sum(np.sqrt(eigvals_M)) ** 2

    return float(np.clip(fidelity, 0, 1))


def pr_auc(y_true: np.ndarray, y_scores: np.ndarray, n_thresholds: int = 100) -> float:
    """Compute area under precision-recall curve.

    Args:
        y_true: Binary labels (0 or 1)
        y_scores: Predicted scores
        n_thresholds: Number of thresholds to evaluate

    Returns:
        PR-AUC value
    """

    thresholds = np.linspace(0, 1, n_thresholds)
    precisions = []
    recalls = []

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    # Compute AUC using trapezoidal rule
    recalls = np.array(recalls)
    precisions = np.array(precisions)

    # Sort by recall
    idx = np.argsort(recalls)
    recalls = recalls[idx]
    precisions = precisions[idx]

    auc = np.trapz(precisions, recalls)
    return float(auc)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MSE value
    """

    return float(np.mean((y_true - y_pred) ** 2))


def relative_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Compute mean relative error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small constant to avoid division by zero

    Returns:
        Mean relative error
    """

    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))))
