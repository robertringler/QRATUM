r"""Constraint-induced distortion map with metric contraction bounds.

Implements Ch8 §8.4–§8.5 of the CIIR monograph:

Distortion map (D8.7, T8.4):
    Φ: R → H_C
    d_{H_C}(Φ(r₁), Φ(r₂)) ≤ d_R(r₁, r₂) · exp(−K_min · D)

Non-invertibility (T8.5):
    ∃ r₁ ≠ r₂ : Φ(r₁) = Φ(r₂)

The distortion is *structural*, not accidental: the constraint geometry
induces information compression that necessarily loses distinguishability
between ontic states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quasim.ciir.theory import InterfaceMap, RealTensor


@dataclass
class DistortionAnalysis:
    """Results of distortion map analysis.

    Attributes
    ----------
    d_source : array
        Pairwise distances in reality space R.
    d_image : array
        Pairwise distances in cognitive space H_C.
    contraction_bound : array
        Upper bound d_R · exp(-K_min · D) per pair.
    ratio : array
        d_image / d_source (should be ≤ exp(-K_min · D)).
    bound_satisfied : bool
        Whether T8.4 contraction bound holds for all pairs.
    non_invertible_pairs : list of tuple
        Pairs (i, j) where Φ(r_i) = Φ(r_j) but r_i ≠ r_j (T8.5).
    """

    d_source: RealTensor
    d_image: RealTensor
    contraction_bound: RealTensor
    ratio: RealTensor
    bound_satisfied: bool
    non_invertible_pairs: list[tuple[int, int]]


def pairwise_distances(X: RealTensor) -> RealTensor:
    r"""Compute pairwise Frobenius distances between batch elements.

    Parameters
    ----------
    X : array, shape (N, ...)

    Returns
    -------
    D : array, shape (N, N)
    """
    N = X.shape[0]
    flat = X.reshape(N, -1)
    # ||x_i - x_j||² = ||x_i||² + ||x_j||² - 2 x_i · x_j
    sq_norms = np.sum(flat**2, axis=1)
    gram = flat @ flat.T
    D_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * gram
    return np.sqrt(np.maximum(D_sq, 0))


def analyse_distortion(
    interface: InterfaceMap,
    X_samples: RealTensor,
    collapse_tol: float = 1e-8,
) -> DistortionAnalysis:
    r"""Analyse the constraint-induced distortion of the interface map.

    Verifies T8.4 (contraction bound) and T8.5 (non-invertibility)
    on a sample of ontic states.

    Parameters
    ----------
    interface : InterfaceMap
        The CIIR interface map Π.
    X_samples : array, shape (N, R, D)
        Sample of ontic state tensors from R.
    collapse_tol : float
        Tolerance for declaring Φ(r₁) = Φ(r₂) in T8.5.

    Returns
    -------
    DistortionAnalysis
    """
    N = X_samples.shape[0]

    # Apply interface map
    Y_samples = interface.apply(X_samples)

    # Pairwise distances
    d_source = pairwise_distances(X_samples)
    d_image = pairwise_distances(Y_samples)

    # Contraction bound: d_image ≤ d_source · exp(-K_min · D)
    alpha = interface.contraction_factor
    contraction_bound = d_source * alpha

    # Ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(d_source > 1e-12, d_image / d_source, 0.0)

    # Check bound (upper triangle only, excluding diagonal)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    bound_satisfied = bool(np.all(d_image[mask] <= contraction_bound[mask] + 1e-10))

    # Non-invertibility: find pairs with d_image ≈ 0 but d_source > 0
    non_invertible = []
    for i in range(N):
        for j in range(i + 1, N):
            if d_source[i, j] > 1e-8 and d_image[i, j] < collapse_tol:
                non_invertible.append((i, j))

    return DistortionAnalysis(
        d_source=d_source,
        d_image=d_image,
        contraction_bound=contraction_bound,
        ratio=ratio,
        bound_satisfied=bound_satisfied,
        non_invertible_pairs=non_invertible,
    )


def distortion_entropy(
    interface: InterfaceMap,
    X_samples: RealTensor,
    n_bins: int = 50,
) -> float:
    r"""Compute the distortion entropy H_Φ (D8.21).

    H_Φ = −Σ_k p_k log p_k

    where p_k is the fraction of pairwise distance ratios
    d_image/d_source falling in bin k.

    A higher entropy indicates more uniform distortion;
    lower entropy indicates concentrated information loss.
    """
    Y_samples = interface.apply(X_samples)
    d_source = pairwise_distances(X_samples)
    d_image = pairwise_distances(Y_samples)

    mask = np.triu(np.ones_like(d_source, dtype=bool), k=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(d_source[mask] > 1e-12, d_image[mask] / d_source[mask], 0.0)

    counts, _ = np.histogram(ratios, bins=n_bins, range=(0, 1))
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))
