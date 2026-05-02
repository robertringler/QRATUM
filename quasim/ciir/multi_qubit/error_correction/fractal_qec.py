r"""Error Correction Agent — fractal recursive quantum error correction.

Implements the fractal recursion of Ch21:

    p_L(n+1) = p_* ( p_L(n) / p_* )^α

with α > 1 below threshold (super-exponential suppression) and α < 1 above.

The base layer is an abstracted [[7,1,3]] Steane-like code whose logical
error rate under independent depolarising noise at physical rate p is:

    p_L^(1) ≈ C_d · (p / p_*)^{⌊(d+1)/2⌋}

with d = 3 (minimum distance), so the leading term is p²/p_* (up to
combinatorial constants).  Here we use the standard approximation:

    p_L^(1)(p) = A · p^t

with A = C(7,4) (combinatorial weight), t = ⌊(d+1)/2⌋ = 2, p_* = 1/A.

References
----------
* Gottesman (1997) — Stabilizer Codes and Quantum Error Correction
* Aharonov & Ben-Or (1999) — Fault-Tolerant Quantum Computation
* Ch21, CIIR monograph — Theorem 21.2 (Threshold Theorem)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# [[7,1,3]] Steane-code inspired single-level logical error rate
# ---------------------------------------------------------------------------

def steane_logical_error_rate(p_phys: float) -> float:
    r"""Compute logical error rate for one level of a [[7,1,3]] code.

    p_L = C(7,4) · p^2 · (1-p)^5 + … ≈ 21 p² for small p.

    This counts all weight-2 error patterns (which cannot be corrected by
    a distance-3 code).

    Parameters
    ----------
    p_phys : physical gate error probability per qubit

    Returns
    -------
    float — logical error probability per logical gate
    """
    from math import comb

    # Weight-2 patterns (and above) that cannot be corrected
    p_L = sum(
        comb(7, k) * (p_phys ** k) * ((1 - p_phys) ** (7 - k))
        for k in range(2, 8)  # k ≥ t+1 = 2
    )
    return float(np.clip(p_L, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Fractal recursion
# ---------------------------------------------------------------------------

def fractal_recursion(
    p_phys: float,
    n_levels: int,
    p_star: Optional[float] = None,
    alpha: Optional[float] = None,
) -> List[float]:
    r"""Compute logical error rates p_L(0), p_L(1), …, p_L(n_levels).

    Uses the closed-form recursion:

        p_L(n+1) = p_* ( p_L(n) / p_* )^α

    If p_star and alpha are not provided, they are estimated from the
    single-level code formula and the ratio p_L(1) / p_phys^2.

    Parameters
    ----------
    p_phys   : physical error rate
    n_levels : number of concatenation levels
    p_star   : threshold (estimated if None)
    alpha    : exponent > 1 (estimated if None)

    Returns
    -------
    list of length n_levels + 1
    """
    p0 = p_phys
    p1 = steane_logical_error_rate(p_phys)

    if p_star is None:
        # Threshold: p_L(1) = p_phys when p_phys = p_star  →  solve numerically
        from scipy.optimize import brentq
        try:
            p_star = brentq(lambda p: steane_logical_error_rate(p) - p, 1e-8, 0.5)
        except Exception:
            p_star = 1 / 21.0  # analytic estimate

    if alpha is None:
        if p_phys < p_star and p0 > 0:
            alpha = max(1.01, np.log(p1 / p_star) / np.log(p0 / p_star))
        else:
            alpha = 0.5  # above threshold

    rates = [p0]
    p_n = p0
    for _ in range(n_levels):
        ratio = p_n / p_star
        if ratio <= 0:
            p_n = 0.0
        else:
            sign = 1.0 if ratio > 0 else -1.0
            p_n = p_star * (abs(ratio) ** alpha)
        p_n = float(np.clip(p_n, 0.0, 1.0))
        rates.append(p_n)

    return rates


# ---------------------------------------------------------------------------
# Fit scaling law: log p_L(n) ~ α^n · const
# ---------------------------------------------------------------------------

def fit_fractal_scaling(rates: List[float]) -> Tuple[float, float]:
    r"""Fit log p_L(n) = A · α^n + B to the recursion data.

    Returns
    -------
    (alpha_fit, r_squared) : estimated exponent and coefficient of determination
    """
    n_levels = len(rates) - 1
    if n_levels < 2:
        return (1.0, 0.0)

    log_rates = []
    ns = []
    for i, r in enumerate(rates[1:], start=1):  # skip level 0
        if r > 0:
            log_rates.append(np.log(r))
            ns.append(i)

    if len(log_rates) < 2:
        return (1.0, 0.0)

    ns_arr = np.array(ns, dtype=float)
    log_arr = np.array(log_rates, dtype=float)

    # Fit log|log p_L(n)| ~ n log alpha + const  (double-log method)
    log_log = np.log(np.abs(log_arr) + 1e-30)
    coeffs = np.polyfit(ns_arr, log_log, 1)
    alpha_fit = float(np.exp(coeffs[0]))

    # R²
    predicted = np.polyval(coeffs, ns_arr)
    ss_res = np.sum((log_log - predicted) ** 2)
    ss_tot = np.sum((log_log - log_log.mean()) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-30))

    return (alpha_fit, r2)


# ---------------------------------------------------------------------------
# High-level FractalQEC class
# ---------------------------------------------------------------------------

@dataclass
class FractalQECResult:
    """Results of a fractal QEC simulation."""

    p_phys: float
    p_star: float
    alpha: float
    rates: List[float]          # p_L at each level (0 = physical)
    alpha_fit: float
    r_squared: float
    below_threshold: bool


class FractalQEC:
    """Fractal recursive quantum error correction engine.

    Simulates logical error rates across multiple concatenation levels
    and verifies super-exponential suppression below threshold.

    Parameters
    ----------
    n_levels : int — maximum concatenation depth (default 8)
    """

    def __init__(self, n_levels: int = 8) -> None:
        self.n_levels = n_levels
        # Compute p_star once
        from scipy.optimize import brentq
        try:
            self._p_star = brentq(
                lambda p: steane_logical_error_rate(p) - p, 1e-8, 0.5
            )
        except Exception:
            self._p_star = 1 / 21.0

    @property
    def p_star(self) -> float:
        return self._p_star

    def run(self, p_phys: float) -> FractalQECResult:
        """Simulate fractal recursion for given physical error rate."""
        rates = fractal_recursion(p_phys, self.n_levels, p_star=self._p_star)
        alpha_fit, r2 = fit_fractal_scaling(rates)
        below = p_phys < self._p_star

        # Compute alpha from recursion formula at level 1
        if p_phys < self._p_star and p_phys > 0:
            p1 = steane_logical_error_rate(p_phys)
            if p1 > 0 and p_phys > 0:
                alpha_est = np.log(p1 / self._p_star) / np.log(p_phys / self._p_star)
            else:
                alpha_est = 2.0
        else:
            alpha_est = 0.5

        return FractalQECResult(
            p_phys=p_phys,
            p_star=self._p_star,
            alpha=float(alpha_est),
            rates=rates,
            alpha_fit=alpha_fit,
            r_squared=r2,
            below_threshold=below,
        )

    def sweep_threshold(
        self,
        p_range: Optional[NDArray] = None,
    ) -> Tuple[NDArray, NDArray]:
        """Sweep p_phys and return (p_phys_array, p_L_final_array)."""
        if p_range is None:
            p_range = np.logspace(-4, np.log10(0.3), 50)
        p_finals = []
        for p in p_range:
            res = self.run(float(p))
            p_finals.append(res.rates[-1])
        return p_range, np.array(p_finals)


# Fix missing import
from typing import Optional
