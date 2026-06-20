"""POC precision model — shared shot-noise quantization for all RQ simulations.

The POC reads out optical amplitudes via homodyne detection with finite
signal-to-noise ratio. With N_ph photons per mode per shot and S-shot
averaging, the effective per-amplitude precision is delta ~ 1/sqrt(N_ph * S).
We model this as additive Gaussian noise on each amplitude plus a uniform
quantization to P bits, which is the conservative engineering abstraction
used throughout POC-MONO-2.0 (Phase 3 precision model) and POC-APPS-1.0.

This module is imported by rq1, rq2, rq3 so the precision abstraction is
identical across every benchmark — a single source of truth.
"""
from __future__ import annotations

import numpy as np


def bits_to_sigma(precision_bits: float, dynamic_range: float = 1.0) -> float:
    """Effective additive noise std-dev for a given precision in bits.

    A P-bit converter over a symmetric dynamic range [-R, R] has quantization
    step q = 2R / 2^P and quantization-noise std q/sqrt(12). For the POC the
    shot noise dominates and is comparable in magnitude, so we take the noise
    std as one LSB (a conservative ~factor sqrt(12) pessimism vs pure
    quantization), i.e. sigma = 2*dynamic_range / 2^P.
    """
    return 2.0 * dynamic_range / (2.0 ** precision_bits)


def apply_poc_readout(
    x: np.ndarray,
    precision_bits: float,
    rng: np.random.Generator,
    dynamic_range: float | None = None,
) -> np.ndarray:
    """Simulate a POC measurement of a (real or complex) vector/matrix x.

    Adds shot noise at the given precision then quantizes to the P-bit grid.
    For complex inputs the real and imaginary parts are read out independently
    (two homodyne quadratures), each at the stated precision.
    """
    if dynamic_range is None:
        dynamic_range = float(np.max(np.abs(x))) or 1.0
    sigma = bits_to_sigma(precision_bits, dynamic_range)
    q = 2.0 * dynamic_range / (2.0 ** precision_bits)

    def _quant_real(v: np.ndarray) -> np.ndarray:
        noisy = v + rng.normal(0.0, sigma, size=v.shape)
        return np.round(noisy / q) * q

    if np.iscomplexobj(x):
        return _quant_real(x.real) + 1j * _quant_real(x.imag)
    return _quant_real(x)


def shots_for_bits(precision_bits: float) -> int:
    """Shot budget implied by the precision wall: S ~ 4^P (Vol III Thm 11).

    delta ~ 1/sqrt(S)  =>  S ~ (1/delta)^2 = (2^P)^2 = 4^P. Returned as an int
    for resource accounting in the result tables.
    """
    return int(round(4.0 ** precision_bits))
