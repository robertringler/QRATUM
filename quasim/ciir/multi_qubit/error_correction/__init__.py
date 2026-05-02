"""Error Correction Agent — fractal recursive quantum error correction."""

from .fractal_qec import (
    FractalQEC,
    steane_logical_error_rate,
    fractal_recursion,
    fit_fractal_scaling,
)

__all__ = [
    "FractalQEC",
    "steane_logical_error_rate",
    "fractal_recursion",
    "fit_fractal_scaling",
]
