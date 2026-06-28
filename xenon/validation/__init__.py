"""Closed-loop validation harnesses for XENON.

The in-silico recovery harness generates experimental observations from a known
ground-truth mechanism and verifies that XENON's (corrected) Bayesian inference
recovers it — turning the previously mechanism-independent mock loop into a real,
falsifiable validation of the inference engine.
"""

from xenon.validation.insilico_recovery import (
    RecoveryResult,
    make_two_state_mechanism,
    predict_steady_state,
    run_recovery,
)

__all__ = [
    "RecoveryResult",
    "make_two_state_mechanism",
    "predict_steady_state",
    "run_recovery",
]
