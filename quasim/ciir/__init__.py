"""CIIR-QuASIM Bridge: Executable mapping from Constraint-Induced Interface Realism to QuASIM tensors.

This package implements the constructive derivation:

    CIIR (formal theory) → Tensorized operators → Numerical evolution
    → QuASIM runtime integration → QRATUM system hooks

Formally:
    Φ: T_CIIR → (L, O, U) → QuASIM tensors → Executable modules

where T_CIIR = (M, {C_i}, {O_j}, Π) is the CIIR theory tuple.

Modules
-------
theory : Axiomatic foundation — state manifold, constraint/observer operators, interface map
loss : Variational formulation — CIIR loss functional derived from axioms
evolution : State evolution — projected gradient dynamics with operator splitting
tensors : Tensorization — batch-compatible tensor algebra for all CIIR objects
observers : Non-commutative observer formalism with sequential measurement effects
distortion : Constraint-induced distortion map with metric contraction bounds
benchmarks : Validation suite for constraint satisfaction, interference, convergence
cli : CLI entry point ``quasim-ciir``
config : YAML configuration schema

References
----------
CIIR Monograph Chapters 3–8:
    Ch3 D3.1–D3.21 (primitive definitions)
    Ch4 Ax4.1–Ax4.6 (axiom system)
    Ch5 D5.1–D5.12, T5.1–T5.4 (algebraic structure)
    Ch6 D6.1–D6.16, T6.1–T6.8 (representation theory)
    Ch7 D7.1–D7.17, T7.1–T7.7 (dynamics)
    Ch8 D8.1–D8.22, T8.1–T8.9 (measurement theory)
"""

__version__ = "0.1.0"
