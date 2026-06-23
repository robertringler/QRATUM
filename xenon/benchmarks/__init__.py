"""Head-to-head benchmarks for XENON mechanism inference (directive Phase 4).

These compare XENON against independent gold-standard tools:
  * simulation correctness vs libRoadRunner (SBML ODE engine);
  * Bayesian parameter recovery vs pyABC.

They are the evidence required before any "industry-leading" claim. The heavy
reference tools (roadrunner, pyabc) are optional; importing this package does not
require them - each comparison imports its reference lazily.
"""
