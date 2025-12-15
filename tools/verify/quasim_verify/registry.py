"""Check registry mapping IDs to implementations."""

from collections.abc import Callable

from .checks import (audit_chain, comp_artifacts, comp_cmmc_map,
                     doc_language_lint, econ_montecarlo, econ_phi_qevf,
                     tech_benchmarks, tech_compression, tech_rl_convergence,
                     tech_telemetry_rmse)

# Registry mapping check IDs to their runner functions
REG: dict[str, Callable] = {
    "TECH-001": tech_benchmarks.run,
    "TECH-002": tech_telemetry_rmse.run,
    "TECH-003": tech_rl_convergence.run,
    "TECH-004": tech_compression.run,
    "ECON-201": econ_phi_qevf.run,
    "ECON-202": econ_montecarlo.run,
    "COMP-101": comp_artifacts.run,
    "COMP-102": comp_cmmc_map.run,
    "GOV-301": doc_language_lint.run,
    "DOC-401": audit_chain.run,
}
