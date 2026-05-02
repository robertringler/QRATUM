"""
SEER-Medicare Integration Module

This module provides tools for processing SEER-Medicare linked data
to build treatment timelines and integrate with QRATUM oncology analysis.

RESEARCH USE ONLY - Not for clinical diagnosis or treatment decisions.

Key Components:
- io: File discovery, manifest generation, safe readers
- schema: Normalized data models (ClaimEvent, RegistryCase)
- seer_registry: SEER case file parsing
- medicare_claims: Medicare claims adapters (MEDPAR, NCH, OUTSAF, PDE)
- linkages: Patient key mapping, eligibility windows
- cohort: Cohort building with inclusion/exclusion criteria
- timelines: Treatment timeline construction
- features: Feature engineering for VITRA/XENON
- privacy: Safe logging and redaction utilities
- quality: Data validation and QA metrics

DUA Compliance:
- All processing runs locally; no network upload of data
- Patient-level identifiers are never logged
- Aggregated counts suppress cells < 11 by default
- Run artifacts include deterministic hashes for reproducibility
"""

from __future__ import annotations

__all__ = [
    "io",
    "schema",
    "seer_registry",
    "medicare_claims",
    "linkages",
    "cohort",
    "timelines",
    "features",
    "privacy",
    "quality",
]
