"""
QRATUM Oncology Research Engine - VITRA/XENON Integration

A frontier-level computational oncology research framework that leverages:
- VITRA: Symbolic-causal reasoning for cancer mechanism modeling
- XENON: Deterministic asymmetric adaptive search for intervention optimization

CRITICAL DISCLAIMER:
═══════════════════════════════════════════════════════════════════════════════
This module is FOR RESEARCH USE ONLY. It does NOT:
  ❌ Claim to cure cancer
  ❌ Provide medical advice for patients
  ❌ Replace clinical judgment or FDA-approved treatments

All outputs are:
  ✅ Hypotheses requiring experimental validation
  ✅ Pre-clinical research strategies
  ✅ Computational mechanism discoveries
  ✅ Treatment optimization frameworks for research

This system is a DECISION-SUPPORT TOOL for qualified researchers and
clinicians, NOT a replacement for professional medical care.
═══════════════════════════════════════════════════════════════════════════════

Core Components:
1. CausalOncologyGraph: VITRA-integrated symbolic causal graphs of oncogenesis
2. XENONInterventionSearch: Adaptive search for intervention optimization
3. ValidationPipeline: In-silico → In-vitro → In-vivo → Clinical pathway
4. EthicsComplianceModule: FDA SaMD, AI safety, and regulatory compliance
5. HypothesisPortfolio: Ranked hypotheses with risk analysis
"""

__version__ = "0.1.0"
__author__ = "QRATUM Oncology Research Consortium"

from .causal_graph import (
    CausalOncologyGraph,
    OncogenicNode,
    CausalEdge,
    MutationState,
    EpigeneticState,
    ImmuneEvasionMechanism,
    TumorMicroenvironment,
)
from .intervention_search import (
    XENONInterventionSearch,
    InterventionNode,
    TreatmentSequence,
    AdaptiveTherapyPlan,
)
from .hypothesis_portfolio import (
    HypothesisPortfolio,
    ResearchHypothesis,
    RiskAnalysis,
)
from .validation_pipeline import (
    ValidationPipeline,
    InSilicoValidation,
    InVitroValidation,
    InVivoValidation,
    ClinicalTranslation,
)
from .ethics_compliance import (
    EthicsComplianceModule,
    FDACompliance,
    AISafetyConstraints,
)
from .research_roadmap import (
    ResearchRoadmap,
    Milestone,
    ResearchPhase,
)

__all__ = [
    # Core graph models
    "CausalOncologyGraph",
    "OncogenicNode",
    "CausalEdge",
    "MutationState",
    "EpigeneticState",
    "ImmuneEvasionMechanism",
    "TumorMicroenvironment",
    # Intervention search
    "XENONInterventionSearch",
    "InterventionNode",
    "TreatmentSequence",
    "AdaptiveTherapyPlan",
    # Hypothesis management
    "HypothesisPortfolio",
    "ResearchHypothesis",
    "RiskAnalysis",
    # Validation
    "ValidationPipeline",
    "InSilicoValidation",
    "InVitroValidation",
    "InVivoValidation",
    "ClinicalTranslation",
    # Ethics and compliance
    "EthicsComplianceModule",
    "FDACompliance",
    "AISafetyConstraints",
    # Roadmap
    "ResearchRoadmap",
    "Milestone",
    "ResearchPhase",
]

# Safety disclaimer that must be acknowledged
SAFETY_DISCLAIMER = """
════════════════════════════════════════════════════════════════════════════════
                    QRATUM ONCOLOGY RESEARCH ENGINE
                         MANDATORY SAFETY NOTICE
════════════════════════════════════════════════════════════════════════════════

This computational framework is designed EXCLUSIVELY for:
  • Pre-clinical research hypothesis generation
  • Treatment optimization modeling for research purposes
  • Mechanism discovery and pathway analysis
  • Clinical trial design support

THIS IS NOT:
  • A diagnostic tool
  • A treatment recommendation system
  • A substitute for qualified oncologist judgment
  • FDA-approved for clinical decision-making

All outputs require:
  1. Peer review by domain experts
  2. Experimental validation (in-vitro, in-vivo)
  3. Regulatory approval before any clinical application
  4. Informed consent protocols for any human studies

Use of this system for direct patient care without appropriate clinical
oversight and regulatory approval is PROHIBITED.

════════════════════════════════════════════════════════════════════════════════
"""


def get_disclaimer() -> str:
    """Return the safety disclaimer."""
    return SAFETY_DISCLAIMER


def acknowledge_disclaimer() -> bool:
    """
    Programmatic acknowledgment of disclaimer.

    In production use, this should be connected to an interactive
    confirmation system.
    """
    # Log disclaimer acknowledgment
    import logging

    logging.info("QRATUM Oncology Research Engine disclaimer acknowledged")
    return True
