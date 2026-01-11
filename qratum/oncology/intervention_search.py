"""
XENON Intervention Search - Adaptive Treatment Planning

This module implements the XENON (deterministic asymmetric adaptive search)
engine for exploring intervention trees in oncology research.

Key capabilities:
1. Multi-step intervention sequence optimization
2. Drug combination search with synergy modeling
3. Timing/dosage schedule exploration
4. Resistance suppression strategies
5. Immune re-engagement pathways

The search optimizes for:
- Resistance suppression
- Tumor entropy reduction
- Immune re-engagement
- Minimal toxicity paths

RESEARCH USE ONLY - Not for clinical treatment decisions.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class InterventionType(Enum):
    """Types of therapeutic interventions."""

    TARGETED_THERAPY = "targeted_therapy"
    IMMUNOTHERAPY = "immunotherapy"
    CHEMOTHERAPY = "chemotherapy"
    RADIATION = "radiation"
    METABOLIC_DISRUPTION = "metabolic_disruption"
    EPIGENETIC_REPROGRAMMING = "epigenetic_reprogramming"
    COMBINATION = "combination"
    ADAPTIVE = "adaptive"


class ResistanceMechanism(Enum):
    """Known resistance mechanisms."""

    TARGET_MUTATION = "target_mutation"
    BYPASS_PATHWAY = "bypass_pathway"
    EFFLUX_PUMP = "efflux_pump"
    EPIGENETIC_ADAPTATION = "epigenetic_adaptation"
    METABOLIC_REWIRING = "metabolic_rewiring"
    IMMUNE_EVASION = "immune_evasion"
    MICROENVIRONMENT = "microenvironment"


@dataclass
class DrugProfile:
    """Profile of a therapeutic agent.

    Attributes:
        drug_id: Unique identifier
        name: Drug name
        target_genes: Primary gene targets
        mechanism: Mechanism of action
        intervention_type: Type of intervention
        efficacy_score: Baseline efficacy (0.0 to 1.0)
        toxicity_score: Toxicity level (0.0 to 1.0)
        resistance_mechanisms: Known resistance mechanisms
        synergistic_drugs: Drugs with synergistic effects
        antagonistic_drugs: Drugs with antagonistic effects
    """

    drug_id: str
    name: str
    target_genes: list[str] = field(default_factory=list)
    mechanism: str = ""
    intervention_type: InterventionType = InterventionType.TARGETED_THERAPY
    efficacy_score: float = 0.5
    toxicity_score: float = 0.3
    resistance_mechanisms: list[ResistanceMechanism] = field(default_factory=list)
    synergistic_drugs: list[str] = field(default_factory=list)
    antagonistic_drugs: list[str] = field(default_factory=list)
    half_life_hours: float = 24.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "drug_id": self.drug_id,
            "name": self.name,
            "target_genes": self.target_genes,
            "mechanism": self.mechanism,
            "intervention_type": self.intervention_type.value,
            "efficacy_score": self.efficacy_score,
            "toxicity_score": self.toxicity_score,
            "resistance_mechanisms": [r.value for r in self.resistance_mechanisms],
            "synergistic_drugs": self.synergistic_drugs,
            "antagonistic_drugs": self.antagonistic_drugs,
            "half_life_hours": self.half_life_hours,
            "metadata": self.metadata,
        }


@dataclass
class InterventionNode:
    """A node in the intervention search tree.

    Represents a state after applying an intervention.

    Attributes:
        node_id: Unique identifier
        parent_id: Parent node ID (None for root)
        intervention: Drug profile applied at this step
        dosage: Dosage level (normalized 0.0 to 1.0)
        timing_week: Week number in treatment schedule
        tumor_state: Estimated tumor state after intervention
        resistance_probability: Probability of resistance emerging
        immune_engagement: Level of immune system engagement
        toxicity_accumulated: Cumulative toxicity
        depth: Depth in search tree
    """

    node_id: str
    parent_id: Optional[str] = None
    intervention: Optional[DrugProfile] = None
    dosage: float = 1.0
    timing_week: int = 0
    tumor_state: dict[str, float] = field(default_factory=dict)
    resistance_probability: float = 0.0
    immune_engagement: float = 0.0
    toxicity_accumulated: float = 0.0
    depth: int = 0
    children: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "intervention": self.intervention.to_dict() if self.intervention else None,
            "dosage": self.dosage,
            "timing_week": self.timing_week,
            "tumor_state": self.tumor_state,
            "resistance_probability": self.resistance_probability,
            "immune_engagement": self.immune_engagement,
            "toxicity_accumulated": self.toxicity_accumulated,
            "depth": self.depth,
            "children": self.children,
            "metadata": self.metadata,
        }


@dataclass
class TreatmentSequence:
    """A sequence of interventions.

    Represents a complete treatment plan as a sequence of
    intervention nodes from root to leaf.

    Attributes:
        sequence_id: Unique identifier
        interventions: Ordered list of (drug, dosage, timing) tuples
        total_efficacy: Estimated cumulative efficacy
        total_toxicity: Estimated cumulative toxicity
        resistance_suppression_score: How well resistance is suppressed
        immune_engagement_score: Final immune engagement level
        rationale: Explanation for this sequence
    """

    sequence_id: str
    interventions: list[tuple[DrugProfile, float, int]] = field(default_factory=list)
    total_efficacy: float = 0.0
    total_toxicity: float = 0.0
    resistance_suppression_score: float = 0.0
    immune_engagement_score: float = 0.0
    rationale: str = ""
    risk_factors: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sequence_id": self.sequence_id,
            "interventions": [
                {"drug": d.to_dict(), "dosage": dose, "timing_week": week}
                for d, dose, week in self.interventions
            ],
            "total_efficacy": self.total_efficacy,
            "total_toxicity": self.total_toxicity,
            "resistance_suppression_score": self.resistance_suppression_score,
            "immune_engagement_score": self.immune_engagement_score,
            "rationale": self.rationale,
            "risk_factors": self.risk_factors,
            "provenance": self.provenance,
        }


@dataclass
class AdaptiveTherapyPlan:
    """An adaptive therapy plan with decision rules.

    Adaptive therapy adjusts treatment based on tumor response,
    aiming for long-term disease control rather than eradication.

    Attributes:
        plan_id: Unique identifier
        initial_sequence: Starting treatment sequence
        decision_rules: Rules for treatment modification
        monitoring_schedule: Biomarker monitoring schedule
        adjustment_thresholds: Thresholds for treatment changes
        expected_outcomes: Predicted outcomes under different scenarios
    """

    plan_id: str
    initial_sequence: TreatmentSequence
    decision_rules: list[dict[str, Any]] = field(default_factory=list)
    monitoring_schedule: dict[str, int] = field(default_factory=dict)
    adjustment_thresholds: dict[str, float] = field(default_factory=dict)
    expected_outcomes: dict[str, dict[str, float]] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "plan_id": self.plan_id,
            "initial_sequence": self.initial_sequence.to_dict(),
            "decision_rules": self.decision_rules,
            "monitoring_schedule": self.monitoring_schedule,
            "adjustment_thresholds": self.adjustment_thresholds,
            "expected_outcomes": self.expected_outcomes,
            "rationale": self.rationale,
        }


class XENONInterventionSearch:
    """
    XENON adaptive search engine for intervention optimization.

    Implements deterministic look-ahead search over:
    - Drug combinations
    - Timing sequences
    - Dosage schedules
    - Adaptive therapy strategies

    Optimization objectives:
    - Resistance suppression
    - Tumor entropy reduction
    - Immune re-engagement
    - Minimal toxicity paths

    RESEARCH USE ONLY - Not for clinical treatment decisions.
    """

    RESEARCH_DISCLAIMER = """
    This intervention search is for RESEARCH PURPOSES ONLY.
    Outputs require experimental validation and clinical trials.
    Not approved for clinical treatment planning.
    """

    def __init__(
        self,
        seed: int = 42,
        max_depth: int = 6,
        max_toxicity: float = 0.8,
        resistance_penalty: float = 2.0,
    ) -> None:
        """Initialize the intervention search engine.

        Args:
            seed: Random seed for reproducibility
            max_depth: Maximum depth of search tree
            max_toxicity: Maximum allowed cumulative toxicity
            resistance_penalty: Penalty weight for resistance emergence
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.max_depth = max_depth
        self.max_toxicity = max_toxicity
        self.resistance_penalty = resistance_penalty

        self._drug_library: dict[str, DrugProfile] = {}
        self._search_tree: dict[str, InterventionNode] = {}
        self._root_id: Optional[str] = None

        self.created_at = datetime.now(timezone.utc).isoformat()
        self.search_history: list[dict[str, Any]] = []

        logger.info(
            f"Initialized XENONInterventionSearch (seed={seed}, max_depth={max_depth})"
        )

    def add_drug(self, drug: DrugProfile) -> None:
        """Add a drug to the search library.

        Args:
            drug: DrugProfile to add
        """
        self._drug_library[drug.drug_id] = drug
        logger.debug(f"Added drug {drug.drug_id} to library")

    def get_drug(self, drug_id: str) -> Optional[DrugProfile]:
        """Get a drug by ID."""
        return self._drug_library.get(drug_id)

    def initialize_search(
        self,
        initial_tumor_state: dict[str, float],
    ) -> str:
        """Initialize the search tree with a root node.

        Args:
            initial_tumor_state: Initial tumor state parameters

        Returns:
            Root node ID
        """
        root_id = self._generate_node_id()
        root_node = InterventionNode(
            node_id=root_id,
            tumor_state=initial_tumor_state,
            depth=0,
        )

        self._search_tree = {root_id: root_node}
        self._root_id = root_id

        logger.info(f"Initialized search tree with root {root_id}")
        return root_id

    def expand_node(
        self,
        node_id: str,
        candidate_drugs: Optional[list[str]] = None,
        dosage_levels: list[float] = None,
    ) -> list[str]:
        """Expand a node with possible interventions.

        Args:
            node_id: Node to expand
            candidate_drugs: List of drug IDs to consider (None = all)
            dosage_levels: Dosage levels to try (default [0.5, 0.75, 1.0])

        Returns:
            List of child node IDs
        """
        if dosage_levels is None:
            dosage_levels = [0.5, 0.75, 1.0]

        parent = self._search_tree.get(node_id)
        if parent is None:
            raise ValueError(f"Node {node_id} not found")

        if parent.depth >= self.max_depth:
            logger.debug(f"Node {node_id} at max depth, not expanding")
            return []

        if candidate_drugs is None:
            candidate_drugs = list(self._drug_library.keys())

        child_ids = []
        for drug_id in candidate_drugs:
            drug = self._drug_library.get(drug_id)
            if drug is None:
                continue

            for dosage in dosage_levels:
                child = self._create_child_node(parent, drug, dosage)

                # Skip if toxicity exceeds limit
                if child.toxicity_accumulated > self.max_toxicity:
                    continue

                self._search_tree[child.node_id] = child
                parent.children.append(child.node_id)
                child_ids.append(child.node_id)

        return child_ids

    def _create_child_node(
        self,
        parent: InterventionNode,
        drug: DrugProfile,
        dosage: float,
    ) -> InterventionNode:
        """Create a child node by applying an intervention.

        Args:
            parent: Parent node
            drug: Drug to apply
            dosage: Dosage level

        Returns:
            New child node
        """
        child_id = self._generate_node_id()

        # Compute new tumor state
        new_tumor_state = self._compute_tumor_response(
            parent.tumor_state, drug, dosage, parent.resistance_probability
        )

        # Compute resistance probability
        new_resistance = self._compute_resistance_probability(
            parent.resistance_probability,
            drug,
            dosage,
            parent.depth,
        )

        # Compute immune engagement
        new_immune = self._compute_immune_engagement(
            parent.immune_engagement,
            drug,
            dosage,
        )

        # Compute cumulative toxicity
        new_toxicity = parent.toxicity_accumulated + drug.toxicity_score * dosage

        return InterventionNode(
            node_id=child_id,
            parent_id=parent.node_id,
            intervention=drug,
            dosage=dosage,
            timing_week=parent.timing_week + 4,  # 4-week cycles
            tumor_state=new_tumor_state,
            resistance_probability=new_resistance,
            immune_engagement=new_immune,
            toxicity_accumulated=new_toxicity,
            depth=parent.depth + 1,
        )

    def _compute_tumor_response(
        self,
        tumor_state: dict[str, float],
        drug: DrugProfile,
        dosage: float,
        resistance_prob: float,
    ) -> dict[str, float]:
        """Compute tumor state response to intervention.

        Args:
            tumor_state: Current tumor state
            drug: Applied drug
            dosage: Dosage level
            resistance_prob: Current resistance probability

        Returns:
            New tumor state after intervention
        """
        new_state = dict(tumor_state)

        # Basic response model
        efficacy = drug.efficacy_score * dosage * (1 - resistance_prob)

        # Update tumor burden
        if "tumor_burden" in new_state:
            new_state["tumor_burden"] *= 1 - efficacy * 0.5

        # Update proliferation rate
        if "proliferation_rate" in new_state:
            new_state["proliferation_rate"] *= 1 - efficacy * 0.3

        # Add drug-specific effects
        for target in drug.target_genes:
            target_key = f"{target}_activity"
            if target_key in new_state:
                new_state[target_key] *= 1 - efficacy * 0.7

        return new_state

    def _compute_resistance_probability(
        self,
        current_prob: float,
        drug: DrugProfile,
        dosage: float,
        depth: int,
    ) -> float:
        """Compute probability of resistance emerging.

        Args:
            current_prob: Current resistance probability
            drug: Applied drug
            dosage: Dosage level
            depth: Current search depth

        Returns:
            New resistance probability
        """
        # Resistance increases with each treatment cycle
        base_increase = 0.05 * (1 + depth * 0.2)

        # Drug-specific resistance risk
        drug_risk = len(drug.resistance_mechanisms) * 0.02

        # Higher dosage can select for resistant clones
        dosage_factor = dosage * 0.1

        new_prob = current_prob + base_increase + drug_risk + dosage_factor
        return min(new_prob, 1.0)

    def _compute_immune_engagement(
        self,
        current_engagement: float,
        drug: DrugProfile,
        dosage: float,
    ) -> float:
        """Compute immune system engagement level.

        Args:
            current_engagement: Current immune engagement
            drug: Applied drug
            dosage: Dosage level

        Returns:
            New immune engagement level
        """
        if drug.intervention_type == InterventionType.IMMUNOTHERAPY:
            # Immunotherapy increases engagement
            increase = drug.efficacy_score * dosage * 0.3
            return min(current_engagement + increase, 1.0)
        elif drug.intervention_type == InterventionType.CHEMOTHERAPY:
            # Chemotherapy may suppress immunity
            decrease = drug.toxicity_score * dosage * 0.1
            return max(current_engagement - decrease, 0.0)
        else:
            # Other therapies have neutral or minor effects
            return current_engagement

    def _generate_node_id(self) -> str:
        """Generate a unique node ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        random_part = self.rng.randint(0, 1000000)
        raw = f"{timestamp}-{random_part}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def search_best_sequences(
        self,
        initial_tumor_state: dict[str, float],
        n_sequences: int = 10,
        beam_width: int = 20,
    ) -> list[TreatmentSequence]:
        """Search for best treatment sequences using beam search.

        Args:
            initial_tumor_state: Initial tumor state
            n_sequences: Number of sequences to return
            beam_width: Beam width for search

        Returns:
            List of best TreatmentSequences
        """
        self.initialize_search(initial_tumor_state)

        # Priority queue: (negative_score, node_id)
        frontier = [(0.0, self._root_id)]
        completed = []

        while frontier and len(completed) < n_sequences * 2:
            # Expand best nodes
            new_frontier = []

            for _, node_id in frontier[:beam_width]:
                node = self._search_tree.get(node_id)
                if node is None:
                    continue

                if node.depth >= self.max_depth:
                    # Terminal node - create sequence
                    sequence = self._extract_sequence(node)
                    completed.append(sequence)
                    continue

                # Expand node
                child_ids = self.expand_node(node_id)

                for child_id in child_ids:
                    child = self._search_tree.get(child_id)
                    if child:
                        score = self._compute_node_score(child)
                        heapq.heappush(new_frontier, (-score, child_id))

            frontier = sorted(new_frontier)[:beam_width]

        # Score and rank completed sequences
        completed.sort(
            key=lambda s: s.total_efficacy
            - s.total_toxicity
            + s.resistance_suppression_score,
            reverse=True,
        )

        # Log search summary
        self.search_history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "initial_state": initial_tumor_state,
                "sequences_found": len(completed),
                "nodes_explored": len(self._search_tree),
            }
        )

        return completed[:n_sequences]

    def _compute_node_score(self, node: InterventionNode) -> float:
        """Compute heuristic score for a node.

        Args:
            node: Node to score

        Returns:
            Heuristic score (higher is better)
        """
        # Tumor burden reduction
        tumor_score = 1.0 - node.tumor_state.get("tumor_burden", 1.0)

        # Resistance penalty
        resistance_score = 1.0 - node.resistance_probability * self.resistance_penalty

        # Immune engagement bonus
        immune_score = node.immune_engagement

        # Toxicity penalty
        toxicity_score = 1.0 - node.toxicity_accumulated

        # Combined score
        return tumor_score * 0.3 + resistance_score * 0.3 + immune_score * 0.2 + toxicity_score * 0.2

    def _extract_sequence(self, leaf_node: InterventionNode) -> TreatmentSequence:
        """Extract treatment sequence from leaf to root.

        Args:
            leaf_node: Leaf node of the sequence

        Returns:
            TreatmentSequence from root to leaf
        """
        sequence_id = self._generate_node_id()
        interventions = []
        current = leaf_node

        # Walk back to root
        while current.parent_id is not None:
            if current.intervention is not None:
                interventions.append(
                    (current.intervention, current.dosage, current.timing_week)
                )
            current = self._search_tree.get(current.parent_id)
            if current is None:
                break

        # Reverse to get root-to-leaf order
        interventions.reverse()

        # Compute aggregate scores
        total_efficacy = 1.0 - leaf_node.tumor_state.get("tumor_burden", 1.0)
        total_toxicity = leaf_node.toxicity_accumulated
        resistance_suppression = 1.0 - leaf_node.resistance_probability
        immune_engagement = leaf_node.immune_engagement

        # Generate rationale
        rationale = self._generate_rationale(interventions, leaf_node)

        return TreatmentSequence(
            sequence_id=sequence_id,
            interventions=interventions,
            total_efficacy=total_efficacy,
            total_toxicity=total_toxicity,
            resistance_suppression_score=resistance_suppression,
            immune_engagement_score=immune_engagement,
            rationale=rationale,
            provenance=[f"XENON_search_{self.seed}"],
        )

    def _generate_rationale(
        self,
        interventions: list[tuple[DrugProfile, float, int]],
        final_node: InterventionNode,
    ) -> str:
        """Generate rationale for treatment sequence.

        Args:
            interventions: List of interventions
            final_node: Final state node

        Returns:
            Rationale string
        """
        if not interventions:
            return "No interventions in sequence."

        parts = []
        parts.append("Treatment sequence rationale:")

        for i, (drug, dosage, week) in enumerate(interventions):
            parts.append(
                f"  {i + 1}. Week {week}: {drug.name} at {dosage * 100:.0f}% dosage"
            )
            parts.append(f"     - Targets: {', '.join(drug.target_genes)}")
            parts.append(f"     - Type: {drug.intervention_type.value}")

        parts.append(
            f"\nPredicted outcomes:"
            f"\n  - Tumor burden reduction: {(1 - final_node.tumor_state.get('tumor_burden', 1)) * 100:.1f}%"
            f"\n  - Resistance probability: {final_node.resistance_probability * 100:.1f}%"
            f"\n  - Immune engagement: {final_node.immune_engagement * 100:.1f}%"
            f"\n  - Cumulative toxicity: {final_node.toxicity_accumulated * 100:.1f}%"
        )

        return "\n".join(parts)

    def create_adaptive_plan(
        self,
        initial_sequence: TreatmentSequence,
    ) -> AdaptiveTherapyPlan:
        """Create an adaptive therapy plan from a treatment sequence.

        Args:
            initial_sequence: Initial treatment sequence

        Returns:
            AdaptiveTherapyPlan with decision rules
        """
        plan_id = self._generate_node_id()

        # Define decision rules based on sequence
        decision_rules = [
            {
                "condition": "tumor_burden_increase > 20%",
                "action": "escalate_dosage",
                "rationale": "Counter early resistance signal",
            },
            {
                "condition": "toxicity_grade >= 3",
                "action": "reduce_dosage_25%",
                "rationale": "Manage adverse events while maintaining efficacy",
            },
            {
                "condition": "stable_disease_12_weeks",
                "action": "consider_drug_holiday",
                "rationale": "Adaptive therapy - reduce selection pressure",
            },
            {
                "condition": "confirmed_resistance_mutation",
                "action": "switch_agent",
                "rationale": "Address molecular resistance mechanism",
            },
        ]

        # Define monitoring schedule
        monitoring_schedule = {
            "imaging": 8,  # Every 8 weeks
            "liquid_biopsy": 4,  # Every 4 weeks
            "blood_markers": 2,  # Every 2 weeks
            "clinical_assessment": 4,  # Every 4 weeks
        }

        # Define adjustment thresholds
        adjustment_thresholds = {
            "progression_threshold": 0.2,  # 20% tumor growth
            "toxicity_threshold": 0.5,  # Grade 3+ toxicity
            "response_threshold": 0.3,  # 30% tumor reduction
        }

        # Expected outcomes under different scenarios
        expected_outcomes = {
            "favorable_response": {
                "progression_free_survival_months": 18.0,
                "overall_response_rate": 0.65,
            },
            "moderate_response": {
                "progression_free_survival_months": 12.0,
                "overall_response_rate": 0.40,
            },
            "poor_response": {
                "progression_free_survival_months": 6.0,
                "overall_response_rate": 0.15,
            },
        }

        rationale = (
            "This adaptive therapy plan emphasizes disease control over maximal "
            "tumor reduction, following principles of adaptive therapy. Regular "
            "monitoring enables early detection of resistance and toxicity, "
            "allowing timely treatment modifications."
        )

        return AdaptiveTherapyPlan(
            plan_id=plan_id,
            initial_sequence=initial_sequence,
            decision_rules=decision_rules,
            monitoring_schedule=monitoring_schedule,
            adjustment_thresholds=adjustment_thresholds,
            expected_outcomes=expected_outcomes,
            rationale=rationale,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize search engine state to dictionary."""
        return {
            "seed": self.seed,
            "max_depth": self.max_depth,
            "max_toxicity": self.max_toxicity,
            "resistance_penalty": self.resistance_penalty,
            "created_at": self.created_at,
            "drug_library": {k: v.to_dict() for k, v in self._drug_library.items()},
            "search_tree_size": len(self._search_tree),
            "search_history": self.search_history,
            "disclaimer": self.RESEARCH_DISCLAIMER,
        }


def create_example_drug_library() -> dict[str, DrugProfile]:
    """Create an example drug library for EGFR-driven cancers.

    This is a simplified library for demonstration purposes.
    Real libraries require extensive clinical and pharmacological data.

    Returns:
        Dictionary of DrugProfiles
    """
    drugs = {}

    # EGFR inhibitors
    drugs["erlotinib"] = DrugProfile(
        drug_id="erlotinib",
        name="Erlotinib",
        target_genes=["EGFR"],
        mechanism="EGFR tyrosine kinase inhibitor",
        intervention_type=InterventionType.TARGETED_THERAPY,
        efficacy_score=0.7,
        toxicity_score=0.3,
        resistance_mechanisms=[
            ResistanceMechanism.TARGET_MUTATION,
            ResistanceMechanism.BYPASS_PATHWAY,
        ],
        synergistic_drugs=["pembrolizumab"],
    )

    drugs["osimertinib"] = DrugProfile(
        drug_id="osimertinib",
        name="Osimertinib",
        target_genes=["EGFR"],
        mechanism="Third-generation EGFR TKI (T790M active)",
        intervention_type=InterventionType.TARGETED_THERAPY,
        efficacy_score=0.8,
        toxicity_score=0.25,
        resistance_mechanisms=[
            ResistanceMechanism.TARGET_MUTATION,
            ResistanceMechanism.BYPASS_PATHWAY,
        ],
    )

    # MEK inhibitor for combination
    drugs["trametinib"] = DrugProfile(
        drug_id="trametinib",
        name="Trametinib",
        target_genes=["MEK1", "MEK2"],
        mechanism="MEK inhibitor",
        intervention_type=InterventionType.TARGETED_THERAPY,
        efficacy_score=0.6,
        toxicity_score=0.35,
        resistance_mechanisms=[ResistanceMechanism.BYPASS_PATHWAY],
        synergistic_drugs=["erlotinib", "osimertinib"],
    )

    # Immunotherapy
    drugs["pembrolizumab"] = DrugProfile(
        drug_id="pembrolizumab",
        name="Pembrolizumab",
        target_genes=["PD1"],
        mechanism="PD-1 checkpoint inhibitor",
        intervention_type=InterventionType.IMMUNOTHERAPY,
        efficacy_score=0.5,
        toxicity_score=0.2,
        resistance_mechanisms=[ResistanceMechanism.IMMUNE_EVASION],
        synergistic_drugs=["erlotinib"],
    )

    # Chemotherapy
    drugs["carboplatin_pemetrexed"] = DrugProfile(
        drug_id="carboplatin_pemetrexed",
        name="Carboplatin + Pemetrexed",
        target_genes=["DNA_synthesis"],
        mechanism="Platinum-based chemotherapy with antifolate",
        intervention_type=InterventionType.CHEMOTHERAPY,
        efficacy_score=0.55,
        toxicity_score=0.6,
        resistance_mechanisms=[
            ResistanceMechanism.EFFLUX_PUMP,
            ResistanceMechanism.METABOLIC_REWIRING,
        ],
    )

    return drugs
