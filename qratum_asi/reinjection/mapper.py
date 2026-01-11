"""Discovery-to-Prior Mapper.

Maps validated discoveries to biodiscovery library priors for reinjection.
Handles cross-vertical mapping with provenance tracking.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from qradle.merkle import MerkleChain

from qratum_asi.reinjection.types import (
    DiscoveryDomain,
    ReinjectionCandidate,
)


@dataclass
class PriorUpdate:
    """Represents an update to a biodiscovery prior.

    Attributes:
        prior_id: Unique prior identifier
        prior_type: Type of prior (e.g., compound_affinity, pathway_weight)
        current_value: Current prior value
        proposed_value: Proposed updated value
        confidence_delta: Change in confidence
        source_discovery_id: Discovery that triggered this update
        supporting_evidence: Evidence supporting the update
    """

    prior_id: str
    prior_type: str
    current_value: float
    proposed_value: float
    confidence_delta: float
    source_discovery_id: str
    supporting_evidence: list[str]
    domain: DiscoveryDomain = DiscoveryDomain.BIODISCOVERY
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def value_delta(self) -> float:
        """Compute change in value."""
        return self.proposed_value - self.current_value

    @property
    def relative_change(self) -> float:
        """Compute relative change as percentage."""
        if self.current_value == 0:
            return float("inf") if self.proposed_value != 0 else 0.0
        return abs(self.value_delta) / abs(self.current_value)

    def compute_hash(self) -> str:
        """Compute hash of prior update."""
        content = {
            "prior_id": self.prior_id,
            "prior_type": self.prior_type,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "source_discovery_id": self.source_discovery_id,
        }
        return hashlib.sha3_256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize prior update."""
        return {
            "prior_id": self.prior_id,
            "prior_type": self.prior_type,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "value_delta": self.value_delta,
            "relative_change": self.relative_change,
            "confidence_delta": self.confidence_delta,
            "source_discovery_id": self.source_discovery_id,
            "supporting_evidence": self.supporting_evidence,
            "domain": self.domain.value,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "update_hash": self.compute_hash(),
        }


@dataclass
class MappingResult:
    """Result of a discovery-to-prior mapping.

    Attributes:
        mapping_id: Unique mapping identifier
        candidate_id: Source candidate ID
        prior_updates: List of proposed prior updates
        total_priors_affected: Number of priors affected
        average_confidence_improvement: Average confidence improvement
        merkle_proof: Merkle proof of mapping
    """

    mapping_id: str
    candidate_id: str
    prior_updates: list[PriorUpdate]
    total_priors_affected: int
    average_confidence_improvement: float
    merkle_proof: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cross_vertical_impacts: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize mapping result."""
        return {
            "mapping_id": self.mapping_id,
            "candidate_id": self.candidate_id,
            "prior_updates": [u.to_dict() for u in self.prior_updates],
            "total_priors_affected": self.total_priors_affected,
            "average_confidence_improvement": self.average_confidence_improvement,
            "merkle_proof": self.merkle_proof,
            "timestamp": self.timestamp,
            "cross_vertical_impacts": self.cross_vertical_impacts,
        }


# Prior type mapping by domain
DOMAIN_PRIOR_TYPES: dict[DiscoveryDomain, list[str]] = {
    DiscoveryDomain.BIODISCOVERY: [
        "compound_affinity",
        "target_specificity",
        "bioavailability",
        "toxicity_risk",
    ],
    DiscoveryDomain.GENOMICS: [
        "variant_pathogenicity",
        "gene_expression_weight",
        "regulatory_impact",
        "population_frequency",
    ],
    DiscoveryDomain.DRUG_DISCOVERY: [
        "drug_efficacy",
        "drug_target_binding",
        "adme_score",
        "clinical_success_probability",
    ],
    DiscoveryDomain.CLIMATE_BIOLOGY: [
        "environmental_sensitivity",
        "adaptation_potential",
        "epigenetic_plasticity",
    ],
    DiscoveryDomain.LONGEVITY: [
        "aging_pathway_weight",
        "intervention_efficacy",
        "biomarker_correlation",
    ],
    DiscoveryDomain.NEURAL: [
        "neural_pathway_strength",
        "cognitive_impact",
        "neuroprotection_score",
    ],
    DiscoveryDomain.ECONOMIC_BIOLOGICAL: [
        "health_economic_correlation",
        "population_risk_factor",
        "intervention_cost_benefit",
    ],
    DiscoveryDomain.CROSS_VERTICAL: [
        "cross_domain_correlation",
        "synergy_coefficient",
        "integration_weight",
    ],
}


class DiscoveryPriorMapper:
    """Maps discoveries to biodiscovery library priors.

    Handles:
    - Domain-specific prior type resolution
    - Confidence-weighted update computation
    - Cross-vertical impact assessment
    - Provenance tracking
    """

    def __init__(self, merkle_chain: MerkleChain | None = None):
        """Initialize mapper.

        Args:
            merkle_chain: Optional Merkle chain for provenance
        """
        self.merkle_chain = merkle_chain or MerkleChain()
        self.mapping_history: list[MappingResult] = []
        self._mapping_counter = 0

        # Initialize mock prior database
        self._prior_database: dict[str, dict[str, Any]] = {}

    def map_discovery_to_priors(
        self,
        candidate: ReinjectionCandidate,
    ) -> MappingResult:
        """Map a validated discovery to biodiscovery priors.

        Args:
            candidate: Validated reinjection candidate

        Returns:
            MappingResult with proposed prior updates
        """
        self._mapping_counter += 1
        mapping_id = f"map_{candidate.candidate_id}_{self._mapping_counter:04d}"

        # Get applicable prior types for this domain
        prior_types = DOMAIN_PRIOR_TYPES.get(candidate.domain, ["generic_prior"])

        # Generate prior updates based on discovery data
        prior_updates = self._compute_prior_updates(
            candidate=candidate,
            prior_types=prior_types,
        )

        # Compute cross-vertical impacts
        cross_vertical_impacts = self._compute_cross_vertical_impacts(candidate)

        # Calculate average confidence improvement
        if prior_updates:
            avg_confidence = sum(u.confidence_delta for u in prior_updates) / len(prior_updates)
        else:
            avg_confidence = 0.0

        # Log to merkle chain
        self.merkle_chain.add_event(
            "mapping_completed",
            {
                "mapping_id": mapping_id,
                "candidate_id": candidate.candidate_id,
                "priors_affected": len(prior_updates),
                "avg_confidence_improvement": avg_confidence,
            },
        )

        result = MappingResult(
            mapping_id=mapping_id,
            candidate_id=candidate.candidate_id,
            prior_updates=prior_updates,
            total_priors_affected=len(prior_updates),
            average_confidence_improvement=avg_confidence,
            merkle_proof=self.merkle_chain.get_chain_proof(),
            cross_vertical_impacts=cross_vertical_impacts,
        )

        self.mapping_history.append(result)
        return result

    def _compute_prior_updates(
        self,
        candidate: ReinjectionCandidate,
        prior_types: list[str],
    ) -> list[PriorUpdate]:
        """Compute prior updates from discovery data."""
        updates: list[PriorUpdate] = []
        score = candidate.score
        payload = candidate.data_payload

        for prior_type in prior_types:
            # Check if this prior is in target_priors (used for prioritization)
            has_matching_target = any(
                prior_type in t.lower() for t in candidate.target_priors
            )

            # Get or initialize current prior value
            prior_id = f"{candidate.domain.value}_{prior_type}"
            current_value = self._get_current_prior_value(prior_id)

            # Compute proposed update based on discovery confidence and mutual information
            # Higher MI and confidence = larger update
            update_magnitude = score.mutual_information * score.confidence * 0.1
            proposed_value = current_value + update_magnitude

            # Confidence delta is proportional to discovery confidence
            confidence_delta = score.confidence * 0.05

            # Extract supporting evidence from payload
            evidence = payload.get("evidence", [])
            if not evidence:
                evidence = [f"Discovery {candidate.discovery_id}"]

            update = PriorUpdate(
                prior_id=prior_id,
                prior_type=prior_type,
                current_value=current_value,
                proposed_value=proposed_value,
                confidence_delta=confidence_delta,
                source_discovery_id=candidate.discovery_id,
                supporting_evidence=evidence if isinstance(evidence, list) else [evidence],
                domain=candidate.domain,
            )

            # Only add if there's meaningful change or matches target priors
            if update.relative_change > 0.001 or has_matching_target:
                updates.append(update)

        return updates

    def _compute_cross_vertical_impacts(
        self,
        candidate: ReinjectionCandidate,
    ) -> dict[str, float]:
        """Compute cross-vertical impact scores."""
        impacts: dict[str, float] = {}

        # Base impact is cross_impact score
        base_impact = candidate.score.cross_impact

        # Domain-specific cross-vertical relationships
        cross_vertical_map: dict[DiscoveryDomain, list[tuple[str, float]]] = {
            DiscoveryDomain.BIODISCOVERY: [
                ("VITRA", 0.9),
                ("ECORA", 0.3),
            ],
            DiscoveryDomain.GENOMICS: [
                ("VITRA", 1.0),
                ("NEURA", 0.4),
                ("CAPRA", 0.2),
            ],
            DiscoveryDomain.DRUG_DISCOVERY: [
                ("VITRA", 0.95),
                ("CAPRA", 0.3),
            ],
            DiscoveryDomain.CLIMATE_BIOLOGY: [
                ("ECORA", 0.9),
                ("VITRA", 0.6),
            ],
            DiscoveryDomain.LONGEVITY: [
                ("VITRA", 0.85),
                ("NEURA", 0.5),
            ],
            DiscoveryDomain.NEURAL: [
                ("NEURA", 0.95),
                ("VITRA", 0.4),
            ],
            DiscoveryDomain.ECONOMIC_BIOLOGICAL: [
                ("CAPRA", 0.8),
                ("VITRA", 0.5),
                ("STRATA", 0.3),
            ],
            DiscoveryDomain.CROSS_VERTICAL: [
                ("VITRA", 0.6),
                ("ECORA", 0.4),
                ("NEURA", 0.4),
                ("CAPRA", 0.4),
            ],
        }

        relationships = cross_vertical_map.get(candidate.domain, [])
        for vertical, weight in relationships:
            impacts[vertical] = base_impact * weight

        return impacts

    def _get_current_prior_value(self, prior_id: str) -> float:
        """Get current value of a prior from database."""
        if prior_id in self._prior_database:
            return self._prior_database[prior_id].get("value", 0.5)
        # Default prior value
        return 0.5

    def apply_mapping(self, mapping_result: MappingResult) -> bool:
        """Apply a mapping result to the prior database.

        Args:
            mapping_result: Mapping to apply

        Returns:
            True if applied successfully
        """
        for update in mapping_result.prior_updates:
            self._prior_database[update.prior_id] = {
                "value": update.proposed_value,
                "confidence": update.confidence_delta,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "source": update.source_discovery_id,
            }

        # Log application
        self.merkle_chain.add_event(
            "mapping_applied",
            {
                "mapping_id": mapping_result.mapping_id,
                "priors_updated": len(mapping_result.prior_updates),
            },
        )

        return True

    def rollback_mapping(self, mapping_id: str) -> bool:
        """Rollback a previously applied mapping.

        Args:
            mapping_id: ID of mapping to rollback

        Returns:
            True if rollback succeeded
        """
        # Find mapping in history
        mapping = None
        for m in self.mapping_history:
            if m.mapping_id == mapping_id:
                mapping = m
                break

        if not mapping:
            return False

        # Restore previous values
        for update in mapping.prior_updates:
            self._prior_database[update.prior_id] = {
                "value": update.current_value,
                "confidence": 0.0,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "source": "rollback",
            }

        # Log rollback
        self.merkle_chain.add_event(
            "mapping_rolled_back",
            {
                "mapping_id": mapping_id,
                "priors_restored": len(mapping.prior_updates),
            },
        )

        return True

    def get_mapping_stats(self) -> dict[str, Any]:
        """Get statistics about mappings."""
        if not self.mapping_history:
            return {
                "total_mappings": 0,
                "total_priors_updated": 0,
                "average_confidence_improvement": 0.0,
            }

        total_priors = sum(m.total_priors_affected for m in self.mapping_history)
        avg_confidence = sum(m.average_confidence_improvement for m in self.mapping_history) / len(
            self.mapping_history
        )

        return {
            "total_mappings": len(self.mapping_history),
            "total_priors_updated": total_priors,
            "average_confidence_improvement": avg_confidence,
            "priors_in_database": len(self._prior_database),
        }
