"""Discovery Contracts and Cross-Vertical Intents.

Implements QIL (QRATUM Intent Language) contracts for cross-vertical
synthesis and discovery acceleration.

Key Features:
- Contract-bound execution with confidence scoring
- Cross-vertical intent routing via TXO
- Q-FORGE hypothesis generation integration
- Full provenance tracking
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from qradle.merkle import MerkleChain
from qratum_asi.discovery_acceleration.workflows import (
    DiscoveryType,
)


class ContractStatus(Enum):
    """Status of a discovery contract."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    AUTHORIZED = "authorized"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class IntentType(Enum):
    """Types of cross-vertical intents."""

    SYNTHESIS = "synthesis"  # Combine outputs from multiple verticals
    QUERY = "query"  # Query single vertical
    TRANSFORM = "transform"  # Transform data between verticals
    VALIDATE = "validate"  # Validate across verticals
    FEDERATE = "federate"  # Federated computation across sites


@dataclass
class VerticalBinding:
    """Binding to a vertical module.

    Attributes:
        vertical_name: Name of vertical (e.g., VITRA, ECORA, CAPRA)
        operation: Operation to perform
        parameters: Operation parameters
        required_zone: Minimum security zone required
    """

    vertical_name: str
    operation: str
    parameters: dict[str, Any]
    required_zone: str = "Z1"

    def compute_hash(self) -> str:
        """Compute hash of binding."""
        content = {
            "vertical_name": self.vertical_name,
            "operation": self.operation,
            "parameters": self.parameters,
            "required_zone": self.required_zone,
        }
        return hashlib.sha3_256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize binding."""
        return {
            "vertical_name": self.vertical_name,
            "operation": self.operation,
            "parameters": self.parameters,
            "required_zone": self.required_zone,
            "hash": self.compute_hash(),
        }


@dataclass
class CrossVerticalIntent:
    """Intent for cross-vertical operation.

    Attributes:
        intent_id: Unique intent identifier
        intent_type: Type of intent
        discovery_type: Associated discovery type
        source_verticals: Source vertical bindings
        target_vertical: Target vertical for output
        synthesis_goal: Goal of the synthesis
        confidence_threshold: Minimum required confidence
        required_authorizers: Required dual-control authorizers
    """

    intent_id: str
    intent_type: IntentType
    discovery_type: DiscoveryType
    source_verticals: list[VerticalBinding]
    target_vertical: str
    synthesis_goal: str
    confidence_threshold: float = 0.75
    required_authorizers: list[str] = field(default_factory=list)
    status: ContractStatus = ContractStatus.DRAFT
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def compute_intent_hash(self) -> str:
        """Compute hash of intent."""
        content = {
            "intent_id": self.intent_id,
            "intent_type": self.intent_type.value,
            "discovery_type": self.discovery_type.value,
            "source_verticals": [v.to_dict() for v in self.source_verticals],
            "target_vertical": self.target_vertical,
            "synthesis_goal": self.synthesis_goal,
            "confidence_threshold": self.confidence_threshold,
        }
        return hashlib.sha3_256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize intent."""
        return {
            "intent_id": self.intent_id,
            "intent_type": self.intent_type.value,
            "discovery_type": self.discovery_type.value,
            "source_verticals": [v.to_dict() for v in self.source_verticals],
            "target_vertical": self.target_vertical,
            "synthesis_goal": self.synthesis_goal,
            "confidence_threshold": self.confidence_threshold,
            "required_authorizers": self.required_authorizers,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "intent_hash": self.compute_intent_hash(),
        }


@dataclass
class DiscoveryContract:
    """Contract for discovery workflow execution.

    Implements contract-bound execution with:
    - Confidence scoring
    - Q-FORGE hypothesis integration
    - Full provenance tracking
    - Rollback capability

    Attributes:
        contract_id: Unique contract identifier
        discovery_type: Type of discovery
        intents: List of cross-vertical intents
        hypotheses: Generated hypotheses with confidence
        authorizations: Authorization records
        execution_log: Execution audit log
        merkle_chain: Provenance chain
    """

    contract_id: str
    discovery_type: DiscoveryType
    intents: list[CrossVerticalIntent] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    authorizations: list[dict[str, Any]] = field(default_factory=list)
    execution_log: list[dict[str, Any]] = field(default_factory=list)
    merkle_chain: MerkleChain = field(default_factory=MerkleChain)
    status: ContractStatus = ContractStatus.DRAFT
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None

    def __post_init__(self):
        """Initialize contract."""
        self.merkle_chain.add_event(
            "contract_created",
            {
                "contract_id": self.contract_id,
                "discovery_type": self.discovery_type.value,
                "timestamp": self.created_at,
            },
        )

    def add_intent(self, intent: CrossVerticalIntent) -> None:
        """Add a cross-vertical intent to the contract.

        Args:
            intent: Intent to add
        """
        self.intents.append(intent)
        self.merkle_chain.add_event(
            "intent_added",
            {
                "intent_id": intent.intent_id,
                "intent_type": intent.intent_type.value,
                "intent_hash": intent.compute_intent_hash(),
            },
        )

    def add_hypothesis(
        self,
        hypothesis_id: str,
        description: str,
        confidence: float,
        supporting_evidence: list[str],
        domains: list[str],
    ) -> dict[str, Any]:
        """Add a Q-FORGE generated hypothesis.

        Args:
            hypothesis_id: Unique hypothesis identifier
            description: Hypothesis description
            confidence: Confidence score (0-1)
            supporting_evidence: List of supporting evidence
            domains: Related domains

        Returns:
            Created hypothesis record
        """
        hypothesis = {
            "hypothesis_id": hypothesis_id,
            "description": description,
            "confidence": confidence,
            "supporting_evidence": supporting_evidence,
            "domains": domains,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.hypotheses.append(hypothesis)
        self.merkle_chain.add_event(
            "hypothesis_generated",
            {
                "hypothesis_id": hypothesis_id,
                "confidence": confidence,
                "domains": domains,
            },
        )

        return hypothesis

    def authorize(
        self,
        authorizer_id: str,
        authorization_type: str,
        scope: list[str],
    ) -> dict[str, Any]:
        """Add an authorization to the contract.

        Args:
            authorizer_id: ID of authorizer
            authorization_type: Type of authorization
            scope: Scope of authorization

        Returns:
            Authorization record
        """
        authorization = {
            "authorizer_id": authorizer_id,
            "authorization_type": authorization_type,
            "scope": scope,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.authorizations.append(authorization)
        self.merkle_chain.add_event(
            "authorization_added",
            {
                "authorizer_id": authorizer_id,
                "authorization_type": authorization_type,
            },
        )

        # Check if all required authorizations are met
        self._check_authorization_status()

        return authorization

    def _check_authorization_status(self) -> None:
        """Check and update authorization status."""
        authorizer_ids = {a["authorizer_id"] for a in self.authorizations}

        # Check all intents
        all_authorized = True
        for intent in self.intents:
            required = set(intent.required_authorizers)
            if not required.issubset(authorizer_ids):
                all_authorized = False
                break

        if all_authorized and self.status == ContractStatus.SUBMITTED:
            self.status = ContractStatus.AUTHORIZED
            self.merkle_chain.add_event(
                "contract_authorized",
                {
                    "contract_id": self.contract_id,
                },
            )

    def submit(self) -> bool:
        """Submit contract for authorization.

        Returns:
            True if submission succeeded
        """
        if self.status != ContractStatus.DRAFT:
            return False

        if not self.intents:
            return False

        self.status = ContractStatus.SUBMITTED
        self.merkle_chain.add_event(
            "contract_submitted",
            {
                "contract_id": self.contract_id,
                "num_intents": len(self.intents),
            },
        )

        return True

    def execute(self) -> dict[str, Any]:
        """Execute the contract.

        Returns:
            Execution results
        """
        if self.status != ContractStatus.AUTHORIZED:
            raise ValueError(f"Contract not authorized: {self.contract_id}")

        self.status = ContractStatus.EXECUTING
        self.merkle_chain.add_event(
            "contract_execution_started",
            {
                "contract_id": self.contract_id,
            },
        )

        results = {
            "contract_id": self.contract_id,
            "intents_executed": [],
            "hypotheses": self.hypotheses,
            "success": True,
        }

        # Execute each intent
        for intent in self.intents:
            try:
                intent_result = self._execute_intent(intent)
                results["intents_executed"].append(intent_result)
                intent.status = ContractStatus.COMPLETED
            except Exception as e:
                intent.status = ContractStatus.FAILED
                results["success"] = False
                self.execution_log.append(
                    {
                        "intent_id": intent.intent_id,
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

        # Update status
        if results["success"]:
            self.status = ContractStatus.COMPLETED
            self.completed_at = datetime.now(timezone.utc).isoformat()
            self.merkle_chain.add_event(
                "contract_completed",
                {
                    "contract_id": self.contract_id,
                    "intents_completed": len(results["intents_executed"]),
                },
            )
        else:
            self.status = ContractStatus.FAILED
            self.merkle_chain.add_event(
                "contract_failed",
                {
                    "contract_id": self.contract_id,
                },
            )

        return results

    def _execute_intent(self, intent: CrossVerticalIntent) -> dict[str, Any]:
        """Execute a single intent.

        Args:
            intent: Intent to execute

        Returns:
            Intent execution result
        """
        # Log execution start
        self.execution_log.append(
            {
                "intent_id": intent.intent_id,
                "event": "execution_started",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Simulate intent execution (placeholder for actual vertical calls)
        result = {
            "intent_id": intent.intent_id,
            "intent_type": intent.intent_type.value,
            "source_verticals": [v.vertical_name for v in intent.source_verticals],
            "target_vertical": intent.target_vertical,
            "synthesis_result": {
                "confidence": 0.85,
                "insights": [
                    f"Cross-domain insight from {intent.synthesis_goal}",
                ],
            },
            "execution_time_ms": 150,
        }

        # Log execution completion
        self.execution_log.append(
            {
                "intent_id": intent.intent_id,
                "event": "execution_completed",
                "confidence": result["synthesis_result"]["confidence"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return result

    def compute_contract_hash(self) -> str:
        """Compute hash of entire contract.

        Returns:
            Contract hash
        """
        content = {
            "contract_id": self.contract_id,
            "discovery_type": self.discovery_type.value,
            "intents": [i.to_dict() for i in self.intents],
            "hypotheses": self.hypotheses,
            "status": self.status.value,
        }
        return hashlib.sha3_256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def verify_provenance(self) -> bool:
        """Verify contract provenance chain.

        Returns:
            True if chain is valid
        """
        return self.merkle_chain.verify_integrity()

    def to_dict(self) -> dict[str, Any]:
        """Serialize contract.

        Returns:
            Contract dictionary
        """
        return {
            "contract_id": self.contract_id,
            "discovery_type": self.discovery_type.value,
            "intents": [i.to_dict() for i in self.intents],
            "hypotheses": self.hypotheses,
            "authorizations": self.authorizations,
            "execution_log": self.execution_log,
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "contract_hash": self.compute_contract_hash(),
            "provenance_valid": self.verify_provenance(),
        }


def create_discovery_contract(
    discovery_type: DiscoveryType,
    synthesis_goal: str,
    verticals: list[tuple[str, str, dict[str, Any]]],
    target_vertical: str,
    required_authorizers: list[str],
) -> DiscoveryContract:
    """Factory function to create a discovery contract.

    Args:
        discovery_type: Type of discovery
        synthesis_goal: Goal of the synthesis
        verticals: List of (vertical_name, operation, parameters) tuples
        target_vertical: Target vertical for output
        required_authorizers: Required authorizers for dual-control

    Returns:
        Configured DiscoveryContract
    """
    contract_id = (
        f"dc_{discovery_type.value}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )

    contract = DiscoveryContract(
        contract_id=contract_id,
        discovery_type=discovery_type,
    )

    # Create vertical bindings
    source_verticals = [
        VerticalBinding(
            vertical_name=name,
            operation=operation,
            parameters=params,
        )
        for name, operation, params in verticals
    ]

    # Create intent
    intent = CrossVerticalIntent(
        intent_id=f"intent_{contract_id}_001",
        intent_type=IntentType.SYNTHESIS,
        discovery_type=discovery_type,
        source_verticals=source_verticals,
        target_vertical=target_vertical,
        synthesis_goal=synthesis_goal,
        required_authorizers=required_authorizers,
    )

    contract.add_intent(intent)

    return contract


# Pre-defined contract templates for each discovery type
DISCOVERY_CONTRACT_TEMPLATES: dict[DiscoveryType, dict[str, Any]] = {
    DiscoveryType.COMPLEX_DISEASE_GENETICS: {
        "verticals": [
            ("VITRA", "germline_analysis", {"reference": "GRCh38"}),
            ("VITRA", "gwas_association", {"significance": 5e-8}),
        ],
        "target_vertical": "VITRA",
        "synthesis_goal": "Identify hidden genetic causes of complex diseases via federated ZK-GWAS",
    },
    DiscoveryType.PERSONALIZED_DRUG_DESIGN: {
        "verticals": [
            ("VITRA", "sequence_analysis", {"analysis_type": "pharmacogenomics"}),
            ("VITRA", "drug_screening", {"target_type": "personalized"}),
        ],
        "target_vertical": "VITRA",
        "synthesis_goal": "Design personalized drugs based on individual DNA profiles",
    },
    DiscoveryType.CLIMATE_GENE_CONNECTIONS: {
        "verticals": [
            ("ECORA", "climate_projection", {"scenario": "SSP2-4.5"}),
            ("VITRA", "epigenetics_analysis", {"type": "methylation"}),
        ],
        "target_vertical": "VITRA",
        "synthesis_goal": "Identify epigenetic impacts of climate/pollution on human genetics",
    },
    DiscoveryType.NATURAL_DRUG_DISCOVERY: {
        "verticals": [
            ("VITRA", "sequence_analysis", {"source": "biodataset"}),
            ("VITRA", "drug_screening", {"compounds": "natural"}),
        ],
        "target_vertical": "VITRA",
        "synthesis_goal": "Discover safer antibiotics/cancer drugs from natural sources",
    },
    DiscoveryType.ECONOMIC_BIOLOGICAL_MODEL: {
        "verticals": [
            ("CAPRA", "monte_carlo", {"scenario": "health_crisis"}),
            ("VITRA", "population_genetics", {"trait": "behavioral"}),
        ],
        "target_vertical": "CAPRA",
        "synthesis_goal": "Predict health crises impacting markets or genetic factors in behaviors",
    },
    DiscoveryType.ANTI_AGING_PATHWAYS: {
        "verticals": [
            ("NEURA", "neural_simulation", {"model": "aging"}),
            ("VITRA", "longevity_analysis", {"pathways": True}),
            ("ECORA", "environmental_exposure", {"factors": ["pollution", "diet"]}),
        ],
        "target_vertical": "VITRA",
        "synthesis_goal": "Identify anti-aging pathways with safe rollback-enabled exploration",
    },
}
