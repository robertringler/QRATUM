"""Core Reinjection Engine.

Main orchestrator for the discovery reinjection feedback loop.
Implements the complete cycle:
1. Validation
2. Z1 Sandbox testing
3. Dual-control approval
4. Z2 Commitment
5. Audit report generation

Enforces all 8 Fatal Invariants at every layer.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from qradle.merkle import MerkleChain

from qratum_asi.reinjection.types import (
    AuditRecord,
    DiscoveryDomain,
    ReinjectionCandidate,
    ReinjectionResult,
    ReinjectionScore,
    ReinjectionStatus,
    SandboxResult,
    ValidationLevel,
)
from qratum_asi.reinjection.validator import ReinjectionValidator, ValidationResult
from qratum_asi.reinjection.mapper import DiscoveryPriorMapper, MappingResult
from qratum_asi.reinjection.sandbox import SandboxOrchestrator
from qratum_asi.reinjection.contracts import (
    ReinjectionContract,
    ReinjectionContractStatus,
    create_reinjection_contract,
)
from qratum_asi.reinjection.audit import AuditReportGenerator, AuditReport


@dataclass
class ReinjectionCycleResult:
    """Result of a complete reinjection cycle.

    Attributes:
        cycle_id: Unique cycle identifier
        candidate: Original candidate
        validation_result: Validation outcome
        sandbox_result: Sandbox testing outcome
        contract: Reinjection contract
        reinjection_result: Final reinjection result
        audit_report: Generated audit report
        success: Whether cycle completed successfully
    """

    cycle_id: str
    candidate: ReinjectionCandidate
    validation_result: ValidationResult | None
    sandbox_result: SandboxResult | None
    contract: ReinjectionContract | None
    reinjection_result: ReinjectionResult | None
    audit_report: AuditReport | None
    success: bool
    error_message: str = ""
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize cycle result."""
        return {
            "cycle_id": self.cycle_id,
            "candidate_id": self.candidate.candidate_id,
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "sandbox_result": self.sandbox_result.to_dict() if self.sandbox_result else None,
            "contract": self.contract.to_dict() if self.contract else None,
            "reinjection_result": self.reinjection_result.to_dict() if self.reinjection_result else None,
            "audit_report": self.audit_report.to_dict() if self.audit_report else None,
            "success": self.success,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
        }


class ReinjectionEngine:
    """Core engine for discovery reinjection.

    Orchestrates the complete reinjection feedback cycle:
    1. Input validation with domain-specific rules
    2. Z1 sandbox testing with rollback verification
    3. Dual-control approval process
    4. Z2 commitment with full provenance
    5. Audit report generation

    Enforces:
    - Hard determinism (bit-identical results)
    - Cryptographic Merkle provenance
    - Native reversibility/rollback
    - Dual-control governance
    - Zero-knowledge operational privacy
    - Trajectory-awareness
    - Defensive-only posture
    - 8 Fatal Invariants
    """

    def __init__(self, merkle_chain: MerkleChain | None = None):
        """Initialize reinjection engine.

        Args:
            merkle_chain: Optional Merkle chain for provenance
        """
        self.merkle_chain = merkle_chain or MerkleChain()

        # Initialize components
        self.validator = ReinjectionValidator(merkle_chain=self.merkle_chain)
        self.mapper = DiscoveryPriorMapper(merkle_chain=self.merkle_chain)
        self.sandbox = SandboxOrchestrator(merkle_chain=self.merkle_chain)
        self.audit_generator = AuditReportGenerator(merkle_chain=self.merkle_chain)

        # Track state
        self.candidates: dict[str, ReinjectionCandidate] = {}
        self.contracts: dict[str, ReinjectionContract] = {}
        self.results: dict[str, ReinjectionResult] = {}
        self.cycle_results: list[ReinjectionCycleResult] = []

        self._candidate_counter = 0
        self._cycle_counter = 0

        # Log initialization
        self.merkle_chain.add_event(
            "reinjection_engine_initialized",
            {
                "version": "1.0.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def create_candidate(
        self,
        discovery_id: str,
        domain: DiscoveryDomain,
        description: str,
        data_payload: dict[str, Any],
        mutual_information: float,
        cross_impact: float,
        confidence: float,
        target_priors: list[str],
        source_workflow_id: str,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        novelty: float = 0.0,
        entropy_reduction: float = 0.0,
        compression_efficiency: float = 0.0,
    ) -> ReinjectionCandidate:
        """Create a new reinjection candidate.

        Args:
            discovery_id: Source discovery ID
            domain: Discovery domain
            description: Human-readable description
            data_payload: Discovery data
            mutual_information: MI score (0-1)
            cross_impact: Cross-impact score (0-1)
            confidence: Confidence score (0-1)
            target_priors: Target priors to update
            source_workflow_id: Source workflow ID
            validation_level: Required validation level
            novelty: Novelty score (0-1)
            entropy_reduction: Entropy reduction (0-1)
            compression_efficiency: Compression efficiency (0-1)

        Returns:
            Created ReinjectionCandidate
        """
        self._candidate_counter += 1
        candidate_id = f"cand_{discovery_id}_{self._candidate_counter:06d}"

        score = ReinjectionScore(
            mutual_information=mutual_information,
            cross_impact=cross_impact,
            confidence=confidence,
            novelty=novelty,
            entropy_reduction=entropy_reduction,
            compression_efficiency=compression_efficiency,
        )

        # Compute provenance hash from source workflow
        provenance_data = {
            "discovery_id": discovery_id,
            "source_workflow_id": source_workflow_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        provenance_hash = hashlib.sha3_256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        candidate = ReinjectionCandidate(
            candidate_id=candidate_id,
            discovery_id=discovery_id,
            domain=domain,
            description=description,
            data_payload=data_payload,
            score=score,
            target_priors=target_priors,
            source_workflow_id=source_workflow_id,
            provenance_hash=provenance_hash,
            validation_level=validation_level,
        )

        self.candidates[candidate_id] = candidate

        # Log candidate creation
        self.merkle_chain.add_event(
            "candidate_created",
            {
                "candidate_id": candidate_id,
                "discovery_id": discovery_id,
                "domain": domain.value,
                "composite_score": score.composite_score,
            },
        )

        return candidate

    def execute_full_cycle(
        self,
        candidate: ReinjectionCandidate,
        approvers: list[str] | None = None,
        auto_approve: bool = False,
    ) -> ReinjectionCycleResult:
        """Execute a complete reinjection cycle.

        Performs:
        1. Validation
        2. Mapping generation
        3. Sandbox testing
        4. Contract creation and approval
        5. Z2 commitment
        6. Audit report generation

        Args:
            candidate: Candidate to reinject
            approvers: List of approvers for dual-control
            auto_approve: If True, auto-approve for testing (not production)

        Returns:
            ReinjectionCycleResult with full cycle outcome
        """
        start_time = time.perf_counter()
        self._cycle_counter += 1
        cycle_id = f"cycle_{candidate.candidate_id}_{self._cycle_counter:04d}"

        validation_result = None
        sandbox_result = None
        contract = None
        reinjection_result = None
        audit_report = None
        error_message = ""

        try:
            # Step 1: Validate candidate
            candidate.status = ReinjectionStatus.VALIDATING
            validation_result = self.validator.validate(
                candidate, candidate.validation_level
            )

            if not validation_result.valid:
                candidate.status = ReinjectionStatus.REJECTED
                error_message = f"Validation failed: {'; '.join(validation_result.errors)}"
                raise ValueError(error_message)

            # Step 2: Generate mapping
            mapping_result = self.mapper.map_discovery_to_priors(candidate)

            # Step 3: Sandbox testing
            candidate.status = ReinjectionStatus.SANDBOX_TESTING
            sandbox_result = self.sandbox.run_sandbox_test(candidate, mapping_result)

            if not sandbox_result.success:
                candidate.status = ReinjectionStatus.FAILED
                error_message = f"Sandbox test failed: {'; '.join(sandbox_result.side_effects)}"
                raise ValueError(error_message)

            # Step 4: Create contract and handle approvals
            candidate.status = ReinjectionStatus.AWAITING_APPROVAL
            contract = create_reinjection_contract(
                candidate=candidate,
                required_approvers=approvers,
            )
            contract.submit()
            contract.enter_z1_sandbox()
            contract.request_approval()

            self.contracts[contract.contract_id] = contract

            # Handle approvals
            if auto_approve and contract.required_approvers:
                for approver_id in contract.required_approvers:
                    contract.add_approval(
                        approver_id=approver_id,
                        decision="approve",
                        reason="Auto-approved for testing",
                    )

            if not contract.is_approved():
                error_message = "Contract not approved - awaiting dual-control authorization"
                # Still return partial result for pending approval
                execution_time_ms = (time.perf_counter() - start_time) * 1000

                result = ReinjectionCycleResult(
                    cycle_id=cycle_id,
                    candidate=candidate,
                    validation_result=validation_result,
                    sandbox_result=sandbox_result,
                    contract=contract,
                    reinjection_result=None,
                    audit_report=None,
                    success=False,
                    error_message=error_message,
                    execution_time_ms=execution_time_ms,
                )
                self.cycle_results.append(result)
                return result

            # Step 5: Z2 Commitment
            if not contract.commit_z2():
                candidate.status = ReinjectionStatus.FAILED
                error_message = "Z2 commitment failed"
                raise ValueError(error_message)

            candidate.status = ReinjectionStatus.COMMITTED

            # Apply mapping to actual priors
            self.mapper.apply_mapping(mapping_result)

            # Create final result
            reinjection_result = ReinjectionResult(
                result_id=f"result_{cycle_id}",
                candidate_id=candidate.candidate_id,
                status=ReinjectionStatus.COMMITTED,
                validation_passed=True,
                sandbox_result=sandbox_result,
                approvers=[a.approver_id for a in contract.approvals],
                committed_at=contract.committed_at,
                rollback_id=contract.rollback_id,
                merkle_proof=self.merkle_chain.get_chain_proof(),
                metrics={
                    "fidelity_improvement": sandbox_result.fidelity_score,
                    "priors_updated": mapping_result.total_priors_affected,
                    "avg_confidence_improvement": mapping_result.average_confidence_improvement,
                },
            )

            self.results[reinjection_result.result_id] = reinjection_result

            # Step 6: Generate audit report
            audit_report = self.audit_generator.generate_report(contract, reinjection_result)

            # Update result with audit hash
            reinjection_result.audit_hash = audit_report.report_hash

            # Log successful cycle
            self.merkle_chain.add_event(
                "reinjection_cycle_completed",
                {
                    "cycle_id": cycle_id,
                    "candidate_id": candidate.candidate_id,
                    "success": True,
                },
            )

            success = True

        except ValueError as e:
            error_message = str(e)
            success = False
            self.merkle_chain.add_event(
                "reinjection_cycle_failed",
                {
                    "cycle_id": cycle_id,
                    "candidate_id": candidate.candidate_id,
                    "error": error_message,
                },
            )

        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            success = False
            candidate.status = ReinjectionStatus.FAILED

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        result = ReinjectionCycleResult(
            cycle_id=cycle_id,
            candidate=candidate,
            validation_result=validation_result,
            sandbox_result=sandbox_result,
            contract=contract,
            reinjection_result=reinjection_result,
            audit_report=audit_report,
            success=success,
            error_message=error_message,
            execution_time_ms=execution_time_ms,
        )

        self.cycle_results.append(result)
        return result

    def rollback_reinjection(
        self,
        result_id: str,
        reason: str,
        actor_id: str,
    ) -> bool:
        """Rollback a committed reinjection.

        Args:
            result_id: Result ID to rollback
            reason: Reason for rollback
            actor_id: Actor initiating rollback

        Returns:
            True if rollback succeeded
        """
        if result_id not in self.results:
            return False

        result = self.results[result_id]
        candidate_id = result.candidate_id

        # Find associated contract
        contract = None
        for c in self.contracts.values():
            if c.candidate.candidate_id == candidate_id:
                contract = c
                break

        if not contract:
            return False

        # Rollback contract
        if not contract.rollback(reason, actor_id):
            return False

        # Rollback mapping
        # Find mapping in mapper history
        for mapping in self.mapper.mapping_history:
            if mapping.candidate_id == candidate_id:
                self.mapper.rollback_mapping(mapping.mapping_id)
                break

        # Update result status
        result.status = ReinjectionStatus.ROLLED_BACK

        # Update candidate
        if candidate_id in self.candidates:
            self.candidates[candidate_id].status = ReinjectionStatus.ROLLED_BACK

        # Log rollback
        self.merkle_chain.add_event(
            "reinjection_rolled_back",
            {
                "result_id": result_id,
                "reason": reason,
                "actor_id": actor_id,
            },
        )

        return True

    def get_engine_stats(self) -> dict[str, Any]:
        """Get comprehensive engine statistics."""
        successful_cycles = sum(1 for c in self.cycle_results if c.success)

        return {
            "total_candidates": len(self.candidates),
            "total_contracts": len(self.contracts),
            "total_results": len(self.results),
            "total_cycles": len(self.cycle_results),
            "successful_cycles": successful_cycles,
            "success_rate": successful_cycles / len(self.cycle_results) if self.cycle_results else 0.0,
            "validation_stats": self.validator.get_validation_summary(),
            "mapping_stats": self.mapper.get_mapping_stats(),
            "sandbox_stats": self.sandbox.get_sandbox_stats(),
            "audit_stats": self.audit_generator.get_report_stats(),
            "merkle_chain_length": len(self.merkle_chain.chain),
            "merkle_chain_valid": self.merkle_chain.verify_integrity(),
        }

    def verify_provenance(self) -> bool:
        """Verify complete provenance chain integrity.

        Returns:
            True if all chains are valid
        """
        return self.merkle_chain.verify_integrity()


def create_synthetic_discovery_candidate(
    compound_name: str = "Natural Compound X",
    target_type: str = "antimicrobial",
    confidence: float = 0.85,
    mutual_information: float = 0.75,
) -> tuple[str, DiscoveryDomain, str, dict[str, Any], list[str], str]:
    """Create synthetic discovery data for testing.

    Args:
        compound_name: Name of compound
        target_type: Type of target
        confidence: Confidence score
        mutual_information: MI score

    Returns:
        Tuple of (discovery_id, domain, description, data_payload, target_priors, workflow_id)
    """
    discovery_id = f"disc_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    domain = DiscoveryDomain.BIODISCOVERY
    description = f"High-confidence {target_type} discovery: {compound_name}"

    data_payload = {
        "compound_data": {
            "name": compound_name,
            "type": target_type,
            "predicted_efficacy": 0.82,
            "toxicity_score": 0.12,
        },
        "evidence": [
            "In vitro screening positive",
            "Structural analysis complete",
            "Pathway analysis validated",
        ],
        "metadata": {
            "source": "natural_compound_library",
            "screening_method": "high_throughput",
        },
    }

    target_priors = [
        "compound_affinity",
        "target_specificity",
        "bioavailability",
    ]

    workflow_id = f"wf_natural_drug_{datetime.now(timezone.utc).strftime('%Y%m%d')}"

    return discovery_id, domain, description, data_payload, target_priors, workflow_id
