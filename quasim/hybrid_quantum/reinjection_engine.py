"""Reinjection Evaluation Engine for QRATUM.

This module implements the proposal-only reinjection evaluation engine
for managing the lifecycle of quantum execution outputs.

Core Principles:
- Proposal-only: Never auto-commit
- Pre-validation scoring before approval
- Merkle root tracking for provenance
- Dual-control approval integration
- Full rollback capability

The reinjection engine evaluates quantum outputs and creates standardized
proposal artifacts for human/dual-approved reinjection into QRATUM pipelines.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

# Trust invariant constant - never modify from external input
TRUST_INVARIANT_ASSERTION = "ℛ(t) ≥ 0"


class ProposalStatus(Enum):
    """Status of a reinjection proposal."""

    PENDING = "pending"
    PRE_VALIDATED = "pre_validated"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    ROLLED_BACK = "rolled_back"
    EXPIRED = "expired"


class ProposalCluster(Enum):
    """Priority enhancement clusters for proposals."""

    P0 = "P0"  # Quantum-Classical Hybrid Speed
    P1 = "P1"  # Epistemic Perfection / Scientific Substrate
    P2 = "P2"  # Distributed Fortress / Human Jurisdiction Interface
    P3 = "P3"  # Operational Anti-Entropy


@dataclass
class MerkleNode:
    """Node in merkle tree for provenance tracking.

    Attributes:
        hash: SHA-256 hash of this node
        left_child: Hash of left child (or None for leaf)
        right_child: Hash of right child (or None for leaf)
        data: Original data for leaf nodes
        timestamp: When node was created
    """

    hash: str
    left_child: str | None = None
    right_child: str | None = None
    data: str | None = None
    timestamp: str = ""

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class PreValidationScore:
    """Pre-validation scoring results.

    Attributes:
        score: Overall score (0-1)
        passed: Whether pre-validation passed
        checks: Individual check results
        warnings: Warning messages
        errors: Error messages
        timestamp: When scoring was performed
    """

    score: float = 0.0
    passed: bool = False
    checks: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    @property
    def checks_passed(self) -> int:
        """Count of passed checks."""
        return sum(1 for v in self.checks.values() if v)

    @property
    def checks_total(self) -> int:
        """Total number of checks."""
        return len(self.checks)


@dataclass
class ProposalArtifact:
    """Standardized proposal artifact for reinjection.

    This follows the QRATUM proposal artifact template format.

    Attributes:
        proposal_id: Unique identifier
        cluster: Priority enhancement cluster (P0-P3)
        metrics_target: Primary and secondary metric targets
        inputs: Merkle roots of input data
        expected_outputs: Expected output artifacts and metrics delta
        fallback_strategy: Description of fallback handling
        rollback_path: Merkle lineage pointer for rollback
        cryptographic_signatures: Dual-control signature placeholders
        invariant_assertion: Trust invariant assertion
        status: Current proposal status
        pre_validation: Pre-validation scoring results
        approvals: List of approval records
        rejections: List of rejection records
        created_at: Creation timestamp
        updated_at: Last update timestamp
        metadata: Additional metadata
    """

    proposal_id: str
    cluster: ProposalCluster = ProposalCluster.P1
    metrics_target: dict[str, str] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    expected_outputs: dict[str, Any] = field(default_factory=dict)
    fallback_strategy: str = "classical_tensor_fallback"
    rollback_path: str = ""
    cryptographic_signatures: list[str] = field(default_factory=list)
    invariant_assertion: str = TRUST_INVARIANT_ASSERTION
    status: ProposalStatus = ProposalStatus.PENDING
    pre_validation: PreValidationScore | None = None
    approvals: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set defaults and enforce invariant assertion."""
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        # Enforce invariant assertion cannot be tampered with
        self.invariant_assertion = TRUST_INVARIANT_ASSERTION

    def to_json(self) -> str:
        """Convert to standardized JSON format.

        Returns:
            JSON string representation
        """
        return json.dumps(
            {
                "proposal_id": self.proposal_id,
                "cluster": self.cluster.value,
                "metrics_target": self.metrics_target,
                "inputs": self.inputs,
                "expected_outputs": self.expected_outputs,
                "fallback_strategy": self.fallback_strategy,
                "rollback_path": self.rollback_path,
                "cryptographic_signatures": self.cryptographic_signatures,
                "invariant_assertion": self.invariant_assertion,
                "status": self.status.value,
                "pre_validation": (
                    {
                        "score": self.pre_validation.score,
                        "passed": self.pre_validation.passed,
                        "checks": self.pre_validation.checks,
                    }
                    if self.pre_validation
                    else None
                ),
                "approvals_count": len(self.approvals),
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ProposalArtifact":
        """Create artifact from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            ProposalArtifact instance

        Note:
            The invariant_assertion field is always set to the constant
            TRUST_INVARIANT_ASSERTION to prevent tampering via JSON input.
        """
        data = json.loads(json_str)
        # Note: invariant_assertion is not read from JSON to prevent tampering
        # The __post_init__ method enforces the constant value
        return cls(
            proposal_id=data["proposal_id"],
            cluster=ProposalCluster(data["cluster"]),
            metrics_target=data.get("metrics_target", {}),
            inputs=data.get("inputs", []),
            expected_outputs=data.get("expected_outputs", {}),
            fallback_strategy=data.get("fallback_strategy", ""),
            rollback_path=data.get("rollback_path", ""),
            cryptographic_signatures=data.get("cryptographic_signatures", []),
            # Invariant assertion forced to constant - cannot be set from JSON
            invariant_assertion=TRUST_INVARIANT_ASSERTION,
            status=ProposalStatus(data.get("status", "pending")),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


class MerkleTreeBuilder:
    """Builder for merkle trees used in provenance tracking.

    Provides methods to construct merkle trees from data and
    verify merkle proofs.
    """

    @staticmethod
    def compute_hash(data: str) -> str:
        """Compute SHA-256 hash of data.

        Args:
            data: String data to hash

        Returns:
            Hex-encoded SHA-256 hash
        """
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def compute_merkle_root(data_items: list[str]) -> tuple[str, list[MerkleNode]]:
        """Compute merkle root from list of data items.

        Args:
            data_items: List of data strings to include

        Returns:
            Tuple of (root_hash, list_of_nodes)
        """
        if not data_items:
            return "", []

        # Create leaf nodes
        nodes = []
        for item in data_items:
            item_hash = MerkleTreeBuilder.compute_hash(item)
            nodes.append(MerkleNode(hash=item_hash, data=item))

        all_nodes = nodes.copy()

        # Build tree bottom-up
        while len(nodes) > 1:
            next_level = []

            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else left

                combined = left.hash + right.hash
                parent_hash = MerkleTreeBuilder.compute_hash(combined)

                parent = MerkleNode(
                    hash=parent_hash,
                    left_child=left.hash,
                    right_child=right.hash,
                )
                next_level.append(parent)
                all_nodes.append(parent)

            nodes = next_level

        return nodes[0].hash if nodes else "", all_nodes

    @staticmethod
    def verify_merkle_proof(
        item: str, proof: list[tuple[str, str]], root: str
    ) -> bool:
        """Verify a merkle proof for an item.

        Args:
            item: Data item to verify
            proof: List of (sibling_hash, direction) tuples
            root: Expected merkle root

        Returns:
            True if proof is valid
        """
        current_hash = MerkleTreeBuilder.compute_hash(item)

        for sibling_hash, direction in proof:
            if direction == "left":
                combined = sibling_hash + current_hash
            else:
                combined = current_hash + sibling_hash

            current_hash = MerkleTreeBuilder.compute_hash(combined)

        return current_hash == root


class ReinjectionEvaluationEngine:
    """Proposal-only reinjection evaluation engine.

    This engine evaluates quantum execution outputs and creates standardized
    proposal artifacts for human/dual-approved reinjection into QRATUM pipelines.

    Core features:
    - Pre-validation scoring
    - Merkle root tracking
    - Dual-control approval integration
    - Proposal lifecycle management

    Example:
        >>> engine = ReinjectionEvaluationEngine()
        >>> proposal = engine.create_proposal(
        ...     execution_result=result,
        ...     cluster=ProposalCluster.P0,
        ...     metrics_target={"primary": "10x_speedup", "secondary": "zk_latency_5s"},
        ... )
        >>> if proposal.pre_validation.passed:
        ...     # Submit for dual-control approval
        ...     engine.submit_for_approval(proposal.proposal_id)
    """

    def __init__(
        self,
        require_dual_approval: bool = True,
        pre_validation_threshold: float = 0.7,
        max_proposal_age_hours: int = 48,
    ):
        """Initialize evaluation engine.

        Args:
            require_dual_approval: Whether dual-control approval is required
            pre_validation_threshold: Minimum score for pre-validation pass
            max_proposal_age_hours: Maximum proposal age before expiration (P2: 48h)
        """
        self.require_dual_approval = require_dual_approval
        self.pre_validation_threshold = pre_validation_threshold
        self.max_proposal_age_hours = max_proposal_age_hours

        self._proposals: dict[str, ProposalArtifact] = {}
        self._merkle_trees: dict[str, list[MerkleNode]] = {}
        self._validation_hooks: list[Callable[[Any], tuple[bool, str]]] = []

    def register_validation_hook(
        self, hook: Callable[[Any], tuple[bool, str]]
    ) -> None:
        """Register a custom validation hook.

        Args:
            hook: Callable that takes data and returns (passed, message)
        """
        self._validation_hooks.append(hook)

    def create_proposal(
        self,
        execution_result: dict[str, Any],
        cluster: ProposalCluster = ProposalCluster.P1,
        metrics_target: dict[str, str] | None = None,
        input_data: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProposalArtifact:
        """Create a new reinjection proposal.

        Args:
            execution_result: Result from quantum execution
            cluster: Priority enhancement cluster
            metrics_target: Primary and secondary metric targets
            input_data: Raw input data for merkle tree
            metadata: Additional metadata

        Returns:
            ProposalArtifact with pre-validation scoring
        """
        proposal_id = str(uuid.uuid4())

        # Build merkle tree from inputs
        inputs_for_merkle = input_data or [json.dumps(execution_result, default=str)]
        merkle_root, merkle_nodes = MerkleTreeBuilder.compute_merkle_root(inputs_for_merkle)
        self._merkle_trees[proposal_id] = merkle_nodes

        # Perform pre-validation
        pre_validation = self._pre_validate(execution_result)

        # Create expected outputs
        expected_outputs = {
            "artifacts": [execution_result.get("provenance_hash", "")],
            "metrics_delta": {
                "trust": execution_result.get("trust_metric", 0),
                "success": execution_result.get("success", False),
            },
        }

        # Create proposal artifact
        proposal = ProposalArtifact(
            proposal_id=proposal_id,
            cluster=cluster,
            metrics_target=metrics_target or {"primary": "success", "secondary": "trust_preserved"},
            inputs=[merkle_root],
            expected_outputs=expected_outputs,
            fallback_strategy=self._determine_fallback_strategy(execution_result),
            rollback_path=merkle_root,
            pre_validation=pre_validation,
            metadata=metadata or {},
        )

        # Update status based on pre-validation
        if pre_validation.passed:
            proposal.status = ProposalStatus.PRE_VALIDATED
        else:
            proposal.status = ProposalStatus.PENDING

        self._proposals[proposal_id] = proposal
        return proposal

    def submit_for_approval(self, proposal_id: str) -> dict[str, Any]:
        """Submit proposal for dual-control approval.

        Args:
            proposal_id: ID of proposal to submit

        Returns:
            Submission result dictionary
        """
        if proposal_id not in self._proposals:
            return {"success": False, "error": "Proposal not found"}

        proposal = self._proposals[proposal_id]

        if proposal.pre_validation is None or not proposal.pre_validation.passed:
            return {
                "success": False,
                "error": "Proposal has not passed pre-validation",
            }

        proposal.status = ProposalStatus.AWAITING_APPROVAL
        proposal.updated_at = datetime.utcnow().isoformat()

        return {
            "success": True,
            "proposal_id": proposal_id,
            "status": proposal.status.value,
            "requires_approvals": 2 if self.require_dual_approval else 1,
        }

    def approve(
        self,
        proposal_id: str,
        approver_id: str,
        signature: str = "",
        notes: str = "",
    ) -> dict[str, Any]:
        """Record approval for a proposal.

        Args:
            proposal_id: ID of proposal to approve
            approver_id: ID of approving entity
            signature: Cryptographic signature (placeholder)
            notes: Optional approval notes

        Returns:
            Approval result dictionary
        """
        if proposal_id not in self._proposals:
            return {"success": False, "error": "Proposal not found"}

        proposal = self._proposals[proposal_id]

        # Check if already approved by this entity
        if any(a["approver_id"] == approver_id for a in proposal.approvals):
            return {"success": False, "error": "Already approved by this entity"}

        # Record approval
        approval_record = {
            "approver_id": approver_id,
            "signature": signature,
            "notes": notes,
            "timestamp": datetime.utcnow().isoformat(),
        }
        proposal.approvals.append(approval_record)

        # Add signature to artifact
        if signature:
            proposal.cryptographic_signatures.append(signature)

        # Check if fully approved
        required_approvals = 2 if self.require_dual_approval else 1
        if len(proposal.approvals) >= required_approvals:
            proposal.status = ProposalStatus.APPROVED

        proposal.updated_at = datetime.utcnow().isoformat()

        return {
            "success": True,
            "proposal_id": proposal_id,
            "approvals_count": len(proposal.approvals),
            "required_approvals": required_approvals,
            "is_fully_approved": proposal.status == ProposalStatus.APPROVED,
        }

    def reject(
        self,
        proposal_id: str,
        rejector_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Reject a proposal.

        Args:
            proposal_id: ID of proposal to reject
            rejector_id: ID of rejecting entity
            reason: Rejection reason

        Returns:
            Rejection result dictionary
        """
        if proposal_id not in self._proposals:
            return {"success": False, "error": "Proposal not found"}

        proposal = self._proposals[proposal_id]

        rejection_record = {
            "rejector_id": rejector_id,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }
        proposal.rejections.append(rejection_record)
        proposal.status = ProposalStatus.REJECTED
        proposal.updated_at = datetime.utcnow().isoformat()

        return {
            "success": True,
            "proposal_id": proposal_id,
            "status": proposal.status.value,
        }

    def execute_approved(
        self,
        proposal_id: str,
        execution_fn: Callable[[dict[str, Any]], Any],
    ) -> dict[str, Any]:
        """Execute an approved proposal (deterministic execution).

        Args:
            proposal_id: ID of approved proposal
            execution_fn: Function to execute the proposal

        Returns:
            Execution result dictionary
        """
        if proposal_id not in self._proposals:
            return {"success": False, "error": "Proposal not found"}

        proposal = self._proposals[proposal_id]

        if proposal.status != ProposalStatus.APPROVED:
            return {
                "success": False,
                "error": f"Proposal not approved (status: {proposal.status.value})",
            }

        try:
            result = execution_fn(proposal.expected_outputs)
            proposal.status = ProposalStatus.EXECUTED
            proposal.updated_at = datetime.utcnow().isoformat()

            return {
                "success": True,
                "proposal_id": proposal_id,
                "status": proposal.status.value,
                "result": result,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "proposal_id": proposal_id,
            }

    def rollback(self, proposal_id: str) -> dict[str, Any]:
        """Rollback an executed proposal.

        Args:
            proposal_id: ID of proposal to rollback

        Returns:
            Rollback result dictionary
        """
        if proposal_id not in self._proposals:
            return {"success": False, "error": "Proposal not found"}

        proposal = self._proposals[proposal_id]

        if proposal.status != ProposalStatus.EXECUTED:
            return {
                "success": False,
                "error": f"Can only rollback executed proposals (status: {proposal.status.value})",
            }

        proposal.status = ProposalStatus.ROLLED_BACK
        proposal.updated_at = datetime.utcnow().isoformat()

        return {
            "success": True,
            "proposal_id": proposal_id,
            "rollback_path": proposal.rollback_path,
            "status": proposal.status.value,
        }

    def get_proposal(self, proposal_id: str) -> ProposalArtifact | None:
        """Get proposal by ID.

        Args:
            proposal_id: Proposal ID to retrieve

        Returns:
            ProposalArtifact or None if not found
        """
        return self._proposals.get(proposal_id)

    def get_pending_proposals(self) -> list[ProposalArtifact]:
        """Get all proposals awaiting approval.

        Returns:
            List of pending proposals
        """
        return [
            p
            for p in self._proposals.values()
            if p.status == ProposalStatus.AWAITING_APPROVAL
        ]

    def get_proposals_by_cluster(self, cluster: ProposalCluster) -> list[ProposalArtifact]:
        """Get proposals filtered by cluster.

        Args:
            cluster: Cluster to filter by

        Returns:
            List of matching proposals
        """
        return [p for p in self._proposals.values() if p.cluster == cluster]

    def verify_merkle_membership(
        self, proposal_id: str, item: str, proof: list[tuple[str, str]]
    ) -> bool:
        """Verify merkle proof for proposal input.

        Args:
            proposal_id: ID of proposal
            item: Item to verify
            proof: Merkle proof path

        Returns:
            True if proof is valid
        """
        if proposal_id not in self._proposals:
            return False

        proposal = self._proposals[proposal_id]
        merkle_root = proposal.inputs[0] if proposal.inputs else ""

        return MerkleTreeBuilder.verify_merkle_proof(item, proof, merkle_root)

    def generate_audit_report(self) -> dict[str, Any]:
        """Generate audit report with ℛ(t) ≥ 0 assertion.

        Returns:
            Audit report dictionary
        """
        proposals_by_status = {}
        for p in self._proposals.values():
            status = p.status.value
            proposals_by_status[status] = proposals_by_status.get(status, 0) + 1

        proposals_by_cluster = {}
        for p in self._proposals.values():
            cluster = p.cluster.value
            proposals_by_cluster[cluster] = proposals_by_cluster.get(cluster, 0) + 1

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "invariant_assertion": "ℛ(t) ≥ 0",
            "total_proposals": len(self._proposals),
            "proposals_by_status": proposals_by_status,
            "proposals_by_cluster": proposals_by_cluster,
            "pending_approval_count": len(self.get_pending_proposals()),
            "dual_approval_required": self.require_dual_approval,
            "max_proposal_age_hours": self.max_proposal_age_hours,
        }

    def _pre_validate(self, execution_result: dict[str, Any]) -> PreValidationScore:
        """Perform pre-validation scoring.

        Args:
            execution_result: Result to validate

        Returns:
            PreValidationScore with check results
        """
        checks = {}
        warnings = []
        errors = []

        # Check 1: Execution success
        success = execution_result.get("success", False)
        checks["execution_success"] = success
        if not success:
            errors.append("Execution did not succeed")

        # Check 2: Trust metric preserved
        trust = execution_result.get("trust_metric", 0)
        checks["trust_preserved"] = trust >= 0
        if trust < 0:
            errors.append(f"Trust invariant violated: ℛ(t) = {trust} < 0")

        # Check 3: Provenance hash present
        provenance_hash = execution_result.get("provenance_hash", "")
        checks["provenance_present"] = bool(provenance_hash)
        if not provenance_hash:
            warnings.append("No provenance hash in result")

        # Check 4: No human escalation required
        escalation = execution_result.get("requires_human_escalation", False)
        checks["no_escalation_required"] = not escalation
        if escalation:
            warnings.append("Result requires human escalation")

        # Run custom validation hooks
        for i, hook in enumerate(self._validation_hooks):
            try:
                passed, message = hook(execution_result)
                checks[f"custom_hook_{i}"] = passed
                if not passed:
                    warnings.append(message)
            except Exception as e:
                checks[f"custom_hook_{i}"] = False
                errors.append(f"Custom hook {i} failed: {str(e)}")

        # Compute score
        passed_count = sum(1 for v in checks.values() if v)
        total_count = len(checks)
        score = passed_count / total_count if total_count > 0 else 0.0

        # Determine if passed
        passed = score >= self.pre_validation_threshold and len(errors) == 0

        return PreValidationScore(
            score=score,
            passed=passed,
            checks=checks,
            warnings=warnings,
            errors=errors,
        )

    def _determine_fallback_strategy(self, execution_result: dict[str, Any]) -> str:
        """Determine fallback strategy based on execution result.

        Args:
            execution_result: Execution result to analyze

        Returns:
            Fallback strategy description
        """
        mode_used = execution_result.get("mode_used", "quantum")

        if mode_used == "fallback":
            return "classical_tensor_fallback_used"
        elif execution_result.get("failures"):
            return "bounded_retry_max_3_then_classical"
        else:
            return "quantum_success_no_fallback_needed"
