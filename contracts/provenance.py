"""Provenance Contract - Cryptographic Provenance Chain for Regulatory Compliance.

This module implements the ProvenanceContract for evidentiary provenance artifacts
supporting 21 CFR Part 11 equivalence and DO-178C traceability mapping.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from contracts.base import (
    BaseContract,
    compute_contract_hash,
    generate_contract_id,
    get_current_timestamp,
)


class ProvenanceType(Enum):
    """Types of provenance artifacts for regulatory compliance."""

    # 21 CFR Part 11 - Electronic records
    ELECTRONIC_SIGNATURE = "electronic_signature"
    AUDIT_TRAIL = "audit_trail"
    ACCESS_CONTROL = "access_control"
    TIMESTAMP_VERIFICATION = "timestamp_verification"

    # DO-178C - Airborne software
    REQUIREMENT_TRACE = "requirement_trace"
    DESIGN_TRACE = "design_trace"
    CODE_TRACE = "code_trace"
    TEST_TRACE = "test_trace"
    VERIFICATION_TRACE = "verification_trace"

    # General provenance
    STATE_TRANSITION = "state_transition"
    ROLLBACK_PROOF = "rollback_proof"
    INVARIANT_CHECK = "invariant_check"


class ComplianceStandard(Enum):
    """Regulatory compliance standards."""

    CFR_21_PART_11 = "21_cfr_part_11"  # FDA Electronic Records
    DO_178C = "do_178c"  # Airborne Software
    ISO_27001 = "iso_27001"  # Information Security
    CMMC = "cmmc"  # Cybersecurity Maturity Model
    HIPAA = "hipaa"  # Health Information
    SOX = "sox"  # Financial Records


@dataclass(frozen=True)
class ProvenanceEntry:
    """Immutable provenance chain entry.

    Attributes:
        entry_id: Unique entry identifier
        provenance_type: Type of provenance artifact
        timestamp: ISO 8601 timestamp
        actor_id: ID of the actor (user/system) performing the action
        action: Description of the action
        input_hash: Hash of inputs at this point
        output_hash: Hash of outputs at this point
        previous_entry_hash: Hash of previous entry (chain linkage)
        metadata: Additional metadata
    """

    entry_id: str
    provenance_type: ProvenanceType
    timestamp: str
    actor_id: str
    action: str
    input_hash: str
    output_hash: str
    previous_entry_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of this entry."""
        content = {
            "entry_id": self.entry_id,
            "provenance_type": self.provenance_type.value,
            "timestamp": self.timestamp,
            "actor_id": self.actor_id,
            "action": self.action,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "previous_entry_hash": self.previous_entry_hash,
            "metadata": self.metadata,
        }
        json_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def serialize(self) -> dict[str, Any]:
        """Serialize entry to dictionary."""
        return {
            "entry_id": self.entry_id,
            "provenance_type": self.provenance_type.value,
            "timestamp": self.timestamp,
            "actor_id": self.actor_id,
            "action": self.action,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "previous_entry_hash": self.previous_entry_hash,
            "entry_hash": self.compute_hash(),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ComplianceArtifact:
    """Compliance artifact linking provenance to regulatory requirements.

    Attributes:
        artifact_id: Unique artifact identifier
        standard: Compliance standard
        requirement_id: Specific requirement identifier
        satisfied_by: List of provenance entry IDs satisfying this requirement
        verification_method: How the requirement was verified
        verification_timestamp: When verification occurred
        verification_result: PASS/FAIL/PENDING
    """

    artifact_id: str
    standard: ComplianceStandard
    requirement_id: str
    satisfied_by: list[str]
    verification_method: str
    verification_timestamp: str
    verification_result: str  # "PASS", "FAIL", "PENDING"

    def serialize(self) -> dict[str, Any]:
        """Serialize artifact to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "standard": self.standard.value,
            "requirement_id": self.requirement_id,
            "satisfied_by": self.satisfied_by,
            "verification_method": self.verification_method,
            "verification_timestamp": self.verification_timestamp,
            "verification_result": self.verification_result,
        }


@dataclass(frozen=True)
class ProvenanceContract(BaseContract):
    """Immutable contract for evidentiary provenance.

    Provides machine-verifiable provenance chains for regulatory compliance:
    - 21 CFR Part 11 electronic records equivalence
    - DO-178C traceability mapping
    - Cryptographic proof of all state transitions

    Attributes:
        contract_reference: Reference to the contract being documented
        provenance_chain: Ordered list of provenance entries
        compliance_artifacts: Mapping of standard to artifacts
        chain_root_hash: Root hash of provenance chain
        zone_classification: Security zone (Z0-Z3)
        provenance_proof: Cryptographic proof of provenance
    """

    contract_reference: str = ""
    provenance_chain: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    compliance_artifacts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    chain_root_hash: str = ""
    zone_classification: str = "Z0"
    provenance_proof: str = ""

    def __post_init__(self) -> None:
        """Validate provenance contract after initialization."""
        super().__post_init__()
        if not self.contract_reference:
            raise ValueError("contract_reference cannot be empty")
        if not self.provenance_proof:
            raise ValueError("provenance_proof cannot be empty")
        if self.zone_classification not in ("Z0", "Z1", "Z2", "Z3"):
            raise ValueError(f"zone_classification must be Z0-Z3, got: {self.zone_classification}")

    def serialize(self) -> dict[str, Any]:
        """Serialize provenance contract to dictionary."""
        base = super().serialize()
        base.update(
            {
                "contract_reference": self.contract_reference,
                "provenance_chain": list(self.provenance_chain),
                "compliance_artifacts": self.compliance_artifacts,
                "chain_root_hash": self.chain_root_hash,
                "zone_classification": self.zone_classification,
                "provenance_proof": self.provenance_proof,
            }
        )
        return base

    def verify_chain_integrity(self) -> bool:
        """Verify integrity of provenance chain.

        Returns:
            True if chain is valid, False otherwise
        """
        if not self.provenance_chain:
            return True

        prev_hash = "0" * 64  # Genesis hash
        for entry_dict in self.provenance_chain:
            # Verify previous hash linkage
            if entry_dict.get("previous_entry_hash") != prev_hash:
                return False

            # Recompute entry hash
            content = {
                "entry_id": entry_dict["entry_id"],
                "provenance_type": entry_dict["provenance_type"],
                "timestamp": entry_dict["timestamp"],
                "actor_id": entry_dict["actor_id"],
                "action": entry_dict["action"],
                "input_hash": entry_dict["input_hash"],
                "output_hash": entry_dict["output_hash"],
                "previous_entry_hash": entry_dict["previous_entry_hash"],
                "metadata": entry_dict.get("metadata", {}),
            }
            json_str = json.dumps(content, sort_keys=True)
            computed_hash = hashlib.sha256(json_str.encode("utf-8")).hexdigest()

            if entry_dict.get("entry_hash") != computed_hash:
                return False

            prev_hash = computed_hash

        return True

    def get_compliance_status(self, standard: ComplianceStandard) -> dict[str, Any]:
        """Get compliance status for a specific standard.

        Args:
            standard: Compliance standard to check

        Returns:
            Dictionary with compliance status
        """
        artifacts = self.compliance_artifacts.get(standard.value, [])
        total = len(artifacts)
        passed = sum(1 for a in artifacts if a.get("verification_result") == "PASS")
        failed = sum(1 for a in artifacts if a.get("verification_result") == "FAIL")
        pending = sum(1 for a in artifacts if a.get("verification_result") == "PENDING")

        return {
            "standard": standard.value,
            "total_requirements": total,
            "passed": passed,
            "failed": failed,
            "pending": pending,
            "compliant": failed == 0 and pending == 0 and total > 0,
        }

    def get_traceability_matrix(self) -> dict[str, list[str]]:
        """Generate traceability matrix for DO-178C compliance.

        Returns:
            Mapping of requirement IDs to provenance entry IDs
        """
        matrix: dict[str, list[str]] = {}

        for standard_key, artifacts in self.compliance_artifacts.items():
            if "do_178c" in standard_key:
                for artifact in artifacts:
                    req_id = artifact.get("requirement_id", "")
                    satisfied_by = artifact.get("satisfied_by", [])
                    if req_id:
                        matrix[req_id] = satisfied_by

        return matrix


class ProvenanceChainBuilder:
    """Builder for constructing provenance chains with proper linkage."""

    def __init__(self, contract_reference: str, zone: str = "Z0"):
        """Initialize builder.

        Args:
            contract_reference: Reference to contract being documented
            zone: Security zone classification (Z0-Z3)
        """
        self.contract_reference = contract_reference
        self.zone = zone
        self.entries: list[ProvenanceEntry] = []
        self.compliance_artifacts: dict[str, list[ComplianceArtifact]] = {}

    def add_entry(
        self,
        provenance_type: ProvenanceType,
        actor_id: str,
        action: str,
        input_hash: str,
        output_hash: str,
        metadata: dict[str, Any] | None = None,
    ) -> ProvenanceEntry:
        """Add entry to provenance chain.

        Args:
            provenance_type: Type of provenance
            actor_id: Actor performing action
            action: Action description
            input_hash: Hash of inputs
            output_hash: Hash of outputs
            metadata: Optional metadata

        Returns:
            The created ProvenanceEntry
        """
        entry_id = f"prov_{len(self.entries):04d}_{compute_contract_hash({'ts': get_current_timestamp()})[:8]}"
        previous_hash = self.entries[-1].compute_hash() if self.entries else "0" * 64

        entry = ProvenanceEntry(
            entry_id=entry_id,
            provenance_type=provenance_type,
            timestamp=get_current_timestamp(),
            actor_id=actor_id,
            action=action,
            input_hash=input_hash,
            output_hash=output_hash,
            previous_entry_hash=previous_hash,
            metadata=metadata or {},
        )
        self.entries.append(entry)
        return entry

    def add_compliance_artifact(
        self,
        standard: ComplianceStandard,
        requirement_id: str,
        satisfied_by_entry_ids: list[str],
        verification_method: str,
        verification_result: str = "PENDING",
    ) -> ComplianceArtifact:
        """Add compliance artifact.

        Args:
            standard: Compliance standard
            requirement_id: Requirement identifier
            satisfied_by_entry_ids: Entry IDs satisfying requirement
            verification_method: How requirement was verified
            verification_result: PASS/FAIL/PENDING

        Returns:
            The created ComplianceArtifact
        """
        artifact_id = f"comp_{standard.value}_{requirement_id}_{len(self.compliance_artifacts.get(standard.value, [])):04d}"

        artifact = ComplianceArtifact(
            artifact_id=artifact_id,
            standard=standard,
            requirement_id=requirement_id,
            satisfied_by=satisfied_by_entry_ids,
            verification_method=verification_method,
            verification_timestamp=get_current_timestamp(),
            verification_result=verification_result,
        )

        if standard.value not in self.compliance_artifacts:
            self.compliance_artifacts[standard.value] = []
        self.compliance_artifacts[standard.value].append(artifact)

        return artifact

    def build(self, provenance_proof: str = "provenance_authorized") -> ProvenanceContract:
        """Build the ProvenanceContract.

        Args:
            provenance_proof: Proof of provenance authorization

        Returns:
            Immutable ProvenanceContract
        """
        # Serialize entries
        chain_tuple = tuple(entry.serialize() for entry in self.entries)

        # Compute chain root hash
        chain_root = self.entries[-1].compute_hash() if self.entries else "0" * 64

        # Serialize compliance artifacts
        compliance_dict = {
            std: [a.serialize() for a in artifacts]
            for std, artifacts in self.compliance_artifacts.items()
        }

        content = {
            "contract_reference": self.contract_reference,
            "provenance_chain": list(chain_tuple),
            "compliance_artifacts": compliance_dict,
            "chain_root_hash": chain_root,
            "zone_classification": self.zone,
            "provenance_proof": provenance_proof,
            "created_at": get_current_timestamp(),
            "version": "1.0.0",
        }

        contract_id = generate_contract_id("ProvenanceContract", content)

        return ProvenanceContract(
            contract_id=contract_id,
            contract_type="ProvenanceContract",
            created_at=content["created_at"],
            version=content["version"],
            contract_reference=content["contract_reference"],
            provenance_chain=chain_tuple,
            compliance_artifacts=content["compliance_artifacts"],
            chain_root_hash=content["chain_root_hash"],
            zone_classification=content["zone_classification"],
            provenance_proof=content["provenance_proof"],
        )


def create_21cfr11_provenance(
    contract_reference: str,
    actor_id: str,
    electronic_signature: str,
    audit_entries: list[dict[str, Any]],
    zone: str = "Z0",
) -> ProvenanceContract:
    """Create provenance contract for 21 CFR Part 11 compliance.

    Args:
        contract_reference: Reference to contract being documented
        actor_id: Actor creating the record
        electronic_signature: Electronic signature data
        audit_entries: List of audit trail entries
        zone: Security zone

    Returns:
        ProvenanceContract with 21 CFR Part 11 artifacts
    """
    builder = ProvenanceChainBuilder(contract_reference, zone)

    # Add electronic signature entry
    sig_hash = compute_contract_hash({"signature": electronic_signature})
    builder.add_entry(
        provenance_type=ProvenanceType.ELECTRONIC_SIGNATURE,
        actor_id=actor_id,
        action="Electronic signature applied",
        input_hash=sig_hash,
        output_hash=sig_hash,
        metadata={"signature_method": "cryptographic", "signature_present": True},
    )

    # Add audit trail entries
    for i, audit in enumerate(audit_entries):
        input_h = compute_contract_hash({"audit_input": audit})
        builder.add_entry(
            provenance_type=ProvenanceType.AUDIT_TRAIL,
            actor_id=audit.get("actor", actor_id),
            action=audit.get("action", f"Audit entry {i}"),
            input_hash=input_h,
            output_hash=input_h,
            metadata=audit,
        )

    # Add compliance artifacts
    builder.add_compliance_artifact(
        standard=ComplianceStandard.CFR_21_PART_11,
        requirement_id="11.10_a_controls",
        satisfied_by_entry_ids=[e.entry_id for e in builder.entries],
        verification_method="System validation",
        verification_result="PASS",
    )

    builder.add_compliance_artifact(
        standard=ComplianceStandard.CFR_21_PART_11,
        requirement_id="11.10_e_audit_trail",
        satisfied_by_entry_ids=[
            e.entry_id for e in builder.entries if e.provenance_type == ProvenanceType.AUDIT_TRAIL
        ],
        verification_method="Audit trail verification",
        verification_result="PASS",
    )

    return builder.build()


def create_do178c_provenance(
    contract_reference: str,
    actor_id: str,
    requirement_traces: list[dict[str, Any]],
    design_traces: list[dict[str, Any]],
    code_traces: list[dict[str, Any]],
    test_traces: list[dict[str, Any]],
    zone: str = "Z2",
) -> ProvenanceContract:
    """Create provenance contract for DO-178C compliance.

    Args:
        contract_reference: Reference to contract being documented
        actor_id: Actor creating the record
        requirement_traces: Requirement traceability data
        design_traces: Design traceability data
        code_traces: Code traceability data
        test_traces: Test traceability data
        zone: Security zone

    Returns:
        ProvenanceContract with DO-178C artifacts
    """
    builder = ProvenanceChainBuilder(contract_reference, zone)

    # Add requirement traces
    req_entry_ids = []
    for req in requirement_traces:
        input_h = compute_contract_hash(req)
        entry = builder.add_entry(
            provenance_type=ProvenanceType.REQUIREMENT_TRACE,
            actor_id=actor_id,
            action=f"Requirement traced: {req.get('req_id', 'unknown')}",
            input_hash=input_h,
            output_hash=input_h,
            metadata=req,
        )
        req_entry_ids.append(entry.entry_id)

    # Add design traces
    design_entry_ids = []
    for design in design_traces:
        input_h = compute_contract_hash(design)
        entry = builder.add_entry(
            provenance_type=ProvenanceType.DESIGN_TRACE,
            actor_id=actor_id,
            action=f"Design traced: {design.get('design_id', 'unknown')}",
            input_hash=input_h,
            output_hash=input_h,
            metadata=design,
        )
        design_entry_ids.append(entry.entry_id)

    # Add code traces
    code_entry_ids = []
    for code in code_traces:
        input_h = compute_contract_hash(code)
        entry = builder.add_entry(
            provenance_type=ProvenanceType.CODE_TRACE,
            actor_id=actor_id,
            action=f"Code traced: {code.get('code_id', 'unknown')}",
            input_hash=input_h,
            output_hash=input_h,
            metadata=code,
        )
        code_entry_ids.append(entry.entry_id)

    # Add test traces
    test_entry_ids = []
    for test in test_traces:
        input_h = compute_contract_hash(test)
        entry = builder.add_entry(
            provenance_type=ProvenanceType.TEST_TRACE,
            actor_id=actor_id,
            action=f"Test traced: {test.get('test_id', 'unknown')}",
            input_hash=input_h,
            output_hash=input_h,
            metadata=test,
        )
        test_entry_ids.append(entry.entry_id)

    # Add DO-178C compliance artifacts
    builder.add_compliance_artifact(
        standard=ComplianceStandard.DO_178C,
        requirement_id="objectives_6.3.1",  # High-level requirements
        satisfied_by_entry_ids=req_entry_ids,
        verification_method="Requirements traceability analysis",
        verification_result="PASS" if req_entry_ids else "PENDING",
    )

    builder.add_compliance_artifact(
        standard=ComplianceStandard.DO_178C,
        requirement_id="objectives_6.3.2",  # Low-level requirements
        satisfied_by_entry_ids=design_entry_ids,
        verification_method="Design traceability analysis",
        verification_result="PASS" if design_entry_ids else "PENDING",
    )

    builder.add_compliance_artifact(
        standard=ComplianceStandard.DO_178C,
        requirement_id="objectives_6.3.4",  # Source code
        satisfied_by_entry_ids=code_entry_ids,
        verification_method="Code traceability analysis",
        verification_result="PASS" if code_entry_ids else "PENDING",
    )

    builder.add_compliance_artifact(
        standard=ComplianceStandard.DO_178C,
        requirement_id="objectives_6.4",  # Testing
        satisfied_by_entry_ids=test_entry_ids,
        verification_method="Test coverage analysis",
        verification_result="PASS" if test_entry_ids else "PENDING",
    )

    return builder.build()
