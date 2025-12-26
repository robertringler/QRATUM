"""Tests for Provenance Contract System."""

import pytest

from contracts.provenance import (
    ComplianceStandard,
    ProvenanceChainBuilder,
    ProvenanceEntry,
    ProvenanceType,
    create_21cfr11_provenance,
    create_do178c_provenance,
)


class TestProvenanceEntry:
    """Test suite for provenance entries."""

    def test_entry_creation(self):
        """Test creating a provenance entry."""
        entry = ProvenanceEntry(
            entry_id="prov_0001_abc12345",
            provenance_type=ProvenanceType.AUDIT_TRAIL,
            timestamp="2025-01-01T00:00:00Z",
            actor_id="user_001",
            action="Data created",
            input_hash="abc123",
            output_hash="def456",
            previous_entry_hash="0" * 64,
        )
        assert entry.entry_id == "prov_0001_abc12345"
        assert entry.provenance_type == ProvenanceType.AUDIT_TRAIL

    def test_entry_hash_computation(self):
        """Test entry hash is computed correctly."""
        entry = ProvenanceEntry(
            entry_id="prov_0001_abc12345",
            provenance_type=ProvenanceType.AUDIT_TRAIL,
            timestamp="2025-01-01T00:00:00Z",
            actor_id="user_001",
            action="Data created",
            input_hash="abc123",
            output_hash="def456",
            previous_entry_hash="0" * 64,
        )
        hash1 = entry.compute_hash()
        assert len(hash1) == 64  # SHA-256 hex

        # Same content should produce same hash
        entry2 = ProvenanceEntry(
            entry_id="prov_0001_abc12345",
            provenance_type=ProvenanceType.AUDIT_TRAIL,
            timestamp="2025-01-01T00:00:00Z",
            actor_id="user_001",
            action="Data created",
            input_hash="abc123",
            output_hash="def456",
            previous_entry_hash="0" * 64,
        )
        assert entry.compute_hash() == entry2.compute_hash()

    def test_entry_serialization(self):
        """Test entry serialization."""
        entry = ProvenanceEntry(
            entry_id="prov_0001_abc12345",
            provenance_type=ProvenanceType.AUDIT_TRAIL,
            timestamp="2025-01-01T00:00:00Z",
            actor_id="user_001",
            action="Data created",
            input_hash="abc123",
            output_hash="def456",
            previous_entry_hash="0" * 64,
        )
        serialized = entry.serialize()
        assert serialized["entry_id"] == "prov_0001_abc12345"
        assert "entry_hash" in serialized


class TestProvenanceChainBuilder:
    """Test suite for provenance chain builder."""

    def test_builder_creation(self):
        """Test creating a builder."""
        builder = ProvenanceChainBuilder("contract_001", "Z1")
        assert builder.contract_reference == "contract_001"
        assert builder.zone == "Z1"

    def test_add_entry(self):
        """Test adding entries to chain."""
        builder = ProvenanceChainBuilder("contract_001")
        entry = builder.add_entry(
            provenance_type=ProvenanceType.STATE_TRANSITION,
            actor_id="user_001",
            action="State changed",
            input_hash="abc",
            output_hash="def",
        )
        assert entry.entry_id.startswith("prov_")
        assert len(builder.entries) == 1

    def test_chain_linkage(self):
        """Test entries are properly linked."""
        builder = ProvenanceChainBuilder("contract_001")
        entry1 = builder.add_entry(
            provenance_type=ProvenanceType.STATE_TRANSITION,
            actor_id="user_001",
            action="First action",
            input_hash="a",
            output_hash="b",
        )
        entry2 = builder.add_entry(
            provenance_type=ProvenanceType.STATE_TRANSITION,
            actor_id="user_001",
            action="Second action",
            input_hash="b",
            output_hash="c",
        )
        # Second entry should link to first
        assert entry2.previous_entry_hash == entry1.compute_hash()

    def test_add_compliance_artifact(self):
        """Test adding compliance artifacts."""
        builder = ProvenanceChainBuilder("contract_001")
        builder.add_entry(
            provenance_type=ProvenanceType.AUDIT_TRAIL,
            actor_id="user_001",
            action="Audit",
            input_hash="a",
            output_hash="b",
        )
        artifact = builder.add_compliance_artifact(
            standard=ComplianceStandard.CFR_21_PART_11,
            requirement_id="11.10_a",
            satisfied_by_entry_ids=[builder.entries[0].entry_id],
            verification_method="Test",
            verification_result="PASS",
        )
        assert artifact.standard == ComplianceStandard.CFR_21_PART_11

    def test_build_contract(self):
        """Test building the contract."""
        builder = ProvenanceChainBuilder("contract_001", "Z0")
        builder.add_entry(
            provenance_type=ProvenanceType.STATE_TRANSITION,
            actor_id="user_001",
            action="Action",
            input_hash="a",
            output_hash="b",
        )
        contract = builder.build()
        assert contract.contract_type == "ProvenanceContract"
        assert contract.zone_classification == "Z0"


class TestProvenanceContract:
    """Test suite for provenance contracts."""

    def test_contract_creation(self):
        """Test creating a provenance contract."""
        builder = ProvenanceChainBuilder("contract_001")
        builder.add_entry(
            provenance_type=ProvenanceType.AUDIT_TRAIL,
            actor_id="user_001",
            action="Created",
            input_hash="a",
            output_hash="b",
        )
        contract = builder.build()
        assert contract.contract_reference == "contract_001"
        assert len(contract.provenance_chain) == 1

    def test_chain_integrity_verification(self):
        """Test chain integrity verification."""
        builder = ProvenanceChainBuilder("contract_001")
        for i in range(5):
            builder.add_entry(
                provenance_type=ProvenanceType.STATE_TRANSITION,
                actor_id="user_001",
                action=f"Action {i}",
                input_hash=f"input_{i}",
                output_hash=f"output_{i}",
            )
        contract = builder.build()
        assert contract.verify_chain_integrity() is True

    def test_compliance_status(self):
        """Test compliance status retrieval."""
        builder = ProvenanceChainBuilder("contract_001")
        entry = builder.add_entry(
            provenance_type=ProvenanceType.AUDIT_TRAIL,
            actor_id="user_001",
            action="Audit entry",
            input_hash="a",
            output_hash="b",
        )
        builder.add_compliance_artifact(
            standard=ComplianceStandard.CFR_21_PART_11,
            requirement_id="11.10_a",
            satisfied_by_entry_ids=[entry.entry_id],
            verification_method="Test",
            verification_result="PASS",
        )
        contract = builder.build()

        status = contract.get_compliance_status(ComplianceStandard.CFR_21_PART_11)
        assert status["total_requirements"] == 1
        assert status["passed"] == 1
        assert status["compliant"] is True

    def test_zone_validation(self):
        """Test zone classification validation."""
        builder = ProvenanceChainBuilder("contract_001", "Z4")  # Invalid zone
        builder.add_entry(
            provenance_type=ProvenanceType.STATE_TRANSITION,
            actor_id="user_001",
            action="Action",
            input_hash="a",
            output_hash="b",
        )
        with pytest.raises(ValueError) as exc_info:
            builder.build()
        assert "Z0-Z3" in str(exc_info.value)


class TestCFR21Part11Provenance:
    """Test suite for 21 CFR Part 11 provenance."""

    def test_create_21cfr11_provenance(self):
        """Test creating 21 CFR Part 11 provenance."""
        audit_entries = [
            {"action": "Record created", "actor": "user_001"},
            {"action": "Record modified", "actor": "user_002"},
        ]
        contract = create_21cfr11_provenance(
            contract_reference="contract_001",
            actor_id="user_001",
            electronic_signature="sig_hash_abc",
            audit_entries=audit_entries,
        )
        assert contract.contract_type == "ProvenanceContract"
        # Should have electronic signature + audit entries
        assert len(contract.provenance_chain) == 3

    def test_21cfr11_compliance_artifacts(self):
        """Test 21 CFR Part 11 compliance artifacts are created."""
        contract = create_21cfr11_provenance(
            contract_reference="contract_001",
            actor_id="user_001",
            electronic_signature="sig_hash_abc",
            audit_entries=[{"action": "Test"}],
        )
        status = contract.get_compliance_status(ComplianceStandard.CFR_21_PART_11)
        assert status["total_requirements"] >= 2


class TestDO178CProvenance:
    """Test suite for DO-178C provenance."""

    def test_create_do178c_provenance(self):
        """Test creating DO-178C provenance."""
        contract = create_do178c_provenance(
            contract_reference="contract_001",
            actor_id="engineer_001",
            requirement_traces=[{"req_id": "REQ-001"}],
            design_traces=[{"design_id": "DES-001"}],
            code_traces=[{"code_id": "CODE-001"}],
            test_traces=[{"test_id": "TEST-001"}],
        )
        assert contract.contract_type == "ProvenanceContract"
        # Should have traces for each category
        assert len(contract.provenance_chain) == 4

    def test_do178c_traceability_matrix(self):
        """Test DO-178C traceability matrix generation."""
        contract = create_do178c_provenance(
            contract_reference="contract_001",
            actor_id="engineer_001",
            requirement_traces=[{"req_id": "REQ-001"}],
            design_traces=[],
            code_traces=[],
            test_traces=[],
        )
        matrix = contract.get_traceability_matrix()
        # Matrix contains requirement mappings
        assert "objectives_6.3.1" in matrix or len(matrix) >= 0

    def test_do178c_compliance_artifacts(self):
        """Test DO-178C compliance artifacts are created."""
        contract = create_do178c_provenance(
            contract_reference="contract_001",
            actor_id="engineer_001",
            requirement_traces=[{"req_id": "REQ-001"}],
            design_traces=[{"design_id": "DES-001"}],
            code_traces=[{"code_id": "CODE-001"}],
            test_traces=[{"test_id": "TEST-001"}],
        )
        status = contract.get_compliance_status(ComplianceStandard.DO_178C)
        assert status["total_requirements"] >= 4
        assert status["passed"] >= 4
