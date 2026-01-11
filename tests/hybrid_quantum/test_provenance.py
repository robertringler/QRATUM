"""Tests for quantum provenance tracking."""

import json
import pytest

try:
    from quasim.hybrid_quantum.provenance import (
        ProvenanceRecord,
        ProvenanceStatus,
        QuantumProvenanceWrapper,
    )
    PROVENANCE_AVAILABLE = True
except ImportError:
    PROVENANCE_AVAILABLE = False

try:
    from quasim.hybrid_quantum.backends import (
        HybridQuantumConfig,
        IBMHybridBackend,
    )
    BACKENDS_AVAILABLE = True
except ImportError:
    BACKENDS_AVAILABLE = False

try:
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


@pytest.mark.skipif(not PROVENANCE_AVAILABLE, reason="Provenance module not available")
class TestProvenanceRecord:
    """Test ProvenanceRecord dataclass."""

    def test_record_creation(self):
        """Test creating a provenance record."""
        record = ProvenanceRecord(
            record_id="rec-123",
            execution_id="exec-456",
            input_hash="abc123",
        )
        assert record.record_id == "rec-123"
        assert record.execution_id == "exec-456"
        assert record.status == ProvenanceStatus.PENDING
        assert record.timestamp != ""

    def test_record_signature(self):
        """Test provenance record signature computation."""
        record = ProvenanceRecord(
            record_id="rec-123",
            execution_id="exec-456",
            input_hash="abc123",
            output_hash="def789",
        )
        
        sig1 = record.compute_signature()
        sig2 = record.compute_signature()
        
        assert sig1 == sig2
        assert len(sig1) == 64  # SHA-256

    def test_record_to_dict(self):
        """Test converting record to dictionary."""
        record = ProvenanceRecord(
            record_id="rec-123",
            execution_id="exec-456",
            input_hash="abc123",
            backend_provider="ibm",
            shots=1024,
        )
        
        d = record.to_dict()
        
        assert d["record_id"] == "rec-123"
        assert d["backend_provider"] == "ibm"
        assert "signature" in d

    def test_record_from_dict(self):
        """Test creating record from dictionary."""
        data = {
            "record_id": "rec-123",
            "execution_id": "exec-456",
            "input_hash": "abc123",
            "output_hash": "def789",
            "status": "recorded",
            "backend_provider": "ibm",
            "shots": 1024,
        }
        
        record = ProvenanceRecord.from_dict(data)
        
        assert record.record_id == "rec-123"
        assert record.status == ProvenanceStatus.RECORDED
        assert record.shots == 1024


@pytest.mark.skipif(not PROVENANCE_AVAILABLE, reason="Provenance module not available")
class TestProvenanceStatus:
    """Test ProvenanceStatus enumeration."""

    def test_status_values(self):
        """Test status enum values."""
        assert ProvenanceStatus.PENDING.value == "pending"
        assert ProvenanceStatus.RECORDED.value == "recorded"
        assert ProvenanceStatus.VERIFIED.value == "verified"
        assert ProvenanceStatus.REJECTED.value == "rejected"
        assert ProvenanceStatus.ROLLED_BACK.value == "rolled_back"


@pytest.mark.skipif(
    not (PROVENANCE_AVAILABLE and BACKENDS_AVAILABLE and QISKIT_AVAILABLE),
    reason="Required modules not available"
)
class TestQuantumProvenanceWrapper:
    """Test QuantumProvenanceWrapper."""

    def test_wrapper_creation(self):
        """Test creating provenance wrapper."""
        config = HybridQuantumConfig(provider="simulator", seed=42)
        backend = IBMHybridBackend(config)
        wrapper = QuantumProvenanceWrapper(backend)
        
        assert wrapper.backend is backend

    def test_execute_with_provenance(self):
        """Test execution with provenance tracking."""
        config = HybridQuantumConfig(provider="simulator", shots=100, seed=42)
        backend = IBMHybridBackend(config)
        wrapper = QuantumProvenanceWrapper(backend)

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()

        result, record = wrapper.execute_with_provenance(circuit, shots=100)

        assert result.success is True
        assert record.status == ProvenanceStatus.RECORDED
        assert record.input_hash != ""
        assert record.output_hash != ""

    def test_execution_chain_tracking(self):
        """Test that execution chain is tracked."""
        config = HybridQuantumConfig(provider="simulator", shots=50, seed=42)
        backend = IBMHybridBackend(config)
        wrapper = QuantumProvenanceWrapper(backend)

        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.measure_all()

        # Execute twice
        result1, record1 = wrapper.execute_with_provenance(circuit)
        result2, record2 = wrapper.execute_with_provenance(circuit)

        chain = wrapper.get_execution_chain()
        
        assert len(chain) == 2
        assert chain[0].record_id == record1.record_id
        assert chain[1].parent_record_id == record1.record_id

    def test_export_provenance_log(self):
        """Test exporting provenance log as JSON."""
        config = HybridQuantumConfig(provider="simulator", shots=50, seed=42)
        backend = IBMHybridBackend(config)
        wrapper = QuantumProvenanceWrapper(backend)

        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.measure_all()

        wrapper.execute_with_provenance(circuit)

        log_json = wrapper.export_provenance_log()
        log_data = json.loads(log_json)

        assert "records" in log_data
        assert "chain" in log_data
        assert len(log_data["records"]) == 1

    def test_verify_provenance(self):
        """Test provenance verification."""
        config = HybridQuantumConfig(provider="simulator", shots=50, seed=42)
        backend = IBMHybridBackend(config)
        wrapper = QuantumProvenanceWrapper(backend)

        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.measure_all()

        result, record = wrapper.execute_with_provenance(circuit)

        # Verify with same result should pass
        verified = wrapper.verify_provenance(record.record_id, result)
        assert verified is True
        
        # Record status should be updated
        updated_record = wrapper.get_record(record.record_id)
        assert updated_record.status == ProvenanceStatus.VERIFIED
