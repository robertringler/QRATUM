"""Unit tests for QIL parser.

Tests the QIL parser, grammar, AST, and serialization.
"""

import pytest

from qil import (
    ParseError,
    compute_hash,
    intent_to_canonical_form,
    parse_intent,
    serialize_intent,
)
from qil.ast import HardwareSpec, TimeSpec


class TestQILParser:
    """Test suite for QIL parser."""

    def test_parse_simple_intent(self):
        """Test parsing a simple intent."""
        qil_text = """
        INTENT test_intent {
            OBJECTIVE test_objective
            AUTHORITY user: alice
        }
        """
        intent = parse_intent(qil_text)

        assert intent.name == "test_intent"
        assert intent.objective.name == "test_objective"
        assert len(intent.authorities) == 1
        assert intent.authorities[0].key == "user"
        assert intent.authorities[0].value == "alice"

    def test_parse_intent_with_hardware(self):
        """Test parsing intent with hardware specification."""
        qil_text = """
        INTENT gpu_intent {
            OBJECTIVE train_model
            HARDWARE ONLY GB200 AND MI300X
            AUTHORITY user: bob
        }
        """
        intent = parse_intent(qil_text)

        assert intent.name == "gpu_intent"
        assert intent.hardware is not None
        assert "GB200" in intent.hardware.only_clusters
        assert "MI300X" in intent.hardware.only_clusters

    def test_parse_intent_with_constraints(self):
        """Test parsing intent with constraints."""
        qil_text = """
        INTENT constrained_intent {
            OBJECTIVE optimize
            CONSTRAINT GPU_VRAM >= 1000
            CONSTRAINT GPU_COUNT == 8
            AUTHORITY user: charlie
        }
        """
        intent = parse_intent(qil_text)

        assert len(intent.constraints) == 2
        assert intent.constraints[0].name == "GPU_VRAM"
        assert intent.constraints[0].operator == ">="
        assert intent.constraints[0].value == 1000

    def test_parse_intent_with_time_specs(self):
        """Test parsing intent with time specifications."""
        qil_text = """
        INTENT timed_intent {
            OBJECTIVE process_data
            TIME deadline: 3600s
            TIME budget: 1800s
            AUTHORITY user: dave
        }
        """
        intent = parse_intent(qil_text)

        assert len(intent.time_specs) == 2
        deadline = next(t for t in intent.time_specs if t.key == "deadline")
        assert deadline.value == 3600
        assert deadline.unit == "s"
        assert deadline.to_seconds() == 3600.0

    def test_parse_intent_with_trust(self):
        """Test parsing intent with trust level."""
        qil_text = """
        INTENT trusted_intent {
            OBJECTIVE secure_compute
            TRUST level: verified
            AUTHORITY user: eve
        }
        """
        intent = parse_intent(qil_text)

        assert intent.trust is not None
        assert intent.trust.level == "verified"

    def test_parse_intent_not_clause(self):
        """Test parsing hardware with NOT clause."""
        qil_text = """
        INTENT excluded_hardware {
            OBJECTIVE compute
            HARDWARE NOT IPU AND GAUDI3
            AUTHORITY user: frank
        }
        """
        intent = parse_intent(qil_text)

        assert intent.hardware is not None
        assert "IPU" in intent.hardware.not_clusters
        assert "GAUDI3" in intent.hardware.not_clusters

    def test_parse_full_intent(self):
        """Test parsing a full-featured intent."""
        qil_text = """
        INTENT hybrid_ai_quantum {
            OBJECTIVE optimize_molecular_design
            HARDWARE ONLY GB200 AND QPU NOT IPU
            CONSTRAINT GPU_VRAM >= 500
            CONSTRAINT QUBITS >= 50
            CAPABILITY llm_training
            CAPABILITY quantum_optimizer
            TIME deadline: 10800s
            TIME budget: 9000s
            AUTHORITY user: research_lead
            TRUST level: verified
        }
        """
        intent = parse_intent(qil_text)

        assert intent.name == "hybrid_ai_quantum"
        assert intent.objective.name == "optimize_molecular_design"
        assert len(intent.constraints) == 2
        assert len(intent.capabilities) == 2
        assert len(intent.time_specs) == 2
        assert intent.trust.level == "verified"
        assert "GB200" in intent.hardware.only_clusters
        assert "QPU" in intent.hardware.only_clusters
        assert "IPU" in intent.hardware.not_clusters

    def test_parse_error_missing_objective(self):
        """Test parse error for missing objective."""
        qil_text = """
        INTENT no_objective {
            AUTHORITY user: test
        }
        """
        with pytest.raises(ParseError, match="Intent must have an OBJECTIVE"):
            parse_intent(qil_text)

    def test_parse_error_invalid_trust_level(self):
        """Test parse error for invalid trust level."""
        qil_text = """
        INTENT bad_trust {
            OBJECTIVE test
            TRUST level: invalid_level
            AUTHORITY user: test
        }
        """
        with pytest.raises(ParseError, match="Invalid trust level"):
            parse_intent(qil_text)

    def test_parse_error_invalid_cluster_type(self):
        """Test parse error for invalid cluster type."""
        qil_text = """
        INTENT bad_cluster {
            OBJECTIVE test
            HARDWARE ONLY INVALID_CLUSTER
            AUTHORITY user: test
        }
        """
        # Should fail to parse cluster type
        with pytest.raises(ParseError):
            parse_intent(qil_text)

    def test_serialize_intent(self):
        """Test intent serialization."""
        qil_text = """
        INTENT serialize_test {
            OBJECTIVE test_objective
            CONSTRAINT MEMORY >= 100
            CAPABILITY test_cap
            TIME deadline: 1800s
            AUTHORITY user: test_user
            TRUST level: trusted
        }
        """
        intent = parse_intent(qil_text)
        serialized = serialize_intent(intent)

        assert serialized["name"] == "serialize_test"
        assert serialized["objective"] == "test_objective"
        assert len(serialized["constraints"]) == 1
        assert len(serialized["capabilities"]) == 1
        assert serialized["trust"] == "trusted"

    def test_intent_hash_deterministic(self):
        """Test that intent hashing is deterministic."""
        qil_text = """
        INTENT hash_test {
            OBJECTIVE test
            AUTHORITY user: test
        }
        """
        intent1 = parse_intent(qil_text)
        intent2 = parse_intent(qil_text)

        hash1 = compute_hash(intent1)
        hash2 = compute_hash(intent2)

        assert hash1 == hash2

    def test_canonical_form_generation(self):
        """Test canonical QIL form generation."""
        qil_text = """
        INTENT canonical_test {
            OBJECTIVE test_objective
            HARDWARE ONLY GB200
            CONSTRAINT VRAM >= 100
            CAPABILITY test_capability
            TIME deadline: 3600s
            AUTHORITY user: alice
            TRUST level: verified
        }
        """
        intent = parse_intent(qil_text)
        canonical = intent_to_canonical_form(intent)

        assert "INTENT canonical_test {" in canonical
        assert "OBJECTIVE test_objective" in canonical
        assert "HARDWARE ONLY GB200" in canonical
        assert "CONSTRAINT VRAM >= 100" in canonical
        assert "CAPABILITY test_capability" in canonical
        assert "TIME deadline: 3600s" in canonical
        assert "AUTHORITY user: alice" in canonical
        assert "TRUST level: verified" in canonical

    def test_intent_requires_cluster(self):
        """Test intent cluster requirement checking."""
        qil_text = """
        INTENT cluster_req {
            OBJECTIVE test
            HARDWARE ONLY GB200 AND QPU
            AUTHORITY user: test
        }
        """
        intent = parse_intent(qil_text)

        assert intent.requires_cluster("GB200")
        assert intent.requires_cluster("QPU")
        assert not intent.requires_cluster("CPU")

    def test_intent_is_cluster_excluded(self):
        """Test intent cluster exclusion checking."""
        qil_text = """
        INTENT cluster_excl {
            OBJECTIVE test
            HARDWARE NOT IPU
            AUTHORITY user: test
        }
        """
        intent = parse_intent(qil_text)

        assert intent.is_cluster_excluded("IPU")
        assert not intent.is_cluster_excluded("GB200")

    def test_hardware_spec_conflict(self):
        """Test hardware spec conflict detection."""
        with pytest.raises(ValueError, match="conflict"):
            HardwareSpec(only_clusters=["GB200"], not_clusters=["GB200"])  # Conflict!

    def test_time_spec_conversion(self):
        """Test time spec unit conversion."""
        # Test seconds
        ts_s = TimeSpec(key="deadline", value=60, unit="s")
        assert ts_s.to_seconds() == 60.0

        # Test milliseconds
        ts_ms = TimeSpec(key="deadline", value=1000, unit="ms")
        assert ts_ms.to_seconds() == 1.0

        # Test minutes
        ts_m = TimeSpec(key="deadline", value=5, unit="m")
        assert ts_m.to_seconds() == 300.0

        # Test hours
        ts_h = TimeSpec(key="deadline", value=2, unit="h")
        assert ts_h.to_seconds() == 7200.0
