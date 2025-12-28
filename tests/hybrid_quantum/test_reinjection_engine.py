"""Tests for the Reinjection Evaluation Engine."""

import pytest
import json

try:
    from quasim.hybrid_quantum.reinjection_engine import (
        ReinjectionEvaluationEngine,
        ProposalArtifact,
        ProposalStatus,
        ProposalCluster,
        PreValidationScore,
        MerkleTreeBuilder,
        MerkleNode,
    )
    REINJECTION_AVAILABLE = True
except ImportError:
    REINJECTION_AVAILABLE = False


@pytest.mark.skipif(not REINJECTION_AVAILABLE, reason="Reinjection engine module not available")
class TestPreValidationScore:
    """Test PreValidationScore class."""

    def test_score_creation(self):
        """Test creating pre-validation score."""
        score = PreValidationScore(
            score=0.8,
            passed=True,
            checks={"check1": True, "check2": True, "check3": False},
        )
        assert score.score == 0.8
        assert score.passed is True
        assert score.checks_passed == 2
        assert score.checks_total == 3

    def test_score_timestamp(self):
        """Test timestamp is set automatically."""
        score = PreValidationScore(score=0.5)
        assert score.timestamp != ""


@pytest.mark.skipif(not REINJECTION_AVAILABLE, reason="Reinjection engine module not available")
class TestMerkleTreeBuilder:
    """Test MerkleTreeBuilder class."""

    def test_compute_hash(self):
        """Test hash computation."""
        hash1 = MerkleTreeBuilder.compute_hash("test data")
        hash2 = MerkleTreeBuilder.compute_hash("test data")
        hash3 = MerkleTreeBuilder.compute_hash("different data")
        
        assert hash1 == hash2  # Deterministic
        assert hash1 != hash3
        assert len(hash1) == 64  # SHA-256 hex

    def test_compute_merkle_root_single_item(self):
        """Test merkle root for single item."""
        root, nodes = MerkleTreeBuilder.compute_merkle_root(["item1"])
        
        assert root != ""
        assert len(nodes) == 1
        assert nodes[0].data == "item1"

    def test_compute_merkle_root_multiple_items(self):
        """Test merkle root for multiple items."""
        items = ["item1", "item2", "item3", "item4"]
        root, nodes = MerkleTreeBuilder.compute_merkle_root(items)
        
        assert root != ""
        # Should have leaf nodes + intermediate nodes
        assert len(nodes) >= 4

    def test_compute_merkle_root_empty(self):
        """Test merkle root for empty list."""
        root, nodes = MerkleTreeBuilder.compute_merkle_root([])
        
        assert root == ""
        assert len(nodes) == 0

    def test_merkle_root_deterministic(self):
        """Test merkle root computation is deterministic."""
        items = ["a", "b", "c", "d"]
        root1, _ = MerkleTreeBuilder.compute_merkle_root(items)
        root2, _ = MerkleTreeBuilder.compute_merkle_root(items)
        
        assert root1 == root2

    def test_verify_merkle_proof(self):
        """Test merkle proof verification."""
        # This is a simplified test - full proof verification
        # would require constructing the proof path
        item = "test"
        item_hash = MerkleTreeBuilder.compute_hash(item)
        
        # Single item tree - proof is empty
        result = MerkleTreeBuilder.verify_merkle_proof(item, [], item_hash)
        assert result is True

    def test_merkle_node_creation(self):
        """Test MerkleNode creation."""
        node = MerkleNode(
            hash="abc123",
            left_child="left",
            right_child="right",
        )
        assert node.hash == "abc123"
        assert node.left_child == "left"
        assert node.timestamp != ""


@pytest.mark.skipif(not REINJECTION_AVAILABLE, reason="Reinjection engine module not available")
class TestProposalArtifact:
    """Test ProposalArtifact class."""

    def test_artifact_creation(self):
        """Test creating proposal artifact."""
        artifact = ProposalArtifact(
            proposal_id="test-123",
            cluster=ProposalCluster.P0,
            metrics_target={"primary": "10x_speedup", "secondary": "zk_5s"},
        )
        assert artifact.proposal_id == "test-123"
        assert artifact.cluster == ProposalCluster.P0
        assert artifact.invariant_assertion == "ℛ(t) ≥ 0"
        assert artifact.status == ProposalStatus.PENDING

    def test_artifact_to_json(self):
        """Test JSON serialization."""
        artifact = ProposalArtifact(
            proposal_id="test-123",
            cluster=ProposalCluster.P1,
            metrics_target={"primary": "fidelity", "secondary": "variance"},
        )
        
        json_str = artifact.to_json()
        data = json.loads(json_str)
        
        assert data["proposal_id"] == "test-123"
        assert data["cluster"] == "P1"
        assert data["invariant_assertion"] == "ℛ(t) ≥ 0"

    def test_artifact_from_json(self):
        """Test JSON deserialization."""
        json_str = json.dumps({
            "proposal_id": "test-456",
            "cluster": "P2",
            "metrics_target": {"primary": "latency"},
            "status": "approved",
        })
        
        artifact = ProposalArtifact.from_json(json_str)
        
        assert artifact.proposal_id == "test-456"
        assert artifact.cluster == ProposalCluster.P2
        assert artifact.status == ProposalStatus.APPROVED


@pytest.mark.skipif(not REINJECTION_AVAILABLE, reason="Reinjection engine module not available")
class TestReinjectionEvaluationEngine:
    """Test ReinjectionEvaluationEngine class."""

    def test_engine_creation(self):
        """Test creating engine with defaults."""
        engine = ReinjectionEvaluationEngine()
        assert engine.require_dual_approval is True
        assert engine.pre_validation_threshold == 0.7
        assert engine.max_proposal_age_hours == 48

    def test_engine_custom_config(self):
        """Test creating engine with custom config."""
        engine = ReinjectionEvaluationEngine(
            require_dual_approval=False,
            pre_validation_threshold=0.5,
            max_proposal_age_hours=24,
        )
        assert engine.require_dual_approval is False
        assert engine.pre_validation_threshold == 0.5
        assert engine.max_proposal_age_hours == 24

    def test_create_proposal_success(self):
        """Test creating proposal from successful execution result."""
        engine = ReinjectionEvaluationEngine()
        
        result = {
            "success": True,
            "trust_metric": 0.95,
            "provenance_hash": "abc123",
        }
        
        proposal = engine.create_proposal(
            execution_result=result,
            cluster=ProposalCluster.P0,
            metrics_target={"primary": "speedup", "secondary": "latency"},
        )
        
        assert proposal.proposal_id != ""
        assert proposal.cluster == ProposalCluster.P0
        assert proposal.pre_validation is not None
        assert proposal.pre_validation.passed is True

    def test_create_proposal_failure(self):
        """Test creating proposal from failed execution result."""
        engine = ReinjectionEvaluationEngine()
        
        result = {
            "success": False,
            "trust_metric": 0.5,
        }
        
        proposal = engine.create_proposal(execution_result=result)
        
        assert proposal.pre_validation is not None
        assert proposal.pre_validation.passed is False
        assert "execution_success" in proposal.pre_validation.checks
        assert proposal.pre_validation.checks["execution_success"] is False

    def test_create_proposal_trust_violation(self):
        """Test proposal with trust invariant violation."""
        engine = ReinjectionEvaluationEngine()
        
        result = {
            "success": True,
            "trust_metric": -0.1,  # Violation!
        }
        
        proposal = engine.create_proposal(execution_result=result)
        
        assert proposal.pre_validation.passed is False
        assert "trust_preserved" in proposal.pre_validation.checks
        assert proposal.pre_validation.checks["trust_preserved"] is False
        assert any("Trust invariant" in e for e in proposal.pre_validation.errors)

    def test_submit_for_approval(self):
        """Test submitting proposal for approval."""
        engine = ReinjectionEvaluationEngine()
        
        result = {"success": True, "trust_metric": 0.9, "provenance_hash": "abc"}
        proposal = engine.create_proposal(execution_result=result)
        
        submission = engine.submit_for_approval(proposal.proposal_id)
        
        assert submission["success"] is True
        assert submission["requires_approvals"] == 2

    def test_submit_failed_validation(self):
        """Test submitting proposal that failed validation."""
        engine = ReinjectionEvaluationEngine()
        
        result = {"success": False}
        proposal = engine.create_proposal(execution_result=result)
        
        submission = engine.submit_for_approval(proposal.proposal_id)
        
        assert submission["success"] is False
        assert "pre-validation" in submission["error"]

    def test_approve_proposal(self):
        """Test approving a proposal."""
        engine = ReinjectionEvaluationEngine()
        
        result = {"success": True, "trust_metric": 0.9, "provenance_hash": "abc"}
        proposal = engine.create_proposal(execution_result=result)
        engine.submit_for_approval(proposal.proposal_id)
        
        # First approval
        approval1 = engine.approve(proposal.proposal_id, "approver_1", notes="LGTM")
        assert approval1["success"] is True
        assert approval1["approvals_count"] == 1
        assert approval1["is_fully_approved"] is False
        
        # Second approval
        approval2 = engine.approve(proposal.proposal_id, "approver_2")
        assert approval2["success"] is True
        assert approval2["approvals_count"] == 2
        assert approval2["is_fully_approved"] is True

    def test_approve_duplicate_rejected(self):
        """Test that duplicate approvals from same entity are rejected."""
        engine = ReinjectionEvaluationEngine()
        
        result = {"success": True, "trust_metric": 0.9, "provenance_hash": "abc"}
        proposal = engine.create_proposal(execution_result=result)
        engine.submit_for_approval(proposal.proposal_id)
        
        engine.approve(proposal.proposal_id, "approver_1")
        duplicate = engine.approve(proposal.proposal_id, "approver_1")
        
        assert duplicate["success"] is False
        assert "Already approved" in duplicate["error"]

    def test_reject_proposal(self):
        """Test rejecting a proposal."""
        engine = ReinjectionEvaluationEngine()
        
        result = {"success": True, "trust_metric": 0.9, "provenance_hash": "abc"}
        proposal = engine.create_proposal(execution_result=result)
        engine.submit_for_approval(proposal.proposal_id)
        
        rejection = engine.reject(
            proposal.proposal_id,
            "rejector_1",
            reason="Does not meet security requirements",
        )
        
        assert rejection["success"] is True
        assert rejection["status"] == "rejected"

    def test_execute_approved(self):
        """Test executing an approved proposal."""
        engine = ReinjectionEvaluationEngine()
        
        result = {"success": True, "trust_metric": 0.9, "provenance_hash": "abc"}
        proposal = engine.create_proposal(execution_result=result)
        engine.submit_for_approval(proposal.proposal_id)
        engine.approve(proposal.proposal_id, "approver_1")
        engine.approve(proposal.proposal_id, "approver_2")
        
        def execution_fn(outputs):
            return {"executed": True}
        
        execution = engine.execute_approved(proposal.proposal_id, execution_fn)
        
        assert execution["success"] is True
        assert execution["status"] == "executed"

    def test_execute_not_approved(self):
        """Test executing non-approved proposal fails."""
        engine = ReinjectionEvaluationEngine()
        
        result = {"success": True, "trust_metric": 0.9, "provenance_hash": "abc"}
        proposal = engine.create_proposal(execution_result=result)
        # Not submitted for approval
        
        execution = engine.execute_approved(proposal.proposal_id, lambda x: x)
        
        assert execution["success"] is False
        assert "not approved" in execution["error"]

    def test_rollback_proposal(self):
        """Test rolling back an executed proposal."""
        engine = ReinjectionEvaluationEngine()
        
        result = {"success": True, "trust_metric": 0.9, "provenance_hash": "abc"}
        proposal = engine.create_proposal(execution_result=result)
        engine.submit_for_approval(proposal.proposal_id)
        engine.approve(proposal.proposal_id, "approver_1")
        engine.approve(proposal.proposal_id, "approver_2")
        engine.execute_approved(proposal.proposal_id, lambda x: x)
        
        rollback = engine.rollback(proposal.proposal_id)
        
        assert rollback["success"] is True
        assert rollback["status"] == "rolled_back"

    def test_get_pending_proposals(self):
        """Test getting pending proposals."""
        engine = ReinjectionEvaluationEngine()
        
        result = {"success": True, "trust_metric": 0.9, "provenance_hash": "abc"}
        proposal = engine.create_proposal(execution_result=result)
        engine.submit_for_approval(proposal.proposal_id)
        
        pending = engine.get_pending_proposals()
        
        assert len(pending) == 1
        assert pending[0].proposal_id == proposal.proposal_id

    def test_get_proposals_by_cluster(self):
        """Test filtering proposals by cluster."""
        engine = ReinjectionEvaluationEngine()
        
        result = {"success": True, "trust_metric": 0.9, "provenance_hash": "abc"}
        engine.create_proposal(execution_result=result, cluster=ProposalCluster.P0)
        engine.create_proposal(execution_result=result, cluster=ProposalCluster.P1)
        engine.create_proposal(execution_result=result, cluster=ProposalCluster.P0)
        
        p0_proposals = engine.get_proposals_by_cluster(ProposalCluster.P0)
        
        assert len(p0_proposals) == 2

    def test_audit_report(self):
        """Test audit report generation."""
        engine = ReinjectionEvaluationEngine()
        
        result = {"success": True, "trust_metric": 0.9, "provenance_hash": "abc"}
        engine.create_proposal(execution_result=result, cluster=ProposalCluster.P0)
        engine.create_proposal(execution_result=result, cluster=ProposalCluster.P1)
        
        report = engine.generate_audit_report()
        
        assert report["invariant_assertion"] == "ℛ(t) ≥ 0"
        assert report["total_proposals"] == 2
        assert "P0" in report["proposals_by_cluster"]
        assert "P1" in report["proposals_by_cluster"]

    def test_custom_validation_hook(self):
        """Test registering custom validation hooks."""
        engine = ReinjectionEvaluationEngine()
        
        def custom_hook(result):
            if result.get("custom_field") == "valid":
                return True, "Custom check passed"
            return False, "Custom check failed"
        
        engine.register_validation_hook(custom_hook)
        
        # Without custom field
        result1 = {"success": True, "trust_metric": 0.9, "provenance_hash": "abc"}
        proposal1 = engine.create_proposal(execution_result=result1)
        assert "custom_hook_0" in proposal1.pre_validation.checks
        assert proposal1.pre_validation.checks["custom_hook_0"] is False
        
        # With custom field
        result2 = {
            "success": True, 
            "trust_metric": 0.9, 
            "provenance_hash": "abc",
            "custom_field": "valid",
        }
        proposal2 = engine.create_proposal(execution_result=result2)
        assert proposal2.pre_validation.checks["custom_hook_0"] is True


@pytest.mark.skipif(not REINJECTION_AVAILABLE, reason="Reinjection engine module not available")
class TestProposalEnums:
    """Test proposal enumeration types."""

    def test_proposal_status_values(self):
        """Test ProposalStatus enum values."""
        assert ProposalStatus.PENDING.value == "pending"
        assert ProposalStatus.PRE_VALIDATED.value == "pre_validated"
        assert ProposalStatus.AWAITING_APPROVAL.value == "awaiting_approval"
        assert ProposalStatus.APPROVED.value == "approved"
        assert ProposalStatus.REJECTED.value == "rejected"
        assert ProposalStatus.EXECUTED.value == "executed"
        assert ProposalStatus.ROLLED_BACK.value == "rolled_back"

    def test_proposal_cluster_values(self):
        """Test ProposalCluster enum values."""
        assert ProposalCluster.P0.value == "P0"
        assert ProposalCluster.P1.value == "P1"
        assert ProposalCluster.P2.value == "P2"
        assert ProposalCluster.P3.value == "P3"


@pytest.mark.skipif(not REINJECTION_AVAILABLE, reason="Reinjection engine module not available")
class TestMerkleProvenanceIntegration:
    """Test merkle provenance integration in proposals."""

    def test_proposal_has_merkle_root(self):
        """Test that proposal has merkle root in inputs."""
        engine = ReinjectionEvaluationEngine()
        
        result = {"success": True, "trust_metric": 0.9, "provenance_hash": "abc"}
        proposal = engine.create_proposal(execution_result=result)
        
        assert len(proposal.inputs) == 1
        assert proposal.inputs[0] != ""  # Merkle root

    def test_proposal_rollback_path(self):
        """Test that proposal has rollback path."""
        engine = ReinjectionEvaluationEngine()
        
        result = {"success": True, "trust_metric": 0.9, "provenance_hash": "abc"}
        proposal = engine.create_proposal(execution_result=result)
        
        assert proposal.rollback_path != ""
        # Rollback path should be merkle root
        assert proposal.rollback_path == proposal.inputs[0]

    def test_custom_input_data_merkle(self):
        """Test merkle root from custom input data."""
        engine = ReinjectionEvaluationEngine()
        
        result = {"success": True, "trust_metric": 0.9}
        input_data = ["data1", "data2", "data3"]
        
        proposal = engine.create_proposal(
            execution_result=result,
            input_data=input_data,
        )
        
        # Should have merkle root of input data
        assert len(proposal.inputs) == 1
        
        # Verify it matches independently computed root
        expected_root, _ = MerkleTreeBuilder.compute_merkle_root(input_data)
        assert proposal.inputs[0] == expected_root
