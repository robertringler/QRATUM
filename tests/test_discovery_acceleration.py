"""Tests for Discovery Acceleration Module.

Tests reproducibility, provenance, and correctness of all 6 discovery workflows.
"""


import pytest

from qradle.core.zones import SecurityZone
from qratum_asi.discovery_acceleration.contracts import (
    DISCOVERY_CONTRACT_TEMPLATES,
    ContractStatus,
    CrossVerticalIntent,
    DiscoveryContract,
    IntentType,
    VerticalBinding,
    create_discovery_contract,
)
from qratum_asi.discovery_acceleration.federated_gwas import (
    FederatedGWASPipeline,
    GWASCohort,
)
from qratum_asi.discovery_acceleration.workflows import (
    DiscoveryAccelerationEngine,
    DiscoveryType,
    WorkflowStage,
)


class TestDiscoveryAccelerationEngine:
    """Tests for DiscoveryAccelerationEngine."""

    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        engine = DiscoveryAccelerationEngine()

        assert len(engine.workflows) == 0
        assert len(engine.results) == 0
        assert engine.merkle_chain.verify_integrity()

    def test_create_workflow(self):
        """Test workflow creation."""
        engine = DiscoveryAccelerationEngine()

        workflow = engine.create_workflow(
            discovery_type=DiscoveryType.COMPLEX_DISEASE_GENETICS,
            parameters={"phenotype": "type_2_diabetes"},
            actor_id="researcher_001",
        )

        assert workflow.workflow_id.startswith("wf_complex_disease_genetics_")
        assert workflow.discovery_type == DiscoveryType.COMPLEX_DISEASE_GENETICS
        assert workflow.state["parameters"]["phenotype"] == "type_2_diabetes"
        assert len(engine.workflows) == 1

    def test_workflow_rollback_points(self):
        """Test rollback point creation and restoration."""
        engine = DiscoveryAccelerationEngine()

        workflow = engine.create_workflow(
            discovery_type=DiscoveryType.ANTI_AGING_PATHWAYS,
            parameters={"pathway": "telomere"},
            actor_id="researcher_001",
        )

        # Modify state
        workflow.state["processing_result"] = {"score": 0.85}

        # Create rollback point
        rp = workflow.create_rollback_point("Before hypothesis testing")
        assert rp.rollback_id.startswith("rb_")
        assert len(workflow.rollback_points) == 1

        # Modify state further
        workflow.state["processing_result"] = {"score": 0.45}

        # Rollback
        success = workflow.rollback_to(rp.rollback_id)
        assert success
        assert workflow.state["processing_result"]["score"] == 0.85

    def test_projections_all_discovery_types(self):
        """Test projections are available for all discovery types."""
        engine = DiscoveryAccelerationEngine()

        for dt in DiscoveryType:
            projections = engine.get_discovery_projections(dt)

            assert "discovery_probability" in projections
            assert "time_savings_factor" in projections
            assert "risk_mitigation_score" in projections
            assert projections["time_savings_factor"] >= 1.0
            assert 0 <= projections["discovery_probability"] <= 1.0

    def test_compliance_mapping_all_discovery_types(self):
        """Test compliance mapping for all discovery types."""
        engine = DiscoveryAccelerationEngine()

        for dt in DiscoveryType:
            compliance = engine.get_compliance_mapping(dt)

            # All should have GDPR, HIPAA, ISO 27001
            assert "gdpr" in compliance
            assert "hipaa" in compliance
            assert "iso_27001" in compliance

            # Each should be compliant
            assert compliance["gdpr"]["status"] == "compliant"

    def test_engine_stats(self):
        """Test engine statistics."""
        engine = DiscoveryAccelerationEngine()

        # Create multiple workflows
        for dt in [
            DiscoveryType.COMPLEX_DISEASE_GENETICS,
            DiscoveryType.PERSONALIZED_DRUG_DESIGN,
            DiscoveryType.COMPLEX_DISEASE_GENETICS,
        ]:
            engine.create_workflow(
                discovery_type=dt,
                parameters={},
                actor_id="test",
            )

        stats = engine.get_engine_stats()

        assert stats["total_workflows"] == 3
        assert stats["workflows_by_type"]["complex_disease_genetics"] == 2
        assert stats["workflows_by_type"]["personalized_drug_design"] == 1
        assert stats["merkle_chain_valid"]


class TestFederatedGWASPipeline:
    """Tests for FederatedGWASPipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly."""
        pipeline = FederatedGWASPipeline()

        assert pipeline.pipeline_id.startswith("gwas_")
        assert len(pipeline.cohorts) == 0
        assert pipeline.verify_provenance()

    def test_cohort_registration(self):
        """Test cohort registration with dual-control."""
        pipeline = FederatedGWASPipeline()

        cohort = pipeline.register_cohort(
            cohort_id="cohort_001",
            site_name="Site A",
            sample_count=5000,
            phenotype="type_2_diabetes",
            ancestry="EUR",
            biokey="test_biokey_123",
            actor_id="researcher_001",
            approver_id="supervisor_001",
        )

        assert cohort.cohort_id == "cohort_001"
        assert cohort.is_registered
        assert cohort.sample_count == 5000
        assert "cohort_001" in pipeline.cohorts

    def test_variant_proof_generation(self):
        """Test ZK proof generation for variants."""
        pipeline = FederatedGWASPipeline()

        # Register cohort first
        pipeline.register_cohort(
            cohort_id="cohort_001",
            site_name="Site A",
            sample_count=5000,
            phenotype="type_2_diabetes",
            ancestry="EUR",
            biokey="test_biokey",
            actor_id="researcher",
            approver_id="supervisor",
        )

        # Generate variant proof
        proof = pipeline.generate_variant_proof(
            cohort_id="cohort_001",
            variant_id="chr6:32500000:A:G",
            statistics={"allele_frequency": 0.15, "p_value": 1e-10},
            actor_id="researcher",
        )

        assert proof.variant_id == "chr6:32500000:A:G"
        assert proof.cohort_id == "cohort_001"
        assert len(proof.commitment) == 64  # SHA3-256 hex

    def test_variant_aggregation(self):
        """Test variant aggregation across cohorts."""
        pipeline = FederatedGWASPipeline()

        # Register multiple cohorts
        for i in range(3):
            pipeline.register_cohort(
                cohort_id=f"cohort_{i:03d}",
                site_name=f"Site {chr(65+i)}",
                sample_count=5000 + i * 1000,
                phenotype="type_2_diabetes",
                ancestry="EUR",
                biokey=f"biokey_{i}",
                actor_id="researcher",
                approver_id="supervisor",
            )

            # Generate proofs
            pipeline.generate_variant_proof(
                cohort_id=f"cohort_{i:03d}",
                variant_id="chr6:32500000:A:G",
                statistics={"allele_frequency": 0.15 + i * 0.01},
                actor_id="researcher",
            )

        # Aggregate
        result = pipeline.aggregate_variant_statistics(
            variant_id="chr6:32500000:A:G",
            actor_id="researcher",
            approver_id="supervisor",
        )

        assert result["num_cohorts"] == 3
        assert result["total_samples"] == 18000  # 5000 + 6000 + 7000

    def test_full_gwas_analysis(self):
        """Test full GWAS analysis workflow."""
        pipeline = FederatedGWASPipeline()

        # Register cohorts
        for i in range(2):
            pipeline.register_cohort(
                cohort_id=f"cohort_{i:03d}",
                site_name=f"Site {chr(65+i)}",
                sample_count=10000,
                phenotype="type_2_diabetes",
                ancestry="EUR",
                biokey=f"biokey_{i}",
                actor_id="researcher",
                approver_id="supervisor",
            )

        # Run analysis
        result = pipeline.run_association_analysis(
            phenotype="type_2_diabetes",
            significance_threshold=5e-8,
            actor_id="researcher",
            approver_id="supervisor",
        )

        assert result.phenotype == "type_2_diabetes"
        assert result.total_cohorts == 2
        assert result.total_samples == 20000
        assert len(result.significant_variants) > 0
        assert len(result.provenance_chain) == 64  # SHA256 hex
        assert result.projections["time_savings_factor"] == 10.0

    def test_provenance_integrity(self):
        """Test provenance chain integrity."""
        pipeline = FederatedGWASPipeline()

        # Perform various operations
        pipeline.register_cohort(
            cohort_id="cohort_001",
            site_name="Site A",
            sample_count=5000,
            phenotype="test",
            ancestry="EUR",
            biokey="biokey",
            actor_id="researcher",
            approver_id="supervisor",
        )

        # Verify provenance
        assert pipeline.verify_provenance()

        stats = pipeline.get_pipeline_stats()
        assert stats["merkle_chain_valid"]


class TestDiscoveryContracts:
    """Tests for DiscoveryContract and CrossVerticalIntent."""

    def test_vertical_binding_hash(self):
        """Test vertical binding hash is deterministic."""
        binding1 = VerticalBinding(
            vertical_name="VITRA",
            operation="sequence_analysis",
            parameters={"sequence_type": "dna"},
        )

        binding2 = VerticalBinding(
            vertical_name="VITRA",
            operation="sequence_analysis",
            parameters={"sequence_type": "dna"},
        )

        assert binding1.compute_hash() == binding2.compute_hash()

    def test_cross_vertical_intent_creation(self):
        """Test cross-vertical intent creation."""
        bindings = [
            VerticalBinding("VITRA", "sequence_analysis", {}),
            VerticalBinding("ECORA", "climate_projection", {}),
        ]

        intent = CrossVerticalIntent(
            intent_id="intent_001",
            intent_type=IntentType.SYNTHESIS,
            discovery_type=DiscoveryType.CLIMATE_GENE_CONNECTIONS,
            source_verticals=bindings,
            target_vertical="VITRA",
            synthesis_goal="Test synthesis",
            required_authorizers=["auth_001", "auth_002"],
        )

        assert intent.intent_id == "intent_001"
        assert len(intent.source_verticals) == 2
        assert intent.status == ContractStatus.DRAFT

    def test_discovery_contract_lifecycle(self):
        """Test full contract lifecycle."""
        contract = DiscoveryContract(
            contract_id="dc_test_001",
            discovery_type=DiscoveryType.PERSONALIZED_DRUG_DESIGN,
        )

        # Add intent
        intent = CrossVerticalIntent(
            intent_id="intent_001",
            intent_type=IntentType.SYNTHESIS,
            discovery_type=DiscoveryType.PERSONALIZED_DRUG_DESIGN,
            source_verticals=[
                VerticalBinding("VITRA", "sequence_analysis", {}),
            ],
            target_vertical="VITRA",
            synthesis_goal="Test",
            required_authorizers=["auth_001"],
        )
        contract.add_intent(intent)

        # Submit
        assert contract.submit()
        assert contract.status == ContractStatus.SUBMITTED

        # Authorize
        contract.authorize(
            authorizer_id="auth_001",
            authorization_type="full",
            scope=["all"],
        )
        assert contract.status == ContractStatus.AUTHORIZED

        # Execute
        result = contract.execute()
        assert result["success"]
        assert contract.status == ContractStatus.COMPLETED
        assert contract.verify_provenance()

    def test_contract_hypothesis_generation(self):
        """Test hypothesis generation in contracts."""
        contract = DiscoveryContract(
            contract_id="dc_test_002",
            discovery_type=DiscoveryType.ANTI_AGING_PATHWAYS,
        )

        hypothesis = contract.add_hypothesis(
            hypothesis_id="hyp_001",
            description="Telomere extension may slow cellular aging",
            confidence=0.82,
            supporting_evidence=["Study A", "Study B"],
            domains=["genomics", "cell_biology"],
        )

        assert hypothesis["confidence"] == 0.82
        assert len(contract.hypotheses) == 1

    def test_create_discovery_contract_factory(self):
        """Test factory function for contract creation."""
        contract = create_discovery_contract(
            discovery_type=DiscoveryType.NATURAL_DRUG_DISCOVERY,
            synthesis_goal="Discover new antibiotics from soil microbiome",
            verticals=[
                ("VITRA", "metagenomics", {"source": "soil"}),
                ("VITRA", "drug_screening", {"target": "bacteria"}),
            ],
            target_vertical="VITRA",
            required_authorizers=["auth_001"],
        )

        assert contract.discovery_type == DiscoveryType.NATURAL_DRUG_DISCOVERY
        assert len(contract.intents) == 1
        assert len(contract.intents[0].source_verticals) == 2

    def test_discovery_contract_templates(self):
        """Test all discovery contract templates exist."""
        for dt in DiscoveryType:
            assert dt in DISCOVERY_CONTRACT_TEMPLATES
            template = DISCOVERY_CONTRACT_TEMPLATES[dt]
            assert "verticals" in template
            assert "target_vertical" in template
            assert "synthesis_goal" in template


class TestInvariantPreservation:
    """Tests for invariant preservation across all workflows."""

    def test_deterministic_hashes(self):
        """Test that hashes are deterministic."""
        cohort1 = GWASCohort(
            cohort_id="test",
            site_name="Site",
            sample_count=1000,
            phenotype="diabetes",
            ancestry="EUR",
            biokey_hash="hash",
        )

        cohort2 = GWASCohort(
            cohort_id="test",
            site_name="Site",
            sample_count=1000,
            phenotype="diabetes",
            ancestry="EUR",
            biokey_hash="hash",
        )

        assert cohort1.compute_commitment() == cohort2.compute_commitment()

    def test_merkle_chain_integrity(self):
        """Test Merkle chain integrity is preserved."""
        pipeline = FederatedGWASPipeline()

        # Perform operations
        pipeline.register_cohort(
            cohort_id="test",
            site_name="Site",
            sample_count=1000,
            phenotype="test",
            ancestry="EUR",
            biokey="key",
            actor_id="actor",
            approver_id="approver",
        )

        # Chain should be valid
        assert pipeline.verify_provenance()

        # Chain length should increase
        assert len(pipeline.merkle_chain.chain) > 1

    def test_zone_enforcement(self):
        """Test security zone enforcement."""
        engine = DiscoveryAccelerationEngine()

        workflow = engine.create_workflow(
            discovery_type=DiscoveryType.COMPLEX_DISEASE_GENETICS,
            parameters={},
            actor_id="researcher",
        )

        # Execute stage should use zone enforcer
        artifact = workflow.execute_stage(
            stage=WorkflowStage.INPUT_VALIDATION,
            zone=SecurityZone.Z1,
            operation=lambda: {"validated": True},
            actor_id="researcher",
        )

        assert artifact.metadata["zone"] == "Z1"
        assert artifact.metadata["actor"] == "researcher"


class TestReproducibility:
    """Tests for reproducibility of discovery workflows."""

    def test_gwas_result_reproducibility(self):
        """Test GWAS results are reproducible."""

        def run_analysis():
            pipeline = FederatedGWASPipeline(pipeline_id="repro_test")

            pipeline.register_cohort(
                cohort_id="cohort_001",
                site_name="Site A",
                sample_count=10000,
                phenotype="test_phenotype",
                ancestry="EUR",
                biokey="biokey_123",
                actor_id="researcher",
                approver_id="supervisor",
            )

            return pipeline.run_association_analysis(
                phenotype="test_phenotype",
                actor_id="researcher",
                approver_id="supervisor",
            )

        result1 = run_analysis()
        result2 = run_analysis()

        # Results should be identical (same manhattan hash)
        assert result1.manhattan_hash == result2.manhattan_hash
        assert result1.significant_variants == result2.significant_variants

    def test_contract_hash_reproducibility(self):
        """Test contract hashes are reproducible."""

        def create_contract():
            return create_discovery_contract(
                discovery_type=DiscoveryType.CLIMATE_GENE_CONNECTIONS,
                synthesis_goal="Test goal",
                verticals=[("VITRA", "test_op", {"param": "value"})],
                target_vertical="VITRA",
                required_authorizers=["auth"],
            )

        contract1 = create_contract()
        contract2 = create_contract()

        # Intent hashes should be identical
        assert (
            contract1.intents[0].compute_intent_hash() == contract2.intents[0].compute_intent_hash()
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
