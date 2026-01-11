"""Tests for the XENONInterventionSearch module."""

import pytest

from qratum.oncology.intervention_search import (
    XENONInterventionSearch,
    InterventionNode,
    TreatmentSequence,
    AdaptiveTherapyPlan,
    DrugProfile,
    InterventionType,
    ResistanceMechanism,
    create_example_drug_library,
)


class TestDrugProfile:
    """Tests for DrugProfile dataclass."""

    def test_creation(self):
        """Test DrugProfile creation."""
        drug = DrugProfile(
            drug_id="osimertinib",
            name="Osimertinib",
            target_genes=["EGFR"],
            mechanism="Third-generation EGFR TKI",
            intervention_type=InterventionType.TARGETED_THERAPY,
            efficacy_score=0.8,
            toxicity_score=0.25,
        )
        assert drug.drug_id == "osimertinib"
        assert drug.efficacy_score == 0.8
        assert "EGFR" in drug.target_genes

    def test_to_dict(self):
        """Test serialization."""
        drug = DrugProfile(
            drug_id="test",
            name="Test Drug",
            intervention_type=InterventionType.IMMUNOTHERAPY,
        )
        d = drug.to_dict()
        assert d["intervention_type"] == "immunotherapy"


class TestInterventionNode:
    """Tests for InterventionNode dataclass."""

    def test_creation(self):
        """Test InterventionNode creation."""
        node = InterventionNode(
            node_id="N001",
            tumor_state={"tumor_burden": 0.5},
            resistance_probability=0.1,
            toxicity_accumulated=0.2,
        )
        assert node.tumor_state["tumor_burden"] == 0.5
        assert node.resistance_probability == 0.1

    def test_to_dict(self):
        """Test serialization."""
        node = InterventionNode(node_id="test", depth=3)
        d = node.to_dict()
        assert d["depth"] == 3


class TestTreatmentSequence:
    """Tests for TreatmentSequence dataclass."""

    def test_creation(self):
        """Test TreatmentSequence creation."""
        sequence = TreatmentSequence(
            sequence_id="S001",
            total_efficacy=0.7,
            total_toxicity=0.3,
            resistance_suppression_score=0.6,
        )
        assert sequence.total_efficacy == 0.7

    def test_to_dict(self):
        """Test serialization."""
        sequence = TreatmentSequence(
            sequence_id="test",
            rationale="Test rationale",
        )
        d = sequence.to_dict()
        assert d["rationale"] == "Test rationale"


class TestXENONInterventionSearch:
    """Tests for XENONInterventionSearch class."""

    def test_initialization(self):
        """Test search engine initialization."""
        search = XENONInterventionSearch(seed=42, max_depth=6)
        assert search.seed == 42
        assert search.max_depth == 6

    def test_add_drug(self):
        """Test adding drugs to library."""
        search = XENONInterventionSearch()
        drug = DrugProfile(
            drug_id="test_drug",
            name="Test Drug",
            intervention_type=InterventionType.TARGETED_THERAPY,
        )
        search.add_drug(drug)
        assert search.get_drug("test_drug") is not None

    def test_initialize_search(self):
        """Test search initialization."""
        search = XENONInterventionSearch()
        initial_state = {"tumor_burden": 1.0, "EGFR_activity": 0.9}
        root_id = search.initialize_search(initial_state)
        assert root_id is not None

    def test_expand_node(self):
        """Test node expansion."""
        search = XENONInterventionSearch(max_depth=3)

        # Add drugs
        drug1 = DrugProfile(
            drug_id="drug1",
            name="Drug 1",
            intervention_type=InterventionType.TARGETED_THERAPY,
            efficacy_score=0.7,
            toxicity_score=0.2,
        )
        drug2 = DrugProfile(
            drug_id="drug2",
            name="Drug 2",
            intervention_type=InterventionType.IMMUNOTHERAPY,
            efficacy_score=0.5,
            toxicity_score=0.15,
        )
        search.add_drug(drug1)
        search.add_drug(drug2)

        # Initialize and expand
        initial_state = {"tumor_burden": 1.0}
        root_id = search.initialize_search(initial_state)
        child_ids = search.expand_node(root_id, dosage_levels=[0.5, 1.0])

        assert len(child_ids) > 0

    def test_search_best_sequences(self):
        """Test best sequence search."""
        search = XENONInterventionSearch(seed=42, max_depth=3)

        # Add drugs
        drugs = create_example_drug_library()
        for drug in drugs.values():
            search.add_drug(drug)

        # Search
        initial_state = {"tumor_burden": 1.0, "EGFR_activity": 0.9}
        sequences = search.search_best_sequences(
            initial_tumor_state=initial_state,
            n_sequences=5,
            beam_width=10,
        )

        assert len(sequences) > 0
        assert all(isinstance(s, TreatmentSequence) for s in sequences)

    def test_create_adaptive_plan(self):
        """Test adaptive therapy plan creation."""
        search = XENONInterventionSearch(seed=42, max_depth=3)

        # Add drugs
        drugs = create_example_drug_library()
        for drug in drugs.values():
            search.add_drug(drug)

        # Get a sequence
        initial_state = {"tumor_burden": 1.0}
        sequences = search.search_best_sequences(
            initial_tumor_state=initial_state,
            n_sequences=1,
        )

        # Create adaptive plan
        if sequences:
            plan = search.create_adaptive_plan(sequences[0])
            assert isinstance(plan, AdaptiveTherapyPlan)
            assert plan.initial_sequence is not None
            assert len(plan.decision_rules) > 0

    def test_toxicity_constraint(self):
        """Test that toxicity constraint is respected."""
        search = XENONInterventionSearch(max_toxicity=0.5)

        # Add high-toxicity drug
        toxic_drug = DrugProfile(
            drug_id="toxic",
            name="Toxic Drug",
            intervention_type=InterventionType.CHEMOTHERAPY,
            toxicity_score=0.8,
        )
        search.add_drug(toxic_drug)

        # Initialize and try to expand
        root_id = search.initialize_search({"tumor_burden": 1.0})
        children = search.expand_node(root_id, dosage_levels=[1.0])

        # High toxicity nodes should be filtered
        # At 100% dose, toxicity would be 0.8 > 0.5 limit
        assert len(children) == 0

    def test_to_dict(self):
        """Test serialization."""
        search = XENONInterventionSearch(seed=42)
        drug = DrugProfile(drug_id="test", name="Test", intervention_type=InterventionType.TARGETED_THERAPY)
        search.add_drug(drug)

        d = search.to_dict()
        assert "drug_library" in d
        assert "disclaimer" in d


class TestExampleDrugLibrary:
    """Tests for example drug library."""

    def test_create_library(self):
        """Test library creation."""
        library = create_example_drug_library()
        assert len(library) > 0

    def test_library_has_expected_drugs(self):
        """Test library contains expected drugs."""
        library = create_example_drug_library()
        assert "osimertinib" in library
        assert "pembrolizumab" in library

    def test_drug_properties(self):
        """Test drug properties are reasonable."""
        library = create_example_drug_library()

        for drug in library.values():
            assert 0.0 <= drug.efficacy_score <= 1.0
            assert 0.0 <= drug.toxicity_score <= 1.0
            assert drug.intervention_type in InterventionType
