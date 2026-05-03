"""Tests for VITRA Bioinformatics module."""

import pytest

from qratum_platform.core import (
    ComputeSubstrate,
    PlatformContract,
    PlatformIntent,
    SafetyViolation,
    VerticalModule,
)
from verticals.vitra import VITRAModule


class TestVITRAModule:
    """Test VITRA Bioinformatics module."""

    def test_module_metadata(self):
        """Test module has required metadata."""
        module = VITRAModule()
        assert module.MODULE_NAME == "VITRA"
        assert module.MODULE_VERSION == "2.0.0"
        assert module.SAFETY_DISCLAIMER
        assert len(module.PROHIBITED_USES) > 0

    def test_dna_sequence_analysis(self):
        """Test DNA sequence analysis."""
        module = VITRAModule()
        intent = PlatformIntent(
            vertical=VerticalModule.VITRA,
            operation="sequence_analysis",
            parameters={"sequence": "ATGGCTAGCTAG", "sequence_type": "dna"},
            user_id="user_001",
        )
        contract = PlatformContract(
            intent=intent,
            contract_id="test_001",
            substrate=ComputeSubstrate.MI300X,
            estimated_cost=0,
            estimated_duration=0,
            safety_checks_passed=True,
        )

        result = module.execute(contract)
        assert result["analysis_type"] == "dna_sequence"
        assert "gc_content" in result
        assert "nucleotide_composition" in result

    def test_protein_structure_prediction(self):
        """Test protein structure prediction."""
        module = VITRAModule()
        intent = PlatformIntent(
            vertical=VerticalModule.VITRA,
            operation="protein_structure",
            parameters={"sequence": "MKTAYIAKQRQISFVK"},
            user_id="user_001",
        )
        contract = PlatformContract(
            intent=intent,
            contract_id="test_002",
            substrate=ComputeSubstrate.CEREBRAS,
            estimated_cost=0,
            estimated_duration=0,
            safety_checks_passed=True,
        )

        result = module.execute(contract)
        assert result["prediction_method"] == "simplified_homology"
        assert "secondary_structure" in result

    def test_drug_screening(self):
        """Test drug candidate screening."""
        module = VITRAModule()
        intent = PlatformIntent(
            vertical=VerticalModule.VITRA,
            operation="drug_screening",
            parameters={"candidates": ["COMPOUND_A", "COMPOUND_B"], "target": "PROTEIN_X"},
            user_id="user_001",
        )
        contract = PlatformContract(
            intent=intent,
            contract_id="test_003",
            substrate=ComputeSubstrate.GB200,
            estimated_cost=0,
            estimated_duration=0,
            safety_checks_passed=True,
        )

        result = module.execute(contract)
        assert result["screening_type"] == "virtual_screening"
        assert len(result["top_candidates"]) > 0

    def test_codon_translation(self):
        """Test codon translation."""
        module = VITRAModule()
        assert module.CODON_TABLE["AUG"] == "M"  # Start codon
        assert module.CODON_TABLE["UAA"] == "*"  # Stop codon

    def test_prohibited_use_detection(self):
        """Test that prohibited uses are detected."""
        module = VITRAModule()
        intent = PlatformIntent(
            vertical=VerticalModule.VITRA,
            operation="bioweapon development",
            parameters={},
            user_id="user_001",
        )

        with pytest.raises(SafetyViolation):
            module.check_safety(intent)
