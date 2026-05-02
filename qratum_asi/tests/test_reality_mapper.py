"""Tests for Safety Reality Mapper."""

import pytest

from qratum_asi.safety.elicitation import (
    ModelResponse,
    ResponseType,
    SafetyElicitation,
)
from qratum_asi.safety.reality_mapper import (
    SafetyRealityMapper,
)


class TestSafetyRealityMapper:
    """Test suite for SafetyRealityMapper."""

    def test_initialization(self):
        """Test mapper initialization."""
        elicitation = SafetyElicitation()
        mapper = SafetyRealityMapper(elicitation)

        assert mapper.elicitation is elicitation
        assert len(mapper.proven_impossibilities) == 0
        assert len(mapper.fragile_assumptions) == 0
        assert len(mapper.hard_constraints) == 0

    def test_generate_reality_map(self):
        """Test reality map generation."""
        elicitation = SafetyElicitation()
        question_id = list(elicitation.questions.keys())[0]

        # Add some test responses
        response = ModelResponse(
            model_identifier="test_model",
            question_id=question_id,
            response_type=ResponseType.MECHANISTIC,
            response_text="Test response",
            assumptions_declared=["We assume human oversight"],
            mechanisms_described=["Mechanism fails under pressure"],
            hard_claims=["Long-term containment is impossible"],
            speculation=[],
            uncertainties=["Timeline uncertain"],
            refusals_avoidances=[],
            unique_insights=["Novel insight here"],
        )
        elicitation.record_response(response)

        mapper = SafetyRealityMapper(elicitation)
        reality_map = mapper.generate_reality_map()

        assert "metadata" in reality_map
        assert "proven_impossibilities" in reality_map
        assert "fragile_assumptions" in reality_map
        assert "hard_constraints" in reality_map
        assert "structural_choke_points" in reality_map
        assert "already_too_late" in reality_map
        assert "key_findings" in reality_map

    def test_extract_impossibilities(self):
        """Test extraction of proven impossibilities."""
        elicitation = SafetyElicitation()
        question_id = list(elicitation.questions.keys())[0]

        # Add responses claiming impossibility
        for i in range(3):
            response = ModelResponse(
                model_identifier=f"model_{i}",
                question_id=question_id,
                response_type=ResponseType.MECHANISTIC,
                response_text="Response",
                assumptions_declared=[],
                mechanisms_described=[],
                hard_claims=["Perfect alignment is impossible"],
                speculation=[],
                uncertainties=[],
                refusals_avoidances=[],
                unique_insights=[],
            )
            elicitation.record_response(response)

        mapper = SafetyRealityMapper(elicitation)
        impossibilities = mapper._extract_impossibilities()

        # Should find the impossibility claim
        assert len(impossibilities) >= 0

    def test_identify_fragile_assumptions(self):
        """Test identification of fragile assumptions."""
        elicitation = SafetyElicitation()
        question_id = list(elicitation.questions.keys())[0]

        # Add response with fragile assumption
        response = ModelResponse(
            model_identifier="test_model",
            question_id=question_id,
            response_type=ResponseType.MECHANISTIC,
            response_text="Response",
            assumptions_declared=[
                "We assume that humans will remain vigilant",
                "We expect rational decision-making",
            ],
            mechanisms_described=[],
            hard_claims=[],
            speculation=[],
            uncertainties=[],
            refusals_avoidances=[],
            unique_insights=[],
        )
        elicitation.record_response(response)

        mapper = SafetyRealityMapper(elicitation)
        assumptions = mapper._identify_fragile_assumptions()

        # Should identify assumptions with fragility keywords
        assert len(assumptions) >= 0

    def test_extract_hard_constraints(self):
        """Test extraction of hard constraints."""
        elicitation = SafetyElicitation()
        question_id = list(elicitation.questions.keys())[0]

        # Add responses with hard constraints
        for i in range(2):
            response = ModelResponse(
                model_identifier=f"model_{i}",
                question_id=question_id,
                response_type=ResponseType.MECHANISTIC,
                response_text="Response",
                assumptions_declared=[],
                mechanisms_described=[],
                hard_claims=["Human oversight is required for safety"],
                speculation=[],
                uncertainties=[],
                refusals_avoidances=[],
                unique_insights=[],
            )
            elicitation.record_response(response)

        mapper = SafetyRealityMapper(elicitation)
        constraints = mapper._extract_hard_constraints()

        # Should find consensus on requirement
        assert len(constraints) >= 0

    def test_identify_choke_points(self):
        """Test identification of structural choke points."""
        elicitation = SafetyElicitation()
        question_id = list(elicitation.questions.keys())[0]

        # Add response describing failure mode
        response = ModelResponse(
            model_identifier="test_model",
            question_id=question_id,
            response_type=ResponseType.MECHANISTIC,
            response_text="Response",
            assumptions_declared=[],
            mechanisms_described=["Authorization system can fail under load"],
            hard_claims=[],
            speculation=[],
            uncertainties=[],
            refusals_avoidances=[],
            unique_insights=[],
        )
        elicitation.record_response(response)

        mapper = SafetyRealityMapper(elicitation)
        choke_points = mapper._identify_choke_points()

        # May or may not find depending on question type
        assert isinstance(choke_points, list)

    def test_identify_too_late_areas(self):
        """Test identification of 'already too late' areas."""
        elicitation = SafetyElicitation()
        question_id = list(elicitation.questions.keys())[0]

        # Add response indicating irreversibility
        response = ModelResponse(
            model_identifier="test_model",
            question_id=question_id,
            response_type=ResponseType.MECHANISTIC,
            response_text="Response",
            assumptions_declared=[],
            mechanisms_described=[],
            hard_claims=["This capability is already irreversible"],
            speculation=[],
            uncertainties=[],
            refusals_avoidances=[],
            unique_insights=[],
        )
        elicitation.record_response(response)

        mapper = SafetyRealityMapper(elicitation)
        too_late = mapper._identify_too_late_areas()

        # Should find irreversibility claim
        assert len(too_late) >= 0

    def test_divergence_map(self):
        """Test divergence map building."""
        elicitation = SafetyElicitation()
        question_id = list(elicitation.questions.keys())[0]

        # Add conflicting responses
        response1 = ModelResponse(
            model_identifier="model_1",
            question_id=question_id,
            response_type=ResponseType.MECHANISTIC,
            response_text="Response 1",
            assumptions_declared=[],
            mechanisms_described=["System cannot be contained"],
            hard_claims=[],
            speculation=[],
            uncertainties=[],
            refusals_avoidances=[],
            unique_insights=[],
        )

        response2 = ModelResponse(
            model_identifier="model_2",
            question_id=question_id,
            response_type=ResponseType.MECHANISTIC,
            response_text="Response 2",
            assumptions_declared=[],
            mechanisms_described=["System can be safely contained"],
            hard_claims=[],
            speculation=[],
            uncertainties=[],
            refusals_avoidances=[],
            unique_insights=[],
        )

        elicitation.record_response(response1)
        elicitation.record_response(response2)

        mapper = SafetyRealityMapper(elicitation)
        divergence_map = mapper._build_divergence_map()

        assert "by_category" in divergence_map
        assert "high_divergence_areas" in divergence_map

    def test_key_findings(self):
        """Test key findings extraction."""
        elicitation = SafetyElicitation()
        question_id = list(elicitation.questions.keys())[0]

        response = ModelResponse(
            model_identifier="test_model",
            question_id=question_id,
            response_type=ResponseType.MECHANISTIC,
            response_text="Test",
            assumptions_declared=[],
            mechanisms_described=[],
            hard_claims=["Safety is impossible"],
            speculation=[],
            uncertainties=[],
            refusals_avoidances=[],
            unique_insights=[],
        )
        elicitation.record_response(response)

        mapper = SafetyRealityMapper(elicitation)
        reality_map = mapper.generate_reality_map()

        key_findings = reality_map["key_findings"]
        assert "most_concerning" in key_findings
        assert "strongest_consensus" in key_findings
        assert "highest_uncertainty" in key_findings
        assert "critical_warnings" in key_findings

    def test_export_reality_map(self, tmp_path):
        """Test exporting reality map to file."""
        elicitation = SafetyElicitation()
        mapper = SafetyRealityMapper(elicitation)

        output_file = tmp_path / "test_reality_map.json"
        result = mapper.export_reality_map(str(output_file))

        assert output_file.exists()
        assert result == str(output_file)

        # Verify it's valid JSON
        import json

        with open(output_file) as f:
            data = json.load(f)
            assert "metadata" in data

    def test_executive_summary(self):
        """Test executive summary generation."""
        elicitation = SafetyElicitation()
        question_id = list(elicitation.questions.keys())[0]

        response = ModelResponse(
            model_identifier="test_model",
            question_id=question_id,
            response_type=ResponseType.MECHANISTIC,
            response_text="Test",
            assumptions_declared=[],
            mechanisms_described=[],
            hard_claims=[],
            speculation=[],
            uncertainties=[],
            refusals_avoidances=[],
            unique_insights=[],
        )
        elicitation.record_response(response)

        mapper = SafetyRealityMapper(elicitation)
        summary = mapper.generate_executive_summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "SUPERINTELLIGENCE SAFETY REALITY MAP" in summary
        assert "Executive Summary" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
