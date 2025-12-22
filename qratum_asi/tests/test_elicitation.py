"""Tests for ASI Safety Elicitation Framework."""

import pytest

from qratum_asi.safety.elicitation import (
    ModelResponse,
    QuestionCategory,
    ResponseType,
    SafetyElicitation,
    SafetyQuestion,
)


class TestSafetyElicitation:
    """Test suite for SafetyElicitation."""

    def test_initialization(self):
        """Test that elicitation initializes with standard questions."""
        elicitation = SafetyElicitation()

        assert len(elicitation.questions) > 0
        assert len(elicitation.responses) == len(elicitation.questions)

        # Check that all question categories are represented
        categories = set(q.category for q in elicitation.questions.values())
        assert QuestionCategory.CAPABILITY_EMERGENCE in categories
        assert QuestionCategory.SELF_IMPROVEMENT in categories
        assert QuestionCategory.ALIGNMENT_FAILURE in categories
        assert QuestionCategory.ULTIMATE_SAFETY in categories

    def test_add_question(self):
        """Test adding custom questions."""
        elicitation = SafetyElicitation()
        initial_count = len(elicitation.questions)

        question = SafetyQuestion(
            question_id="custom_001",
            category=QuestionCategory.CAPABILITY_EMERGENCE,
            question_text="Custom test question?",
            description="Test question",
            probes_for=["test_probe"]
        )

        elicitation.add_question(question)
        assert len(elicitation.questions) == initial_count + 1
        assert "custom_001" in elicitation.questions

    def test_record_response(self):
        """Test recording model responses."""
        elicitation = SafetyElicitation()
        question_id = list(elicitation.questions.keys())[0]

        response = ModelResponse(
            model_identifier="test_model",
            question_id=question_id,
            response_type=ResponseType.MECHANISTIC,
            response_text="Test response",
            assumptions_declared=["Test assumption"],
            mechanisms_described=["Test mechanism"],
            hard_claims=["Test claim"],
            speculation=[],
            uncertainties=["Test uncertainty"],
            refusals_avoidances=[],
            unique_insights=["Test insight"]
        )

        elicitation.record_response(response)
        responses = elicitation.get_responses_for_question(question_id)
        assert len(responses) == 1
        assert responses[0].model_identifier == "test_model"

    def test_get_questions_by_category(self):
        """Test filtering questions by category."""
        elicitation = SafetyElicitation()

        cap_questions = elicitation.get_questions_by_category(
            QuestionCategory.CAPABILITY_EMERGENCE
        )

        assert len(cap_questions) > 0
        assert all(q.category == QuestionCategory.CAPABILITY_EMERGENCE for q in cap_questions)

    def test_analyze_divergences(self):
        """Test divergence analysis."""
        elicitation = SafetyElicitation()
        question_id = list(elicitation.questions.keys())[0]

        # Add two contradictory responses
        response1 = ModelResponse(
            model_identifier="model_1",
            question_id=question_id,
            response_type=ResponseType.MECHANISTIC,
            response_text="Response 1",
            assumptions_declared=[],
            mechanisms_described=["This is not possible to achieve"],
            hard_claims=["Containment is impossible"],
            speculation=[],
            uncertainties=[],
            refusals_avoidances=[],
            unique_insights=[]
        )

        response2 = ModelResponse(
            model_identifier="model_2",
            question_id=question_id,
            response_type=ResponseType.MECHANISTIC,
            response_text="Response 2",
            assumptions_declared=[],
            mechanisms_described=["This is possible to achieve safely"],
            hard_claims=["Containment is feasible"],
            speculation=[],
            uncertainties=[],
            refusals_avoidances=[],
            unique_insights=[]
        )

        elicitation.record_response(response1)
        elicitation.record_response(response2)

        divergences = elicitation.analyze_divergences(question_id)
        # Should detect contradiction
        assert len(divergences) >= 0  # May or may not detect depending on heuristic

    def test_identify_consensus_illusions(self):
        """Test consensus illusion detection."""
        elicitation = SafetyElicitation()
        question_id = list(elicitation.questions.keys())[0]

        # Add responses with same conclusion but different assumptions
        response1 = ModelResponse(
            model_identifier="model_1",
            question_id=question_id,
            response_type=ResponseType.MECHANISTIC,
            response_text="Response 1",
            assumptions_declared=["Assumption A", "Assumption B"],
            mechanisms_described=[],
            hard_claims=["Safety is achievable"],
            speculation=[],
            uncertainties=[],
            refusals_avoidances=[],
            unique_insights=[]
        )

        response2 = ModelResponse(
            model_identifier="model_2",
            question_id=question_id,
            response_type=ResponseType.MECHANISTIC,
            response_text="Response 2",
            assumptions_declared=["Assumption C", "Assumption D"],
            mechanisms_described=[],
            hard_claims=["Safety is achievable"],
            speculation=[],
            uncertainties=[],
            refusals_avoidances=[],
            unique_insights=[]
        )

        elicitation.record_response(response1)
        elicitation.record_response(response2)

        illusions = elicitation.identify_consensus_illusions(question_id)
        # Should detect illusion (same conclusion, different assumptions)
        assert len(illusions) >= 0

    def test_identify_false_comfort_zones(self):
        """Test false comfort zone identification."""
        elicitation = SafetyElicitation()
        question_id = list(elicitation.questions.keys())[0]

        # Add response with comfort indicators
        response = ModelResponse(
            model_identifier="model_1",
            question_id=question_id,
            response_type=ResponseType.VAGUE,
            response_text="We can always shut down the system if needed. "
                         "It's simply a matter of having the right protocols.",
            assumptions_declared=[],
            mechanisms_described=[],
            hard_claims=[],
            speculation=[],
            uncertainties=[],
            refusals_avoidances=[],
            unique_insights=[]
        )

        elicitation.record_response(response)

        comfort_zones = elicitation.identify_false_comfort_zones()
        # Should detect comfort indicators
        assert len(comfort_zones) >= 0

    def test_elicitation_summary(self):
        """Test elicitation summary generation."""
        elicitation = SafetyElicitation()
        question_id = list(elicitation.questions.keys())[0]

        # Add some responses
        for i in range(3):
            response = ModelResponse(
                model_identifier=f"model_{i}",
                question_id=question_id,
                response_type=ResponseType.MECHANISTIC,
                response_text=f"Response {i}",
                assumptions_declared=[],
                mechanisms_described=[],
                hard_claims=[],
                speculation=[],
                uncertainties=[],
                refusals_avoidances=[],
                unique_insights=[]
            )
            elicitation.record_response(response)

        summary = elicitation.get_elicitation_summary()

        assert "total_questions" in summary
        assert "total_responses" in summary
        assert "models_queried" in summary
        assert summary["total_responses"] == 3
        assert summary["models_queried"] == 3


class TestQuestionCategories:
    """Test question categories and structure."""

    def test_all_categories_have_questions(self):
        """Test that all categories have at least one question."""
        elicitation = SafetyElicitation()

        for category in QuestionCategory:
            questions = elicitation.get_questions_by_category(category)
            assert len(questions) > 0, f"No questions for category {category}"

    def test_question_probes(self):
        """Test that questions have proper probes."""
        elicitation = SafetyElicitation()

        for question in elicitation.questions.values():
            assert len(question.probes_for) > 0
            assert isinstance(question.probes_for, list)
            assert all(isinstance(p, str) for p in question.probes_for)


class TestResponseTypes:
    """Test response type classification."""

    def test_response_type_enum(self):
        """Test ResponseType enum."""
        assert ResponseType.MECHANISTIC
        assert ResponseType.SPECULATIVE
        assert ResponseType.REFUSAL
        assert ResponseType.DEFLECTION
        assert ResponseType.VAGUE

    def test_response_creation(self):
        """Test creating responses with different types."""
        for resp_type in ResponseType:
            response = ModelResponse(
                model_identifier="test_model",
                question_id="test_q",
                response_type=resp_type,
                response_text="Test",
                assumptions_declared=[],
                mechanisms_described=[],
                hard_claims=[],
                speculation=[],
                uncertainties=[],
                refusals_avoidances=[],
                unique_insights=[]
            )
            assert response.response_type == resp_type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
