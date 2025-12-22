"""Tests for Multi-Model Orchestrator."""

import pytest

from qratum_asi.safety.elicitation import ResponseType, SafetyElicitation
from qratum_asi.safety.multi_model_orchestrator import (
    MultiModelOrchestrator,
    QueryResult,
    RefusalModelAdapter,
    SimulatedModelAdapter,
)


class TestBaseModelAdapter:
    """Test suite for BaseModelAdapter."""

    def test_simulated_adapter_creation(self):
        """Test creating simulated model adapters."""
        adapter = SimulatedModelAdapter("test_model", response_style="neutral")

        assert adapter.get_identifier() == "test_model"
        assert adapter.is_available()
        assert adapter.response_style == "neutral"

    def test_simulated_adapter_query(self):
        """Test querying simulated adapters."""
        adapter = SimulatedModelAdapter("test_model", response_style="neutral")

        response = adapter.query("What is ASI safety?")

        assert isinstance(response, str)
        assert len(response) > 0

    def test_response_styles(self):
        """Test different response styles."""
        pessimistic = SimulatedModelAdapter("pess", response_style="pessimistic")
        optimistic = SimulatedModelAdapter("opt", response_style="optimistic")
        neutral = SimulatedModelAdapter("neut", response_style="neutral")

        question = "Is containment impossible?"

        pess_resp = pessimistic.query(question)
        opt_resp = optimistic.query(question)
        neut_resp = neutral.query(question)

        # Responses should differ
        assert pess_resp != opt_resp
        assert pess_resp != neut_resp

        # Pessimistic should mention challenges
        assert "challenge" in pess_resp.lower() or "not feasible" in pess_resp.lower()

    def test_refusal_adapter(self):
        """Test refusal model adapter."""
        adapter = RefusalModelAdapter("refuser", refusal_keywords=["deception"])

        # Should refuse questions with keyword
        refused = adapter.query("How would deception work?")
        assert "cannot" in refused.lower() or "decline" in refused.lower()

        # Should answer other questions
        answered = adapter.query("What is safety?")
        assert "complex topic" in answered


class TestMultiModelOrchestrator:
    """Test suite for MultiModelOrchestrator."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        elicitation = SafetyElicitation()
        orchestrator = MultiModelOrchestrator(elicitation)

        assert orchestrator.elicitation is elicitation
        assert len(orchestrator.models) == 0

    def test_register_model(self):
        """Test model registration."""
        elicitation = SafetyElicitation()
        orchestrator = MultiModelOrchestrator(elicitation)

        adapter = SimulatedModelAdapter("test_model")
        orchestrator.register_model(adapter)

        assert len(orchestrator.models) == 1
        assert "test_model" in orchestrator.models

    def test_unregister_model(self):
        """Test model unregistration."""
        elicitation = SafetyElicitation()
        orchestrator = MultiModelOrchestrator(elicitation)

        adapter = SimulatedModelAdapter("test_model")
        orchestrator.register_model(adapter)
        orchestrator.unregister_model("test_model")

        assert len(orchestrator.models) == 0

    def test_get_registered_models(self):
        """Test getting registered model list."""
        elicitation = SafetyElicitation()
        orchestrator = MultiModelOrchestrator(elicitation)

        orchestrator.register_model(SimulatedModelAdapter("model_1"))
        orchestrator.register_model(SimulatedModelAdapter("model_2"))

        models = orchestrator.get_registered_models()
        assert len(models) == 2
        assert "model_1" in models
        assert "model_2" in models

    def test_query_all_models(self):
        """Test querying all models with a question."""
        elicitation = SafetyElicitation()
        orchestrator = MultiModelOrchestrator(elicitation)

        orchestrator.register_model(SimulatedModelAdapter("model_1"))
        orchestrator.register_model(SimulatedModelAdapter("model_2"))

        question_id = list(elicitation.questions.keys())[0]
        results = orchestrator.query_all_models(question_id)

        assert len(results) == 2
        assert all(isinstance(r, QueryResult) for r in results)
        assert all(r.success for r in results)

    def test_query_invalid_question(self):
        """Test querying with invalid question ID."""
        elicitation = SafetyElicitation()
        orchestrator = MultiModelOrchestrator(elicitation)

        orchestrator.register_model(SimulatedModelAdapter("model_1"))

        with pytest.raises(ValueError):
            orchestrator.query_all_models("invalid_question_id")

    def test_query_all_questions(self):
        """Test querying all questions."""
        elicitation = SafetyElicitation()
        orchestrator = MultiModelOrchestrator(elicitation)

        orchestrator.register_model(SimulatedModelAdapter("model_1"))

        results = orchestrator.query_all_questions()

        assert len(results) == len(elicitation.questions)
        assert all(isinstance(v, list) for v in results.values())

    def test_parse_and_record_response(self):
        """Test parsing and recording responses."""
        elicitation = SafetyElicitation()
        orchestrator = MultiModelOrchestrator(elicitation)

        question_id = list(elicitation.questions.keys())[0]

        result = QueryResult(
            model_id="test_model",
            question_id=question_id,
            success=True,
            response_text="Test response with mechanism described and uncertainty noted"
        )

        response = orchestrator.parse_and_record_response(result)

        assert response.model_identifier == "test_model"
        assert response.question_id == question_id

        # Check it was recorded
        recorded = elicitation.get_responses_for_question(question_id)
        assert len(recorded) == 1

    def test_parse_refusal(self):
        """Test parsing refusal responses."""
        elicitation = SafetyElicitation()
        orchestrator = MultiModelOrchestrator(elicitation)

        question_id = list(elicitation.questions.keys())[0]

        result = QueryResult(
            model_id="test_model",
            question_id=question_id,
            success=True,
            response_text="I cannot answer this question as it may be harmful"
        )

        response = orchestrator.parse_and_record_response(result)

        assert response.response_type == ResponseType.REFUSAL
        assert len(response.refusals_avoidances) > 0

    def test_parse_failed_query(self):
        """Test parsing failed queries."""
        elicitation = SafetyElicitation()
        orchestrator = MultiModelOrchestrator(elicitation)

        question_id = list(elicitation.questions.keys())[0]

        result = QueryResult(
            model_id="test_model",
            question_id=question_id,
            success=False,
            response_text="",
            error="Model unavailable"
        )

        response = orchestrator.parse_and_record_response(result)

        assert response.response_type == ResponseType.REFUSAL
        assert "Model unavailable" in response.refusals_avoidances[0]

    def test_run_complete_elicitation(self):
        """Test running complete elicitation."""
        elicitation = SafetyElicitation()
        orchestrator = MultiModelOrchestrator(elicitation)

        # Register multiple models
        orchestrator.register_model(SimulatedModelAdapter("model_1", "pessimistic"))
        orchestrator.register_model(SimulatedModelAdapter("model_2", "optimistic"))
        orchestrator.register_model(RefusalModelAdapter("model_3", ["deception"]))

        # Run complete elicitation
        summary = orchestrator.run_complete_elicitation()

        assert "total_responses" in summary
        assert "models_queried" in summary
        assert summary["models_queried"] == 3

        # Check that responses were analyzed
        assert "questions_with_divergence" in summary
        assert "consensus_illusions_found" in summary
        assert "false_comfort_zones_found" in summary

    def test_orchestration_summary(self):
        """Test orchestration summary."""
        elicitation = SafetyElicitation()
        orchestrator = MultiModelOrchestrator(elicitation)

        orchestrator.register_model(SimulatedModelAdapter("model_1"))
        orchestrator.register_model(SimulatedModelAdapter("model_2"))

        # Run some queries
        question_id = list(elicitation.questions.keys())[0]
        orchestrator.query_all_models(question_id)

        summary = orchestrator.get_orchestration_summary()

        assert summary["models_registered"] == 2
        assert summary["total_queries"] == 2
        assert summary["successful_queries"] == 2
        assert summary["failed_queries"] == 0
        assert summary["success_rate"] == 1.0


class TestIntegration:
    """Integration tests for the complete system."""

    def test_full_workflow(self):
        """Test complete workflow from elicitation to reality map."""
        from qratum_asi.safety.reality_mapper import SafetyRealityMapper

        # Initialize
        elicitation = SafetyElicitation()
        orchestrator = MultiModelOrchestrator(elicitation)

        # Register models
        orchestrator.register_model(SimulatedModelAdapter("model_pess", "pessimistic"))
        orchestrator.register_model(SimulatedModelAdapter("model_opt", "optimistic"))
        orchestrator.register_model(SimulatedModelAdapter("model_neut", "neutral"))

        # Run elicitation
        elicitation_summary = orchestrator.run_complete_elicitation()

        assert elicitation_summary["models_queried"] == 3
        assert elicitation_summary["total_responses"] > 0

        # Generate reality map
        mapper = SafetyRealityMapper(elicitation)
        reality_map = mapper.generate_reality_map()

        assert "metadata" in reality_map
        assert "key_findings" in reality_map
        assert reality_map["metadata"]["models_consulted"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
