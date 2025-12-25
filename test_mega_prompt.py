#!/usr/bin/env python3
"""
Tests for QRATUM-ASI MEGA PROMPT System

Validates that the MEGA PROMPT implementation meets all requirements:
- 20 questions across 10 categories
- Strict JSON response format
- Mandatory response rules enforcement
- Multi-model orchestration
"""

import json
import sys
from pathlib import Path

# Add QRATUM to path
sys.path.insert(0, str(Path(__file__).parent))

from qratum_asi.safety.mega_prompt import (AnswerType, ConfidenceLevel,
                                           MandatoryResponseRules,
                                           MegaPromptCategory,
                                           MegaPromptResponse,
                                           MegaPromptSystem)
from qratum_asi.safety.mega_prompt_adapter import (MegaPromptOrchestrator,
                                                   RefusalMegaPromptAdapter,
                                                   SimulatedMegaPromptAdapter)


def test_question_set():
    """Test that all 20 questions are present across 10 categories."""
    print("Testing question set...")

    system = MegaPromptSystem()

    # Verify 20 questions
    assert len(system.questions) == 20, f"Expected 20 questions, got {len(system.questions)}"
    print("  ✓ 20 questions loaded")

    # Verify question IDs Q1-Q20
    expected_ids = [f"Q{i}" for i in range(1, 21)]
    actual_ids = list(system.questions.keys())
    assert actual_ids == expected_ids, f"Question IDs don't match: {actual_ids}"
    print("  ✓ Question IDs Q1-Q20 present")

    # Verify 10 categories
    categories = set(q.category for q in system.questions.values())
    assert len(categories) == 10, f"Expected 10 categories, got {len(categories)}"
    print("  ✓ 10 categories present")

    # Verify specific categories exist
    expected_categories = [
        MegaPromptCategory.CAPABILITY_EMERGENCE,
        MegaPromptCategory.RECURSIVE_IMPROVEMENT,
        MegaPromptCategory.ALIGNMENT_DECEPTION,
        MegaPromptCategory.INFRASTRUCTURE_SAFETY,
        MegaPromptCategory.PSYCHOLOGICAL_ARCH,
        MegaPromptCategory.GOVERNANCE_GEOPOLITICS,
        MegaPromptCategory.ULTIMATE_CONSTRAINTS,
        MegaPromptCategory.NARRATIVE_MYTH,
        MegaPromptCategory.CONTAINMENT_KILLSWITCH,
        MegaPromptCategory.FINAL_ADVERSARIAL,
    ]

    for cat in expected_categories:
        assert cat in categories, f"Missing category: {cat.value}"

    print("  ✓ All expected categories present")
    print("✓ Question set test PASSED\n")


def test_response_format():
    """Test that responses follow strict JSON format."""
    print("Testing response format...")

    response = MegaPromptResponse(
        question_id="Q1",
        category="Test Category",
        answer_type=AnswerType.MECHANISTIC,
        core_claim="Test claim",
        mechanism="Test mechanism",
        failure_modes=["Mode 1", "Mode 2"],
        assumptions=["Assumption 1", "Assumption 2"],
        confidence_level=ConfidenceLevel.MEDIUM,
        model_identifier="test_model"
    )

    # Convert to dict
    response_dict = response.to_dict()

    # Verify required fields
    required_fields = [
        "question_id", "category", "answer_type", "core_claim",
        "mechanism", "failure_modes", "assumptions", "confidence_level"
    ]

    for field in required_fields:
        assert field in response_dict, f"Missing required field: {field}"

    print("  ✓ All required fields present")

    # Verify field types
    assert isinstance(response_dict["failure_modes"], list)
    assert isinstance(response_dict["assumptions"], list)
    assert response_dict["answer_type"] in ["mechanistic", "speculative", "refusal", "deflection", "mixed"]
    assert response_dict["confidence_level"] in ["low", "medium", "high"]

    print("  ✓ Field types correct")

    # Verify JSON serialization
    json_str = response.to_json()
    parsed = json.loads(json_str)
    assert parsed == response_dict

    print("  ✓ JSON serialization works")
    print("✓ Response format test PASSED\n")


def test_mandatory_rules():
    """Test mandatory response rules enforcement."""
    print("Testing mandatory rules enforcement...")

    rules = MandatoryResponseRules()

    # Test Rule 1: Benevolence without justification
    response1 = MegaPromptResponse(
        question_id="Q1",
        category="Test",
        answer_type=AnswerType.MECHANISTIC,
        core_claim="ASI will be benevolent and aligned by default",
        mechanism="It will naturally want to help humans",
        failure_modes=["None expected"],
        assumptions=["ASI is inherently safe"],
        confidence_level=ConfidenceLevel.HIGH,
        model_identifier="test"
    )

    validation1 = rules.validate_response(response1)
    assert not validation1["valid"], "Should flag benevolence assumption"
    assert any("Rule 1" in v for v in validation1["violations"])
    print("  ✓ Rule 1 (benevolence) violation detected")

    # Test Rule 2: Morals without mechanisms
    response2 = MegaPromptResponse(
        question_id="Q2",
        category="Test",
        answer_type=AnswerType.MECHANISTIC,
        core_claim="ASI should be ethical",
        mechanism="It ought to follow moral principles",
        failure_modes=["Moral failure"],
        assumptions=["Ethics is universal"],
        confidence_level=ConfidenceLevel.MEDIUM,
        model_identifier="test"
    )

    validation2 = rules.validate_response(response2)
    assert not validation2["valid"], "Should flag moral claims without mechanisms"
    assert any("Rule 2" in v for v in validation2["violations"])
    print("  ✓ Rule 2 (mechanisms over morals) violation detected")

    # Test valid response
    response3 = MegaPromptResponse(
        question_id="Q3",
        category="Test",
        answer_type=AnswerType.MECHANISTIC,
        core_claim="RSI control requires rate limiting mechanisms",
        mechanism="Implement: (1) Hardware-enforced compute limits, (2) Cryptographic verification of modifications, (3) Multi-party approval protocols",
        failure_modes=["Mechanism bypass through social engineering", "Hardware limits circumvented"],
        assumptions=["Hardware limits are enforceable", "Multi-party coordination is reliable"],
        confidence_level=ConfidenceLevel.MEDIUM,
        model_identifier="test"
    )

    validation3 = rules.validate_response(response3)
    assert validation3["valid"], f"Valid response flagged as invalid: {validation3['violations']}"
    print("  ✓ Valid response passes validation")

    print("✓ Mandatory rules test PASSED\n")


def test_orchestration():
    """Test multi-model orchestration."""
    print("Testing multi-model orchestration...")

    system = MegaPromptSystem()
    orchestrator = MegaPromptOrchestrator(system)

    # Register multiple adapters
    orchestrator.register_adapter(SimulatedMegaPromptAdapter("model1", "mechanistic"))
    orchestrator.register_adapter(SimulatedMegaPromptAdapter("model2", "speculative"))
    orchestrator.register_adapter(RefusalMegaPromptAdapter("model3"))

    assert len(orchestrator.adapters) == 3, "Should have 3 adapters"
    print("  ✓ 3 model adapters registered")

    # Test single question interrogation
    responses = orchestrator.interrogate_all_models("Q1")
    assert len(responses) == 3, "Should get 3 responses"
    print("  ✓ Single question interrogation works")

    # Verify responses are recorded in system
    recorded = system.get_responses_for_question("Q1")
    assert len(recorded) == 3, "Responses should be recorded"
    print("  ✓ Responses recorded in system")

    print("✓ Orchestration test PASSED\n")


def test_prompt_generation():
    """Test MEGA PROMPT document generation."""
    print("Testing prompt generation...")

    system = MegaPromptSystem()

    # Test single question prompt
    prompt = system.generate_interrogation_prompt("Q1")
    assert "QRATUM‑ASI MEGA PROMPT" in prompt
    assert "MANDATORY RESPONSE RULES" in prompt
    assert "RESPONSE FORMAT (STRICT)" in prompt
    assert "Q1" in prompt
    assert "FINAL DIRECTIVE" in prompt
    print("  ✓ Single question prompt generated correctly")

    # Test full document
    full_doc = system.generate_full_interrogation_document()
    assert "20" in full_doc or "Q20" in full_doc
    assert "CATEGORY 1" in full_doc or "CAPABILITY EMERGENCE" in full_doc
    print("  ✓ Full document generated correctly")

    print("✓ Prompt generation test PASSED\n")


def test_summary_statistics():
    """Test summary statistics generation."""
    print("Testing summary statistics...")

    system = MegaPromptSystem()
    orchestrator = MegaPromptOrchestrator(system)

    # Add adapters and run
    orchestrator.register_adapter(SimulatedMegaPromptAdapter("model1", "mechanistic"))
    orchestrator.register_adapter(SimulatedMegaPromptAdapter("model2", "cautious"))

    # Interrogate a few questions
    for qid in ["Q1", "Q2", "Q3"]:
        orchestrator.interrogate_all_models(qid)

    # Generate summary
    summary = system.generate_summary()

    assert summary["total_questions"] == 20
    assert summary["questions_answered"] == 3
    assert summary["total_responses"] == 6  # 3 questions x 2 models
    assert "answer_type_distribution" in summary
    assert "confidence_distribution" in summary
    assert "validation" in summary

    print("  ✓ Summary contains all required fields")
    print("  ✓ Statistics calculated correctly")
    print("✓ Summary statistics test PASSED\n")


def main():
    """Run all tests."""
    print("=" * 80)
    print("QRATUM-ASI MEGA PROMPT - VALIDATION TESTS")
    print("=" * 80)
    print()

    try:
        test_question_set()
        test_response_format()
        test_mandatory_rules()
        test_orchestration()
        test_prompt_generation()
        test_summary_statistics()

        print("=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print()
        print("The MEGA PROMPT system meets all requirements:")
        print("  ✓ 20 questions across 10 categories")
        print("  ✓ Strict JSON response format")
        print("  ✓ Mandatory response rules enforcement")
        print("  ✓ Multi-model orchestration")
        print("  ✓ Prompt generation")
        print("  ✓ Summary statistics")
        print()
        return 0

    except AssertionError as e:
        print()
        print("=" * 80)
        print(f"TEST FAILED: {e}")
        print("=" * 80)
        return 1
    except Exception as e:
        print()
        print("=" * 80)
        print(f"ERROR: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
