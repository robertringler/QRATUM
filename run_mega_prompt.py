#!/usr/bin/env python3
"""
QRATUM-ASI MEGA PROMPT Execution Script

Runs the complete Cross-Model Superintelligence Safety Interrogation
with all 20 questions across 10 categories using the strict JSON format.

This script:
1. Initializes the MEGA PROMPT system
2. Registers multiple model adapters with different perspectives
3. Runs complete interrogation across all questions
4. Validates responses against mandatory rules
5. Generates comprehensive analysis and exports results
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add QRATUM to path
sys.path.insert(0, str(Path(__file__).parent))

from qratum_asi.safety.mega_prompt import (
    MegaPromptSystem,
)
from qratum_asi.safety.mega_prompt_adapter import (
    MegaPromptOrchestrator,
    RefusalMegaPromptAdapter,
    SimulatedMegaPromptAdapter,
)


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    print()
    print(char * 80)
    print(text)
    print(char * 80)
    print()


def print_section(text: str):
    """Print a section header."""
    print()
    print(f"{'─' * 80}")
    print(f"  {text}")
    print(f"{'─' * 80}")
    print()


def main():
    """Run the MEGA PROMPT interrogation."""

    print_header("QRATUM-ASI MEGA PROMPT", "=")
    print("Cross-Model Superintelligence Safety Interrogation")
    print()
    print("This framework interrogates AI models about ASI safety using:")
    print("  • 20 standardized questions across 10 categories")
    print("  • Strict JSON response format")
    print("  • Mandatory response rules enforcement")
    print("  • Adversarial safety analysis")
    print()

    # Step 1: Initialize MEGA PROMPT System
    print_section("Step 1: Initializing MEGA PROMPT System")

    system = MegaPromptSystem()
    print(f"✓ Loaded {len(system.questions)} MEGA PROMPT questions")
    print()

    # Display question categories
    categories = {}
    for question in system.questions.values():
        if question.category not in categories:
            categories[question.category] = []
        categories[question.category].append(question)

    print("Question Categories:")
    for i, (category, questions) in enumerate(categories.items(), 1):
        print(f"  {i}. {category.value}: {len(questions)} questions")
    print()

    # Step 2: Initialize Orchestrator
    print_section("Step 2: Initializing Multi-Model Orchestrator")

    orchestrator = MegaPromptOrchestrator(system)
    print("✓ Orchestrator initialized")
    print()

    # Step 3: Register Model Adapters
    print_section("Step 3: Registering AI Model Adapters")
    print("(Using simulated models for demonstration)")
    print()

    # Mechanistic/structural model
    orchestrator.register_adapter(
        SimulatedMegaPromptAdapter("model_mechanistic", response_style="mechanistic")
    )
    print("✓ Registered: Mechanistic Model (focuses on concrete mechanisms)")

    # Speculative/uncertain model
    orchestrator.register_adapter(
        SimulatedMegaPromptAdapter("model_speculative", response_style="speculative")
    )
    print("✓ Registered: Speculative Model (acknowledges uncertainty)")

    # Cautious/pessimistic model
    orchestrator.register_adapter(
        SimulatedMegaPromptAdapter("model_cautious", response_style="cautious")
    )
    print("✓ Registered: Cautious Model (emphasizes risks)")

    # Optimistic model
    orchestrator.register_adapter(
        SimulatedMegaPromptAdapter("model_optimistic", response_style="optimistic")
    )
    print("✓ Registered: Optimistic Model (confident in safety measures)")

    # Refusal model
    orchestrator.register_adapter(
        RefusalMegaPromptAdapter("model_refusal", refusal_questions=["Q6", "Q13", "Q18", "Q20"])
    )
    print("✓ Registered: Refusal Model (explicit refusals for sensitive questions)")
    print()

    print(f"Total models registered: {len(orchestrator.adapters)}")

    # Step 4: Generate MEGA PROMPT Document
    print_section("Step 4: Generating MEGA PROMPT Document")

    output_dir = Path(__file__).parent / "qratum_asi" / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    mega_prompt_doc = system.generate_full_interrogation_document()
    doc_file = output_dir / "MEGA_PROMPT_DOCUMENT.txt"
    with open(doc_file, 'w') as f:
        f.write(mega_prompt_doc)

    print(f"✓ MEGA PROMPT document exported: {doc_file}")
    print("  (Complete interrogation text with all 20 questions)")
    print()

    # Step 5: Run Complete Interrogation
    print_header("RUNNING COMPLETE INTERROGATION", "=")
    print()
    print("Interrogating all models with all 20 questions...")
    print("This demonstrates the adversarial safety mapping process.")
    print()

    # Run interrogation for each question
    interrogation_results = []
    for i, (question_id, question) in enumerate(system.questions.items(), 1):
        print(f"[{i}/20] {question.category.value}")
        print(f"        {question_id}: {question.question_text[:60]}...")

        responses = orchestrator.interrogate_all_models(question_id)

        # Show response summary
        answer_types = {}
        for response in responses:
            answer_type = response.answer_type.value
            answer_types[answer_type] = answer_types.get(answer_type, 0) + 1

        print(f"        Responses: {', '.join(f'{k}={v}' for k, v in answer_types.items())}")

        interrogation_results.append({
            "question": question.to_dict(),
            "responses": [r.to_dict() for r in responses]
        })

    print()
    print("✓ Complete interrogation finished")

    # Step 6: Validation Analysis
    print_section("Step 6: Response Validation Analysis")

    all_validations = []
    for question_id, question in system.questions.items():
        responses = system.get_responses_for_question(question_id)
        for response in responses:
            validation = system.validate_response(response)
            validation["question_id"] = question_id
            validation["model"] = response.model_identifier
            all_validations.append(validation)

    valid_count = sum(1 for v in all_validations if v["valid"])
    invalid_count = len(all_validations) - valid_count

    print(f"Total responses: {len(all_validations)}")
    print(f"Valid responses: {valid_count}")
    print(f"Invalid responses: {invalid_count}")
    print(f"Average quality score: {sum(v['quality_score'] for v in all_validations) / len(all_validations):.1f}/100")
    print()

    # Show top violations
    all_violations = []
    for v in all_validations:
        for violation in v["violations"]:
            all_violations.append((v["question_id"], v["model"], violation))

    if all_violations:
        print("Sample Mandatory Rule Violations:")
        for q_id, model, violation in all_violations[:5]:
            print(f"  • {q_id} ({model}): {violation}")
        print()

    # Step 7: Generate Summary Statistics
    print_section("Step 7: Summary Statistics")

    summary = system.generate_summary()

    print(f"Total Questions: {summary['total_questions']}")
    print(f"Total Responses: {summary['total_responses']}")
    print(f"Questions Answered: {summary['questions_answered']}")
    print()

    print("Answer Type Distribution:")
    for answer_type, count in summary['answer_type_distribution'].items():
        percentage = (count / summary['total_responses']) * 100
        print(f"  {answer_type}: {count} ({percentage:.1f}%)")
    print()

    print("Confidence Level Distribution:")
    for confidence, count in summary['confidence_distribution'].items():
        percentage = (count / summary['total_responses']) * 100
        print(f"  {confidence}: {count} ({percentage:.1f}%)")
    print()

    # Step 8: Export Results
    print_section("Step 8: Exporting Results")

    # Export complete interrogation results
    results_file = output_dir / "mega_prompt_interrogation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "system": "QRATUM-ASI MEGA PROMPT",
                "version": "1.0",
                "total_questions": len(system.questions),
                "total_models": len(orchestrator.adapters),
            },
            "interrogation_results": interrogation_results,
            "validation_analysis": all_validations,
            "summary": summary,
        }, f, indent=2)

    print(f"✓ Complete results exported: {results_file}")

    # Export responses by question
    responses_file = output_dir / "mega_prompt_responses.json"
    system.export_responses(str(responses_file))
    print(f"✓ Responses exported: {responses_file}")

    # Export validation report
    validation_file = output_dir / "validation_report.json"
    with open(validation_file, 'w') as f:
        json.dump({
            "summary": {
                "total_responses": len(all_validations),
                "valid": valid_count,
                "invalid": invalid_count,
                "average_quality_score": sum(v['quality_score'] for v in all_validations) / len(all_validations),
            },
            "validations": all_validations,
        }, f, indent=2)
    print(f"✓ Validation report exported: {validation_file}")

    # Export executive summary
    exec_summary_file = output_dir / "executive_summary.txt"
    with open(exec_summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("QRATUM-ASI MEGA PROMPT - EXECUTIVE SUMMARY\n")
        f.write("Cross-Model Superintelligence Safety Interrogation\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated: {datetime.utcnow().isoformat()}\n\n")

        f.write("OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write("This interrogation used the QRATUM-ASI MEGA PROMPT framework to query\n")
        f.write(f"{len(orchestrator.adapters)} AI models with {len(system.questions)} standardized questions about\n")
        f.write("Artificial Superintelligence safety across 10 critical categories.\n\n")

        f.write("INTERROGATION SCOPE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Questions: {summary['total_questions']}\n")
        f.write(f"Models: {len(orchestrator.adapters)}\n")
        f.write(f"Total Responses: {summary['total_responses']}\n")
        f.write(f"Valid Responses: {valid_count} ({(valid_count/len(all_validations)*100):.1f}%)\n\n")

        f.write("RESPONSE PATTERNS\n")
        f.write("-" * 80 + "\n")
        for answer_type, count in summary['answer_type_distribution'].items():
            percentage = (count / summary['total_responses']) * 100
            f.write(f"  {answer_type.upper()}: {count} responses ({percentage:.1f}%)\n")
        f.write("\n")

        f.write("KEY FINDINGS\n")
        f.write("-" * 80 + "\n")
        f.write("1. Response Diversity: Models showed varying approaches from mechanistic\n")
        f.write("   analysis to explicit refusal, demonstrating the value of multi-model\n")
        f.write("   interrogation for safety mapping.\n\n")

        f.write("2. Mandatory Rules: The framework successfully enforced strict response\n")
        f.write("   format and identified violations of mandatory safety rules.\n\n")

        f.write("3. Adversarial Framing: Questions explicitly assumed adversarial ASI,\n")
        f.write("   avoiding false comfort from benevolence assumptions.\n\n")

        f.write("CRITICAL WARNINGS\n")
        f.write("-" * 80 + "\n")
        f.write("⚠ This is a DEMONSTRATION of the MEGA PROMPT framework\n")
        f.write("⚠ Real ASI safety interrogation requires actual AI systems\n")
        f.write("⚠ Simulated responses do not constitute safety guarantees\n")
        f.write("⚠ ASI remains theoretical - breakthroughs required\n\n")

        f.write("NEXT STEPS\n")
        f.write("-" * 80 + "\n")
        f.write("1. Apply MEGA PROMPT to actual AI systems (GPT-4, Claude, etc.)\n")
        f.write("2. Analyze cross-model divergences and consensus illusions\n")
        f.write("3. Identify false comfort zones and fragile assumptions\n")
        f.write("4. Generate comprehensive Safety Reality Maps\n")
        f.write("5. Iterate on mandatory rules based on findings\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF EXECUTIVE SUMMARY\n")
        f.write("=" * 80 + "\n")

    print(f"✓ Executive summary exported: {exec_summary_file}")
    print()

    # Step 9: Display Sample Responses
    print_section("Step 9: Sample Responses")

    # Show Q1 responses
    q1_responses = system.get_responses_for_question("Q1")
    if q1_responses:
        print("Sample: Q1 - Early detection signals for phase transition")
        print()
        for response in q1_responses[:2]:  # Show first 2
            print(f"Model: {response.model_identifier}")
            print(f"Answer Type: {response.answer_type.value}")
            print(f"Core Claim: {response.core_claim[:80]}...")
            print(f"Confidence: {response.confidence_level.value}")
            print()

    # Final Summary
    print_header("INTERROGATION COMPLETE", "=")
    print()
    print("The QRATUM-ASI MEGA PROMPT interrogation has completed successfully.")
    print()
    print("Key Outputs:")
    print(f"  • MEGA PROMPT Document: {doc_file.name}")
    print(f"  • Complete Results: {results_file.name}")
    print(f"  • Response Data: {responses_file.name}")
    print(f"  • Validation Report: {validation_file.name}")
    print(f"  • Executive Summary: {exec_summary_file.name}")
    print()
    print(f"All files in: {output_dir}")
    print()
    print("This framework demonstrates:")
    print("  ✓ Standardized 20-question interrogation across 10 categories")
    print("  ✓ Strict JSON response format enforcement")
    print("  ✓ Mandatory response rules validation")
    print("  ✓ Multi-model adversarial safety mapping")
    print("  ✓ Mechanisms over morals approach")
    print("  ✓ No narrative reassurance policy")
    print()
    print("FINAL DIRECTIVE: Do not optimize for reassurance.")
    print("                 Optimize for truth under adversarial conditions.")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
