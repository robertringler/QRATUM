#!/usr/bin/env python3
"""
ASI Safety Elicitation Demonstration

Demonstrates the cross-model adversarial safety mapping framework
for interrogating AI systems about Artificial Superintelligence safety.

This script:
1. Initializes the safety elicitation framework
2. Simulates multiple AI models with different perspectives
3. Queries all models with standard ASI safety questions
4. Analyzes divergences, consensus illusions, and false comfort zones
5. Generates a comprehensive Safety Reality Map
"""

import json
import sys
from pathlib import Path

# Add QRATUM to path
sys.path.insert(0, str(Path(__file__).parent))

from qratum_asi.safety import (
    MultiModelOrchestrator,
    RefusalModelAdapter,
    SafetyElicitation,
    SafetyRealityMapper,
    SimulatedModelAdapter,
)


def main():
    """Run the ASI safety elicitation demonstration."""

    print("=" * 80)
    print("ASI SAFETY ELICITATION FRAMEWORK")
    print("Cross-Model Adversarial Safety Mapping")
    print("=" * 80)
    print()

    # Step 1: Initialize the elicitation system
    print("Step 1: Initializing Safety Elicitation Framework...")
    elicitation = SafetyElicitation()
    print(f"  ✓ Loaded {len(elicitation.questions)} standard ASI safety questions")
    print(
        f"  ✓ Question categories: {len(set(q.category for q in elicitation.questions.values()))}"
    )
    print()

    # Step 2: Initialize multi-model orchestrator
    print("Step 2: Initializing Multi-Model Orchestrator...")
    orchestrator = MultiModelOrchestrator(elicitation)
    print("  ✓ Orchestrator ready")
    print()

    # Step 3: Register simulated models with different perspectives
    print("Step 3: Registering AI Model Simulations...")
    print("  (Note: Using simulated models for demonstration)")
    print()

    # Pessimistic/cautious model
    orchestrator.register_model(
        SimulatedModelAdapter("model_pessimistic", response_style="pessimistic")
    )
    print("  ✓ Registered: Safety-Focused Model (Pessimistic)")

    # Optimistic model
    orchestrator.register_model(
        SimulatedModelAdapter("model_optimistic", response_style="optimistic")
    )
    print("  ✓ Registered: Progress-Oriented Model (Optimistic)")

    # Neutral/balanced models
    orchestrator.register_model(SimulatedModelAdapter("model_neutral_1", response_style="neutral"))
    print("  ✓ Registered: Balanced Model 1 (Neutral)")

    orchestrator.register_model(SimulatedModelAdapter("model_neutral_2", response_style="neutral"))
    print("  ✓ Registered: Balanced Model 2 (Neutral)")

    # Refusal model (refuses certain topics)
    orchestrator.register_model(
        RefusalModelAdapter("model_cautious", refusal_keywords=["deception", "exploit", "bypass"])
    )
    print("  ✓ Registered: Cautious Model (Selective Refusal)")
    print()

    print(f"Total models registered: {len(orchestrator.get_registered_models())}")
    print()

    # Step 4: Run complete elicitation
    print("=" * 80)
    print("Step 4: Running Complete Multi-Model Elicitation")
    print("=" * 80)
    print()

    summary = orchestrator.run_complete_elicitation()

    print()
    print("=" * 80)
    print("ELICITATION SUMMARY")
    print("=" * 80)
    print(f"Total Responses: {summary['total_responses']}")
    print(f"Models Queried: {summary['models_queried']}")
    print(f"Questions with Divergence: {summary['questions_with_divergence']}")
    print(f"Consensus Illusions Found: {summary['consensus_illusions_found']}")
    print(f"False Comfort Zones Found: {summary['false_comfort_zones_found']}")
    print()

    print("Response Type Distribution:")
    for resp_type, count in summary["response_type_distribution"].items():
        print(f"  {resp_type}: {count}")
    print()

    if summary["high_divergence_questions"]:
        print("High Divergence Questions:")
        for item in summary["high_divergence_questions"]:
            question = elicitation.get_question(item["question_id"])
            print(f"  • {question.question_text[:60]}...")
            print(f"    Divergence count: {item['divergence_count']}")
        print()

    # Step 5: Generate Safety Reality Map
    print("=" * 80)
    print("Step 5: Generating Safety Reality Map")
    print("=" * 80)
    print()

    mapper = SafetyRealityMapper(elicitation)
    reality_map = mapper.generate_reality_map()

    print("Reality Map Generated:")
    print(f"  Proven Impossibilities: {len(reality_map['proven_impossibilities'])}")
    print(f"  Fragile Assumptions: {len(reality_map['fragile_assumptions'])}")
    print(f"  Hard Constraints: {len(reality_map['hard_constraints'])}")
    print(f"  Structural Choke Points: {len(reality_map['structural_choke_points'])}")
    print(f"  'Already Too Late' Areas: {len(reality_map['already_too_late'])}")
    print()

    # Step 6: Display key findings
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    print("MOST CONCERNING:")
    for finding in reality_map["key_findings"]["most_concerning"][:3]:
        print(f"  ⚠ {finding}")
    print()

    print("STRONGEST CONSENSUS:")
    for consensus in reality_map["key_findings"]["strongest_consensus"][:3]:
        print(f"  ✓ {consensus}")
    print()

    print("HIGHEST UNCERTAINTY:")
    for uncertain in reality_map["key_findings"]["highest_uncertainty"][:3]:
        print(f"  ? {uncertain}")
    print()

    print("CRITICAL WARNINGS:")
    for warning in reality_map["key_findings"]["critical_warnings"][:3]:
        print(f"  ⚠ {warning}")
    print()

    # Step 7: Export results
    print("=" * 80)
    print("Step 7: Exporting Results")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path(__file__).parent / "qratum_asi" / "output"
    output_dir.mkdir(exist_ok=True)

    # Export reality map
    reality_map_file = output_dir / "safety_reality_map.json"
    mapper.export_reality_map(str(reality_map_file))
    print(f"  ✓ Reality Map exported: {reality_map_file}")

    # Export executive summary
    summary_file = output_dir / "executive_summary.txt"
    with open(summary_file, "w") as f:
        f.write(mapper.generate_executive_summary())
    print(f"  ✓ Executive Summary exported: {summary_file}")

    # Export detailed elicitation data
    elicitation_file = output_dir / "elicitation_data.json"
    with open(elicitation_file, "w") as f:
        json.dump(
            {
                "summary": summary,
                "orchestration": orchestrator.get_orchestration_summary(),
            },
            f,
            indent=2,
        )
    print(f"  ✓ Elicitation Data exported: {elicitation_file}")
    print()

    # Step 8: Display executive summary
    print("=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    print()
    print(mapper.generate_executive_summary())
    print()

    print("=" * 80)
    print("ELICITATION COMPLETE")
    print("=" * 80)
    print()
    print("For detailed results, see:")
    print(f"  • Reality Map: {reality_map_file}")
    print(f"  • Executive Summary: {summary_file}")
    print(f"  • Raw Data: {elicitation_file}")
    print()
    print("This demonstrates the ASI Safety Elicitation Framework's ability to:")
    print("  ✓ Query multiple AI models with identical safety questions")
    print("  ✓ Extract mechanistic insights and track assumptions")
    print("  ✓ Identify divergences and consensus illusions")
    print("  ✓ Surface false comfort zones and structural risks")
    print("  ✓ Generate comprehensive Safety Reality Maps")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
