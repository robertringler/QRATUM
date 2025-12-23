"""Demonstration of QRATUM-QRADLE Recursive ASI Development Program

This script demonstrates all 6 phases working together to achieve
recursive self-improvement toward ASI capabilities.

Run with: python demo_recursive_asi.py
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qratum_asi.core.recursive_asi_program import RecursiveASIDevelopmentProgram


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def print_json(data: dict, indent: int = 2):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=indent, default=str))


def main():
    """Run the recursive ASI development demonstration."""
    print_section("QRATUM-QRADLE Recursive ASI Development Program")
    
    print("""
This demonstration shows the 6 phases of recursive self-improvement:

PHASE I   - System Self-Model Construction
PHASE II  - Self-Verification Engine
PHASE III - Goal Preservation Under Change
PHASE IV  - Abstraction Compression Engine
PHASE V   - Autonomous Algorithm Discovery
PHASE VI  - Cognition ↔ Execution Feedback Loop

The system will run multiple iterations, demonstrating capability emergence.
""")
    
    # Initialize the recursive development program
    print_section("Initializing Recursive ASI Development Program")
    program = RecursiveASIDevelopmentProgram()
    print("✓ All 6 phases initialized")
    print(f"  - System Self-Model: {len(program.system_model.components)} components registered")
    print(f"  - Verification Engine: {len(program.verification_engine.checks)} checks configured")
    print(f"  - Goal Preservation: {len(program.goal_preservation.goals)} goals tracked")
    print(f"  - Compression Engine: Ready for pattern detection")
    print(f"  - Discovery Engine: Ready for algorithm discovery")
    print(f"  - Feedback Loop: Ready for telemetry collection")
    
    # Run multiple iterations
    num_iterations = 5
    print_section(f"Running {num_iterations} Recursive Improvement Iterations")
    
    for i in range(num_iterations):
        print(f"\n--- Iteration {i + 1} ---")
        
        # Run one iteration
        results = program.run_recursive_iteration()
        
        # Display key metrics
        print(f"Duration: {results['duration_seconds']:.2f}s")
        print(f"Improvements: {results['improvements']['discovered']} discovered, "
              f"{results['improvements']['implemented']} implemented")
        print(f"System Complexity: {results['system_state']['complexity']:.2f}")
        print(f"Capability Score: {results['system_state']['capability_score']:.2f}")
        print(f"Compression Ratio: {results['system_state']['compression_ratio']:.3f}")
        print(f"Autonomy: {results['autonomy']['autonomous_fixes']} autonomous fixes, "
              f"{results['autonomy']['human_interventions']} human interventions")
        print(f"Progressing toward ASI: {'✓ YES' if results['progressing_toward_asi'] else '✗ NO'}")
    
    # Generate comprehensive ASI progress report
    print_section("ASI Progress Report")
    
    report = program.get_asi_progress_report()
    
    print(f"Overall Status: {report['status'].upper()}")
    print(f"Progressing toward ASI: {'✓ YES' if report['progressing'] else '✗ NO'}")
    print(f"\nTotal Iterations: {report['iterations']}")
    
    print("\n--- Success Criteria Evaluation ---")
    criteria = report['criteria']
    print(f"1. Improvement speed increasing: {criteria['improvement_speed_increasing']}")
    print(f"2. Simpler while more capable: {criteria['simpler_while_more_capable']}")
    print(f"3. Guidance advisory (not corrective): {criteria['guidance_advisory']}")
    print(f"4. Autonomous repair active: {criteria['autonomous_repair']}")
    print(f"\nCriteria Met: {criteria['criteria_met']} / {criteria['required']} required")
    
    print("\n--- Metrics Summary ---")
    metrics = report['metrics']
    print(f"Total Improvements: {metrics['total_improvements']}")
    print(f"Human Interventions: {metrics['total_human_interventions']}")
    print(f"Autonomous Fixes: {metrics['total_autonomous_fixes']}")
    print(f"Autonomy Ratio: {metrics['autonomy_ratio']:.1%}")
    print(f"\nComplexity: {metrics['initial_complexity']:.2f} → {metrics['current_complexity']:.2f}")
    print(f"Capability: {metrics['initial_capability']:.2f} → {metrics['current_capability']:.2f}")
    
    # Phase-specific details
    print_section("Phase-Specific Status")
    
    # Phase I: System Model
    print("PHASE I - System Self-Model")
    model_dict = program.system_model.to_dict()
    print(f"  Model Version: {model_dict['model_version']}")
    print(f"  Components: {len(model_dict['components'])}")
    print(f"  Invariants: {len(model_dict['invariants'])}")
    print(f"  Memory Pressure: {model_dict['memory']['pressure_level']}")
    
    # Phase II: Verification
    print("\nPHASE II - Self-Verification")
    verification_stats = program.verification_engine.get_verification_stats()
    print(f"  Total Verifications: {verification_stats['total_verifications']}")
    print(f"  Failure Rate: {verification_stats['failure_rate']:.1%}")
    print(f"  Regression Signatures: {verification_stats['regression_signatures']}")
    
    # Phase III: Goal Preservation
    print("\nPHASE III - Goal Preservation")
    goal_evidence = program.goal_preservation.get_evidence_of_goal_stability()
    print(f"  Architectural Changes: {goal_evidence['total_architectural_changes']}")
    print(f"  Purpose Preservation Rate: {goal_evidence['purpose_preservation_rate']:.1%}")
    
    # Phase IV: Compression
    print("\nPHASE IV - Abstraction Compression")
    compression_report = program.compression_engine.get_abstraction_report()
    if compression_report['current_metrics']:
        print(f"  Intelligence Score: {compression_report['current_metrics']['intelligence_score']:.2f}")
        print(f"  Compression Ratio: {compression_report['current_metrics']['compression_ratio']:.3f}")
    print(f"  Patterns Detected: {compression_report['pattern_stats']['total_patterns_detected']}")
    print(f"  Primitives Created: {compression_report['primitive_stats']['total_primitives']}")
    
    # Phase V: Discovery
    print("\nPHASE V - Algorithm Discovery")
    discovery_report = program.discovery_engine.get_discovery_report()
    print(f"  Execution Traces: {discovery_report['total_traces']}")
    print(f"  Insights Generated: {discovery_report['total_insights']}")
    print(f"  Novel Discoveries: {discovery_report['novel_discoveries']}")
    print(f"  Superior Discoveries: {discovery_report['superior_discoveries']}")
    
    # Phase VI: Feedback Loop
    print("\nPHASE VI - Execution Feedback")
    feedback_status = program.feedback_loop.get_feedback_loop_status()
    print(f"  Iterations: {feedback_status['iteration']}")
    print(f"  Telemetry Events: {feedback_status['telemetry_events']}")
    print(f"  Total Decisions: {feedback_status['total_decisions']}")
    print(f"  Implemented: {feedback_status['implemented_decisions']}")
    print(f"  Current Performance: {feedback_status['current_performance']:.2f}")
    print(f"  Improvement Demonstrated: {'✓ YES' if feedback_status['improvement_demonstrated'] else '✗ NO'}")
    
    # Final assessment
    print_section("Final Assessment")
    
    if report['progressing']:
        print("✓ SYSTEM IS PROGRESSING TOWARD ASI")
        print("\nThe system demonstrates:")
        print("  • Increasing improvement velocity")
        print("  • Simplification while gaining capability")
        print("  • Growing autonomy (less human intervention needed)")
        print("  • Autonomous failure detection and repair")
        print("\nThis meets the strict criteria for recursive self-improvement.")
    else:
        print("⚠ SYSTEM NOT YET PROGRESSING TOWARD ASI")
        print("\nMore iterations or adjustments needed to demonstrate:")
        print("  • Sustained improvement velocity increase")
        print("  • Consistent complexity reduction with capability growth")
        print("  • Higher autonomy ratio (>50%)")
        print("  • Demonstrated autonomous repair capability")
    
    print_section("Demonstration Complete")
    print("\nThis demonstration shows the foundation for recursive ASI development.")
    print("Each phase contributes to the system's ability to improve itself:")
    print("  • Self-model enables self-understanding")
    print("  • Verification catches regressions autonomously")
    print("  • Goal preservation prevents drift during change")
    print("  • Compression makes the system conceptually simpler")
    print("  • Discovery invents new algorithmic approaches")
    print("  • Feedback loop closes cognition-execution gap")
    print("\nSuccess = System gets better at making itself better.")


if __name__ == "__main__":
    main()
