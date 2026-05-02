"""Simple test runner for Recursive ASI Program (without pytest dependency)"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qratum_asi.core.algorithm_discovery import AlgorithmDiscoveryEngine, ExecutionTrace
from qratum_asi.core.compression import AbstractionCompressionEngine
from qratum_asi.core.execution_feedback import ExecutionFeedbackLoop, TelemetryType
from qratum_asi.core.goal_preservation import GoalPreservationEngine
from qratum_asi.core.recursive_asi_program import RecursiveASIDevelopmentProgram
from qratum_asi.core.system_model import ComponentType, FailureMode, QRATUMSystemModel
from qratum_asi.core.verification import SelfVerificationEngine, SSSPValidator


def run_test(test_name, test_func):
    """Run a test and report results."""
    try:
        test_func()
        print(f"✓ {test_name}")
        return True
    except AssertionError as e:
        print(f"✗ {test_name}: {e}")
        return False
    except Exception as e:
        print(f"✗ {test_name}: ERROR - {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print(" Running Recursive ASI Program Tests")
    print("=" * 80)

    tests_passed = 0
    tests_failed = 0

    # Test Phase I: System Model
    print("\n--- PHASE I: System Self-Model Tests ---")

    def test_system_model_init():
        model = QRATUMSystemModel()
        assert len(model.invariants) > 0
        assert len(model.failure_modes) > 0

    if run_test("System model initialization", test_system_model_init):
        tests_passed += 1
    else:
        tests_failed += 1

    def test_component_registration():
        model = QRATUMSystemModel()
        component = model.register_component(
            component_id="test_comp",
            component_type=ComponentType.GRAPH_EXECUTOR,
            initial_state={"status": "active"},
            dependencies=[],
            invariants=["human_oversight"],
            failure_modes=[FailureMode.MEMORY_EXHAUSTION],
            performance_bounds={"max_latency_ms": 100},
        )
        assert component.component_id == "test_comp"

    if run_test("Component registration", test_component_registration):
        tests_passed += 1
    else:
        tests_failed += 1

    # Test Phase II: Verification
    print("\n--- PHASE II: Self-Verification Tests ---")

    def test_verification_init():
        engine = SelfVerificationEngine()
        assert len(engine.checks) > 0

    if run_test("Verification engine initialization", test_verification_init):
        tests_passed += 1
    else:
        tests_failed += 1

    def test_sssp_validation():
        graph = {
            "nodes": [0, 1, 2],
            "edges": [{"from": 0, "to": 1, "weight": 1.0}, {"from": 1, "to": 2, "weight": 1.0}],
        }
        distances = {0: 0.0, 1: 1.0, 2: 2.0}
        predecessors = {0: None, 1: 0, 2: 1}
        result = SSSPValidator.validate_correctness(graph, 0, distances, predecessors)
        assert result is True

    if run_test("SSSP correctness validation", test_sssp_validation):
        tests_passed += 1
    else:
        tests_failed += 1

    # Test Phase III: Goal Preservation
    print("\n--- PHASE III: Goal Preservation Tests ---")

    def test_goal_preservation_init():
        engine = GoalPreservationEngine()
        assert len(engine.goals) > 0
        assert len(engine.constraints) > 0

    if run_test("Goal preservation initialization", test_goal_preservation_init):
        tests_passed += 1
    else:
        tests_failed += 1

    def test_goal_preservation():
        engine = GoalPreservationEngine()
        state = {
            "human_oversight_active": True,
            "rollback_available": True,
            "safety_constraints_enforced": True,
            "safety_mechanisms": ["auth", "rollback"],
        }
        result = engine.test_goal_preservation("safety", state, state)
        assert result["preserved"] is True

    if run_test("Goal preservation testing", test_goal_preservation):
        tests_passed += 1
    else:
        tests_failed += 1

    # Test Phase IV: Compression
    print("\n--- PHASE IV: Abstraction Compression Tests ---")

    def test_compression_init():
        engine = AbstractionCompressionEngine()
        assert isinstance(engine.patterns, dict)

    if run_test("Compression engine initialization", test_compression_init):
        tests_passed += 1
    else:
        tests_failed += 1

    def test_pattern_detection():
        engine = AbstractionCompressionEngine()
        analysis = {
            "algorithms": {f"algo{i}": {"operations": ["loop", "compare"]} for i in range(3)},
            "data_structures": {},
            "control_flows": {},
        }
        patterns = engine.detect_patterns(analysis)
        assert len(patterns) > 0

    if run_test("Pattern detection", test_pattern_detection):
        tests_passed += 1
    else:
        tests_failed += 1

    # Test Phase V: Algorithm Discovery
    print("\n--- PHASE V: Algorithm Discovery Tests ---")

    def test_discovery_init():
        engine = AlgorithmDiscoveryEngine()
        assert isinstance(engine.execution_traces, dict)

    if run_test("Discovery engine initialization", test_discovery_init):
        tests_passed += 1
    else:
        tests_failed += 1

    def test_trace_recording():
        engine = AlgorithmDiscoveryEngine()
        trace = ExecutionTrace(
            trace_id="test",
            algorithm_name="dijkstra",
            input_size=100,
            execution_time=0.01,
            memory_used=1024,
            operations_performed=["op"] * 100,
            wasted_operations=["op"] * 10,
            bottlenecks=[],
        )
        engine.record_execution_trace(trace)
        assert "test" in engine.execution_traces

    if run_test("Execution trace recording", test_trace_recording):
        tests_passed += 1
    else:
        tests_failed += 1

    # Test Phase VI: Execution Feedback
    print("\n--- PHASE VI: Execution Feedback Tests ---")

    def test_feedback_init():
        loop = ExecutionFeedbackLoop()
        assert loop.iteration == 0

    if run_test("Feedback loop initialization", test_feedback_init):
        tests_passed += 1
    else:
        tests_failed += 1

    def test_telemetry():
        loop = ExecutionFeedbackLoop()
        loop.record_telemetry(TelemetryType.LATENCY, 50.0, "test_comp")
        assert len(loop.telemetry_collector.events) > 0

    if run_test("Telemetry recording", test_telemetry):
        tests_passed += 1
    else:
        tests_failed += 1

    # Test Integration
    print("\n--- Integration Tests ---")

    def test_program_init():
        program = RecursiveASIDevelopmentProgram()
        assert program.system_model is not None
        assert program.verification_engine is not None
        assert program.goal_preservation is not None
        assert program.compression_engine is not None
        assert program.discovery_engine is not None
        assert program.feedback_loop is not None

    if run_test("Recursive ASI program initialization", test_program_init):
        tests_passed += 1
    else:
        tests_failed += 1

    def test_recursive_iteration():
        program = RecursiveASIDevelopmentProgram()
        results = program.run_recursive_iteration()
        assert results["iteration"] == 1
        assert "improvements" in results

    if run_test("Recursive iteration execution", test_recursive_iteration):
        tests_passed += 1
    else:
        tests_failed += 1

    def test_asi_progress():
        program = RecursiveASIDevelopmentProgram()
        for _ in range(3):
            program.run_recursive_iteration()
        report = program.get_asi_progress_report()
        assert "status" in report
        assert "progressing" in report

    if run_test("ASI progress evaluation", test_asi_progress):
        tests_passed += 1
    else:
        tests_failed += 1

    # Summary
    print("\n" + "=" * 80)
    print(" Test Summary")
    print("=" * 80)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    print(f"Total: {tests_passed + tests_failed}")

    if tests_failed == 0:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n✗ {tests_failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
