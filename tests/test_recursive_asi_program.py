"""Tests for QRATUM-QRADLE Recursive ASI Development Program"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from qratum_asi.core.algorithm_discovery import (
    AlgorithmDiscoveryEngine,
    ExecutionTrace,
)
from qratum_asi.core.compression import AbstractionCompressionEngine, PatternType
from qratum_asi.core.execution_feedback import ExecutionFeedbackLoop, TelemetryType
from qratum_asi.core.goal_preservation import GoalPreservationEngine
from qratum_asi.core.recursive_asi_program import RecursiveASIDevelopmentProgram
from qratum_asi.core.system_model import (
    ComponentType,
    FailureMode,
    QRATUMSystemModel,
)
from qratum_asi.core.verification import (
    GraphOperationValidator,
    SelfVerificationEngine,
    SSSPValidator,
)


class TestSystemModel:
    """Test PHASE I: System Self-Model Construction"""

    def test_system_model_initialization(self):
        """Test system model initializes correctly."""
        model = QRATUMSystemModel()

        assert len(model.invariants) > 0
        assert len(model.failure_modes) > 0
        assert model.model_version == 1

    def test_component_registration(self):
        """Test registering components."""
        model = QRATUMSystemModel()

        component = model.register_component(
            component_id="test_component",
            component_type=ComponentType.GRAPH_EXECUTOR,
            initial_state={"status": "active"},
            dependencies=[],
            invariants=["human_oversight"],
            failure_modes=[FailureMode.MEMORY_EXHAUSTION],
            performance_bounds={"max_latency_ms": 100},
        )

        assert component.component_id == "test_component"
        assert "test_component" in model.components

    def test_invariant_validation(self):
        """Test invariant validation."""
        model = QRATUMSystemModel()

        results = model.validate_all_invariants()

        assert isinstance(results, dict)
        assert len(results) > 0

    def test_failure_prediction(self):
        """Test failure mode prediction."""
        model = QRATUMSystemModel()

        # Simulate high memory pressure
        model.update_memory_model(
            total_allocated=1024 * 1024 * 1000, allocation_patterns={}  # 1 GB
        )
        model.memory_model.peak_allocated = 1024 * 1024 * 1000
        model.memory_model.pressure_level = "critical"

        predictions = model.get_failure_predictions()

        assert len(predictions) > 0
        assert any(p["failure_mode"] == FailureMode.MEMORY_EXHAUSTION for p in predictions)


class TestVerificationEngine:
    """Test PHASE II: Self-Verification Engine"""

    def test_verification_engine_initialization(self):
        """Test verification engine initializes."""
        engine = SelfVerificationEngine()

        assert len(engine.checks) > 0
        assert len(engine.containment_strategies) > 0

    def test_sssp_correctness_validation(self):
        """Test SSSP correctness validator."""
        graph = {
            "nodes": [0, 1, 2],
            "edges": [{"from": 0, "to": 1, "weight": 1.0}, {"from": 1, "to": 2, "weight": 1.0}],
        }
        distances = {0: 0.0, 1: 1.0, 2: 2.0}
        predecessors = {0: None, 1: 0, 2: 1}

        result = SSSPValidator.validate_correctness(graph, 0, distances, predecessors)

        assert result is True

    def test_sssp_baseline_comparison(self):
        """Test SSSP baseline comparison."""
        test_distances = {0: 0.0, 1: 1.0, 2: 2.0}
        baseline_distances = {0: 0.0, 1: 1.0, 2: 2.0}

        result = SSSPValidator.compare_with_baseline(test_distances, baseline_distances)

        assert result is True

    def test_graph_structure_validation(self):
        """Test graph structure validator."""
        valid_graph = {"nodes": [0, 1, 2], "edges": [{"from": 0, "to": 1, "weight": 1.0}]}

        result = GraphOperationValidator.validate_graph_structure(valid_graph)

        assert result is True

    def test_regression_detection(self):
        """Test intent-based regression detection."""
        engine = SelfVerificationEngine()

        # First behavior
        regression = engine.detect_regression(
            intent="compute_shortest_paths",
            current_behavior={"correctness": True, "performance": 1.0},
        )

        assert regression is False  # First time, no regression

        # Changed behavior
        regression = engine.detect_regression(
            intent="compute_shortest_paths",
            current_behavior={"correctness": False, "performance": 0.5},
        )

        assert regression is True  # Behavior changed significantly


class TestGoalPreservation:
    """Test PHASE III: Goal Preservation Under Change"""

    def test_goal_preservation_initialization(self):
        """Test goal preservation engine initializes."""
        engine = GoalPreservationEngine()

        assert len(engine.goals) > 0
        assert len(engine.constraints) > 0

    def test_constraint_has_rationale(self):
        """Test constraints have rationales."""
        engine = GoalPreservationEngine()

        for constraint in engine.constraints.values():
            assert constraint.rationale is not None
            assert len(constraint.rationale.reason) > 0
            assert len(constraint.rationale.consequences_if_violated) > 0

    def test_goal_preservation_testing(self):
        """Test goal preservation across changes."""
        engine = GoalPreservationEngine()

        state_before = {
            "human_oversight_active": True,
            "rollback_available": True,
            "safety_constraints_enforced": True,
            "safety_mechanisms": ["authorization", "rollback"],
        }

        state_after = {
            "human_oversight_active": True,
            "rollback_available": True,
            "safety_constraints_enforced": True,
            "safety_mechanisms": ["authorization", "rollback"],
        }

        result = engine.test_goal_preservation("safety", state_before, state_after)

        assert result["preserved"] is True

    def test_architectural_change_recording(self):
        """Test recording architectural changes."""
        engine = GoalPreservationEngine()

        state_before = {"implementation": {"version": 1}, "purpose": {"goal": "safety"}}

        state_after = {"implementation": {"version": 2}, "purpose": {"goal": "safety"}}

        change = engine.record_architectural_change(
            change_id="test_change",
            description="Test change",
            affected_components=["component1"],
            state_before=state_before,
            state_after=state_after,
        )

        assert change.preserves_purpose() is True


class TestCompressionEngine:
    """Test PHASE IV: Abstraction Compression Engine"""

    def test_compression_engine_initialization(self):
        """Test compression engine initializes."""
        engine = AbstractionCompressionEngine()

        assert isinstance(engine.patterns, dict)
        assert isinstance(engine.primitives, dict)

    def test_pattern_detection(self):
        """Test detecting repeated patterns."""
        engine = AbstractionCompressionEngine()

        codebase_analysis = {
            "algorithms": {
                "algo1": {"operations": ["loop", "compare", "update"]},
                "algo2": {"operations": ["loop", "compare", "update"]},
                "algo3": {"operations": ["loop", "compare", "update"]},
            },
            "data_structures": {},
            "control_flows": {},
        }

        patterns = engine.detect_patterns(codebase_analysis)

        assert len(patterns) > 0
        assert any(p.pattern_type == PatternType.ALGORITHM for p in patterns)

    def test_abstraction_proposal(self):
        """Test proposing abstraction."""
        engine = AbstractionCompressionEngine()

        # First detect patterns
        codebase_analysis = {
            "algorithms": {
                "algo1": {"operations": ["loop", "compare"]},
                "algo2": {"operations": ["loop", "compare"]},
            },
            "data_structures": {},
            "control_flows": {},
        }

        patterns = engine.detect_patterns(codebase_analysis)
        pattern_ids = [p.pattern_id for p in patterns]

        if pattern_ids:
            primitive = engine.propose_abstraction(
                pattern_ids=pattern_ids[:1],
                primitive_name="IterativeComparison",
                primitive_description="Abstract iterative comparison pattern",
                primitive_complexity=2.0,
            )

            assert primitive.name == "IterativeComparison"
            assert primitive.primitive_id in engine.primitives

    def test_intelligence_measurement(self):
        """Test measuring intelligence growth."""
        engine = AbstractionCompressionEngine()

        metrics1 = engine.compute_metrics(system_behavior_coverage=1.0, system_performance=1.0)
        metrics2 = engine.compute_metrics(system_behavior_coverage=1.0, system_performance=1.1)

        growth = engine.measure_intelligence_growth()

        assert "intelligence_growth" in growth
        assert "compression_improvement" in growth


class TestAlgorithmDiscovery:
    """Test PHASE V: Autonomous Algorithm Discovery"""

    def test_discovery_engine_initialization(self):
        """Test discovery engine initializes."""
        engine = AlgorithmDiscoveryEngine()

        assert isinstance(engine.execution_traces, dict)
        assert isinstance(engine.discoveries, dict)

    def test_execution_trace_recording(self):
        """Test recording execution traces."""
        engine = AlgorithmDiscoveryEngine()

        trace = ExecutionTrace(
            trace_id="test_trace",
            algorithm_name="dijkstra",
            input_size=100,
            execution_time=0.01,
            memory_used=1024,
            operations_performed=["heap_pop"] * 100,
            wasted_operations=["redundant"] * 10,
            bottlenecks=["heap_operations"],
        )

        engine.record_execution_trace(trace)

        assert "test_trace" in engine.execution_traces
        efficiency = trace.get_efficiency_score()
        assert 0.0 <= efficiency <= 1.0

    def test_wasted_work_analysis(self):
        """Test identifying wasted work."""
        engine = AlgorithmDiscoveryEngine()

        trace = ExecutionTrace(
            trace_id="test_trace",
            algorithm_name="test_algo",
            input_size=100,
            execution_time=0.1,
            memory_used=1024,
            operations_performed=["op1"] * 100,
            wasted_operations=["op1"] * 30,
            bottlenecks=[],
        )

        engine.record_execution_trace(trace)
        analyses = engine.analyze_wasted_work()

        assert len(analyses) > 0
        assert analyses[0]["algorithm"] == "test_algo"

    def test_insight_generation(self):
        """Test generating algorithmic insights."""
        engine = AlgorithmDiscoveryEngine()

        # Record inefficient trace
        trace = ExecutionTrace(
            trace_id="test_trace",
            algorithm_name="inefficient_algo",
            input_size=100,
            execution_time=1.0,
            memory_used=1024,
            operations_performed=["op"] * 100,
            wasted_operations=["op"] * 50,  # 50% waste
            bottlenecks=[],
        )

        engine.record_execution_trace(trace)
        insights = engine.generate_insights()

        assert len(insights) > 0


class TestExecutionFeedback:
    """Test PHASE VI: Cognition ↔ Execution Feedback Loop"""

    def test_feedback_loop_initialization(self):
        """Test feedback loop initializes."""
        loop = ExecutionFeedbackLoop()

        assert loop.iteration == 0
        assert isinstance(loop.telemetry_collector, object)

    def test_telemetry_recording(self):
        """Test recording telemetry."""
        loop = ExecutionFeedbackLoop()

        loop.record_telemetry(TelemetryType.LATENCY, 50.0, "test_component")

        assert len(loop.telemetry_collector.events) > 0

    def test_feedback_iteration(self):
        """Test running feedback iteration."""
        loop = ExecutionFeedbackLoop()

        # Record some telemetry
        components = ["comp1", "comp2"]
        for comp in components:
            loop.record_telemetry(TelemetryType.LATENCY, 100.0, comp)
            loop.record_telemetry(TelemetryType.CACHE_MISS, 0.3, comp)
            loop.record_telemetry(TelemetryType.MEMORY_PRESSURE, 0.5, comp)

        results = loop.run_feedback_iteration(components)

        assert results["iteration"] == 1
        assert "profiles" in results
        assert "bottlenecks" in results

    def test_improvement_demonstration(self):
        """Test demonstrating improvement."""
        loop = ExecutionFeedbackLoop()

        # First iteration
        loop.run_feedback_iteration(["comp1"])

        # Second iteration (should show no improvement yet)
        loop.run_feedback_iteration(["comp1"])

        evidence = loop.demonstrate_improvement()

        assert "improved" in evidence
        assert "evidence" in evidence


class TestRecursiveASIProgram:
    """Test integration of all phases"""

    def test_program_initialization(self):
        """Test recursive ASI program initializes."""
        program = RecursiveASIDevelopmentProgram()

        assert program.system_model is not None
        assert program.verification_engine is not None
        assert program.goal_preservation is not None
        assert program.compression_engine is not None
        assert program.discovery_engine is not None
        assert program.feedback_loop is not None

    def test_recursive_iteration(self):
        """Test running recursive iteration."""
        program = RecursiveASIDevelopmentProgram()

        results = program.run_recursive_iteration()

        assert results["iteration"] == 1
        assert "improvements" in results
        assert "system_state" in results
        assert "autonomy" in results

    def test_asi_progress_evaluation(self):
        """Test ASI progress evaluation."""
        program = RecursiveASIDevelopmentProgram()

        # Run a few iterations
        for _ in range(3):
            program.run_recursive_iteration()

        report = program.get_asi_progress_report()

        assert "status" in report
        assert "progressing" in report
        assert "criteria" in report
        assert "metrics" in report

    def test_success_criteria_tracking(self):
        """Test that success criteria are tracked."""
        program = RecursiveASIDevelopmentProgram()

        # Run iterations
        program.run_recursive_iteration()
        program.run_recursive_iteration()

        report = program.get_asi_progress_report()
        criteria = report["criteria"]

        # Check all 4 criteria are evaluated
        assert "improvement_speed_increasing" in criteria
        assert "simpler_while_more_capable" in criteria
        assert "guidance_advisory" in criteria
        assert "autonomous_repair" in criteria


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
