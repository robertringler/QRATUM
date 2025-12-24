"""Unit tests for AION Executor module."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from aion.executor import (
    QRATUMASIExecutor,
    HardwareDetector,
    QRATUMASICompiler,
    ProofPreservingVerifier,
    CrossLanguageOptimizer,
    AdaptiveRuntimeScheduler,
    ExecutionResult,
    ExecutionMetrics,
    HardwareProfile,
    CompilationResult,
    VerificationResult,
    OptimizationResult,
    HypergraphTrace,
    ExecutionPhase,
    create_default_asi_config,
    run_full_qratum_asi_on_aion,
    generate_ascii_report,
)
from aion.sir.hypergraph import HyperGraph, GraphBuilder
from aion.sir.vertices import AIONType, HardwareAffinity, EffectKind


class TestHardwareDetector:
    """Tests for HardwareDetector class."""
    
    def test_detect_returns_profile(self) -> None:
        """Test that detect returns a valid HardwareProfile."""
        detector = HardwareDetector()
        profile = detector.detect()
        
        assert isinstance(profile, HardwareProfile)
        assert profile.cpu_cores >= 1
        assert profile.total_memory_gb > 0
    
    def test_get_devices_returns_list(self) -> None:
        """Test that get_devices returns a list of devices."""
        detector = HardwareDetector()
        devices = detector.get_devices()
        
        assert isinstance(devices, list)
        assert len(devices) > 0
        
        # Should have at least CPU and GPU (emulated)
        device_kinds = {d.kind.name for d in devices}
        assert "CPU" in device_kinds
    
    def test_hardware_profile_to_dict(self) -> None:
        """Test HardwareProfile serialization."""
        profile = HardwareProfile(
            cpu_cores=8,
            cpu_frequency_mhz=3200.0,
            total_memory_gb=32.0,
            gpu_count=1,
            gpu_name="Test GPU",
        )
        
        data = profile.to_dict()
        
        assert data["cpu_cores"] == 8
        assert data["cpu_frequency_mhz"] == 3200.0
        assert data["total_memory_gb"] == 32.0
        assert data["gpu_count"] == 1


class TestQRATUMASICompiler:
    """Tests for QRATUMASICompiler class."""
    
    def test_compile_empty_modules(self) -> None:
        """Test compiling empty module configuration."""
        compiler = QRATUMASICompiler()
        result = compiler.compile({})
        
        assert isinstance(result, CompilationResult)
        assert result.success is True
        assert result.hypergraph is not None
    
    def test_compile_neural_module(self) -> None:
        """Test compiling a neural module."""
        compiler = QRATUMASICompiler()
        modules = {
            "test_neural": {
                "type": "neural",
                "hardware": "gpu",
                "layers": 4,
                "hidden_size": 512,
                "input_shape": [512],
                "output_shape": [512],
                "work_size": 1024 * 1024,
            }
        }
        
        result = compiler.compile(modules)
        
        assert result.success is True
        assert result.vertex_count > 0
        assert result.edge_count >= 0
    
    def test_compile_quantum_module(self) -> None:
        """Test compiling a quantum emulation module."""
        compiler = QRATUMASICompiler()
        modules = {
            "test_quantum": {
                "type": "quantum",
                "hardware": "gpu",
                "qubits": 8,
                "gates": ["H", "CNOT", "RZ"],
                "input_shape": [256],
                "output_shape": [256],
                "work_size": 512 * 1024,
            }
        }
        
        result = compiler.compile(modules)
        
        assert result.success is True
        assert result.vertex_count > 0
    
    def test_compile_multiagent_module(self) -> None:
        """Test compiling a multi-agent module."""
        compiler = QRATUMASICompiler()
        modules = {
            "test_agents": {
                "type": "multiagent",
                "hardware": "cpu",
                "num_agents": 4,
                "input_shape": [1024],
                "output_shape": [1024],
                "work_size": 256 * 1024,
            }
        }
        
        result = compiler.compile(modules)
        
        assert result.success is True
        assert result.vertex_count > 0
    
    def test_compile_generates_proofs(self) -> None:
        """Test that compilation generates proof terms."""
        compiler = QRATUMASICompiler()
        modules = {
            "test_compute": {
                "type": "compute",
                "hardware": "any",
                "operations": ["transform"],
                "input_shape": [128],
                "output_shape": [128],
                "work_size": 64 * 1024,
            }
        }
        
        result = compiler.compile(modules)
        
        assert result.success is True
        assert len(result.proofs) > 0


class TestProofPreservingVerifier:
    """Tests for ProofPreservingVerifier class."""
    
    def test_verify_empty_graph(self) -> None:
        """Test verification with empty graph."""
        verifier = ProofPreservingVerifier()
        graph = HyperGraph(name="empty")
        proofs = []
        
        result = verifier.verify(graph, proofs)
        
        assert isinstance(result, VerificationResult)
    
    def test_verify_with_proofs(self) -> None:
        """Test verification with actual proofs."""
        from aion.proof.verifier import ProofTerm, ProofKind
        
        verifier = ProofPreservingVerifier()
        graph = HyperGraph(name="test")
        
        # Create a simple proof
        proof = ProofTerm(
            kind=ProofKind.MEMORY_SAFETY,
            conclusion="memory_safe(test)",
            premises=["valid_alloc"],
            evidence={"allocations": []},
        )
        
        result = verifier.verify(graph, [proof])
        
        assert isinstance(result, VerificationResult)
        assert result.verification_time >= 0
    
    def test_verification_result_summary(self) -> None:
        """Test VerificationResult summary generation."""
        result = VerificationResult(
            all_verified=True,
            memory_safety=True,
            race_freedom=True,
            deadlock_freedom=True,
            bounded_resources=True,
            verification_time=0.1,
        )
        
        summary = result.summary()
        
        assert summary["all_verified"] is True
        assert summary["properties"]["memory_safety"] is True
        assert summary["verification_time"] == 0.1


class TestCrossLanguageOptimizer:
    """Tests for CrossLanguageOptimizer class."""
    
    def test_optimize_empty_graph(self) -> None:
        """Test optimization of empty graph."""
        optimizer = CrossLanguageOptimizer()
        graph = HyperGraph(name="empty")
        
        result = optimizer.optimize(graph, [])
        
        assert isinstance(result, OptimizationResult)
        assert result.success is True
    
    def test_optimize_with_kernels(self) -> None:
        """Test optimization with kernel vertices."""
        optimizer = CrossLanguageOptimizer()
        
        builder = GraphBuilder(name="kernel_test")
        builder.kernel(
            "kernel1",
            grid=(16, 16, 1),
            block=(16, 16, 1),
            args=[],
            type_info=AIONType(kind="unit"),
            affinity=HardwareAffinity.GPU,
        )
        builder.kernel(
            "kernel2",
            grid=(16, 16, 1),
            block=(16, 16, 1),
            args=[builder.current()],
            type_info=AIONType(kind="unit"),
            affinity=HardwareAffinity.GPU,
        )
        
        graph = builder.build()
        result = optimizer.optimize(graph, [])
        
        assert result.success is True
        assert result.optimized_graph is not None


class TestAdaptiveRuntimeScheduler:
    """Tests for AdaptiveRuntimeScheduler class."""
    
    def test_schedule_empty_graph(self) -> None:
        """Test scheduling empty graph."""
        from aion.optimization.scheduler import Device, DeviceKind
        
        devices = [
            Device(id="cpu0", kind=DeviceKind.CPU, name="CPU 0"),
        ]
        scheduler = AdaptiveRuntimeScheduler(devices)
        
        graph = HyperGraph(name="empty")
        result = scheduler.schedule(graph)
        
        assert result is not None
        assert result.makespan >= 0
    
    def test_schedule_with_vertices(self) -> None:
        """Test scheduling graph with vertices."""
        from aion.optimization.scheduler import Device, DeviceKind
        
        devices = [
            Device(id="cpu0", kind=DeviceKind.CPU, name="CPU 0"),
            Device(id="gpu0", kind=DeviceKind.GPU, name="GPU 0"),
        ]
        scheduler = AdaptiveRuntimeScheduler(devices)
        
        builder = GraphBuilder(name="test")
        builder.const(42, AIONType.int())
        builder.apply("process", [builder.current()], AIONType.int())
        
        graph = builder.build()
        result = scheduler.schedule(graph)
        
        assert result is not None
        assert len(result.tasks) > 0


class TestQRATUMASIExecutor:
    """Tests for QRATUMASIExecutor class."""
    
    def test_execute_minimal_config(self) -> None:
        """Test execution with minimal configuration."""
        executor = QRATUMASIExecutor()
        config = {
            "modules": {
                "test": {
                    "type": "compute",
                    "hardware": "cpu",
                    "input_shape": [64],
                    "output_shape": [64],
                    "work_size": 1024,
                }
            }
        }
        
        result = executor.execute(config)
        
        assert isinstance(result, ExecutionResult)
        assert result.hardware_profile is not None
        assert result.compilation_result is not None
    
    def test_execute_default_config(self) -> None:
        """Test execution with default configuration."""
        executor = QRATUMASIExecutor()
        config = create_default_asi_config()
        
        result = executor.execute(config)
        
        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.total_execution_time > 0
    
    def test_execution_result_report(self) -> None:
        """Test ExecutionResult report generation."""
        result = ExecutionResult(
            success=True,
            hardware_profile=HardwareProfile(cpu_cores=4),
            compilation_result=CompilationResult(success=True, vertex_count=10),
            metrics=ExecutionMetrics(throughput_ops_per_sec=1e6),
            total_execution_time=1.0,
        )
        
        report = result.generate_report()
        
        assert report["success"] is True
        assert report["total_execution_time"] == 1.0
        assert report["hardware"]["cpu_cores"] == 4
        assert report["compilation"]["vertices"] == 10


class TestExecutionMetrics:
    """Tests for ExecutionMetrics class."""
    
    def test_metrics_to_dict(self) -> None:
        """Test ExecutionMetrics serialization."""
        metrics = ExecutionMetrics(
            throughput_ops_per_sec=1e9,
            latency_ms=10.5,
            memory_used_mb=256.0,
            cpu_utilization=0.75,
            gpu_utilization=0.90,
            tasks_completed=100,
        )
        
        data = metrics.to_dict()
        
        assert data["throughput"]["ops_per_sec"] == 1e9
        assert data["latency_ms"] == 10.5
        assert data["memory"]["used_mb"] == 256.0
        assert data["utilization"]["cpu"] == 0.75
        assert data["utilization"]["gpu"] == 0.90
        assert data["tasks"]["completed"] == 100


class TestCreateDefaultASIConfig:
    """Tests for create_default_asi_config function."""
    
    def test_creates_valid_config(self) -> None:
        """Test that default config is valid."""
        config = create_default_asi_config()
        
        assert "name" in config
        assert "modules" in config
        assert len(config["modules"]) > 0
    
    def test_contains_expected_modules(self) -> None:
        """Test that default config contains expected modules."""
        config = create_default_asi_config()
        
        expected_modules = {"q_reality", "q_mind", "q_evolve", "q_will", "q_forge"}
        actual_modules = set(config["modules"].keys())
        
        assert expected_modules.issubset(actual_modules)
    
    def test_modules_have_required_fields(self) -> None:
        """Test that modules have required configuration fields."""
        config = create_default_asi_config()
        
        for name, module in config["modules"].items():
            assert "type" in module, f"Module {name} missing 'type'"
            assert "hardware" in module, f"Module {name} missing 'hardware'"


class TestRunFullQRATUMASIOnAION:
    """Tests for run_full_qratum_asi_on_aion function."""
    
    def test_runs_with_default_config(self) -> None:
        """Test running with default configuration."""
        result = run_full_qratum_asi_on_aion()
        
        assert isinstance(result, ExecutionResult)
        assert result.success is True
    
    def test_runs_with_custom_config(self) -> None:
        """Test running with custom configuration."""
        config = {
            "name": "Custom Test",
            "modules": {
                "single_module": {
                    "type": "compute",
                    "hardware": "cpu",
                    "input_shape": [32],
                    "output_shape": [32],
                    "work_size": 512,
                }
            }
        }
        
        result = run_full_qratum_asi_on_aion(config=config)
        
        assert result.success is True


class TestGenerateASCIIReport:
    """Tests for generate_ascii_report function."""
    
    def test_generates_report_for_success(self) -> None:
        """Test report generation for successful execution."""
        result = ExecutionResult(
            success=True,
            hardware_profile=HardwareProfile(cpu_cores=4),
            compilation_result=CompilationResult(success=True, vertex_count=50),
            verification_result=VerificationResult(all_verified=True, memory_safety=True),
            optimization_result=OptimizationResult(estimated_speedup=2.0),
            metrics=ExecutionMetrics(throughput_ops_per_sec=1e6, latency_ms=5.0),
            total_execution_time=2.0,
        )
        
        report = generate_ascii_report(result)
        
        assert "SUCCESS" in report
        assert "Hardware Profile" in report
        assert "Compilation" in report
        assert "Verification" in report
        assert "Performance Metrics" in report
    
    def test_generates_report_for_failure(self) -> None:
        """Test report generation for failed execution."""
        result = ExecutionResult(
            success=False,
            errors=["Test error 1", "Test error 2"],
            total_execution_time=0.5,
        )
        
        report = generate_ascii_report(result)
        
        assert "FAILED" in report
        assert "Test error 1" in report or "Errors" in report


class TestHypergraphTrace:
    """Tests for HypergraphTrace class."""
    
    def test_trace_creation(self) -> None:
        """Test trace creation with defaults."""
        trace = HypergraphTrace()
        
        assert trace.id is not None
        assert trace.timestamp is not None
        assert trace.phase == ExecutionPhase.INITIALIZATION
    
    def test_trace_with_phase(self) -> None:
        """Test trace with specific phase."""
        trace = HypergraphTrace(phase=ExecutionPhase.COMPILATION)
        
        assert trace.phase == ExecutionPhase.COMPILATION


class TestExecutionPhase:
    """Tests for ExecutionPhase enum."""
    
    def test_all_phases_exist(self) -> None:
        """Test that all expected phases exist."""
        expected_phases = {
            "INITIALIZATION",
            "HARDWARE_DETECTION",
            "COMPILATION",
            "VERIFICATION",
            "OPTIMIZATION",
            "SCHEDULING",
            "EXECUTION",
            "BENCHMARKING",
            "REPORTING",
        }
        
        actual_phases = {p.name for p in ExecutionPhase}
        
        assert expected_phases.issubset(actual_phases)


class TestIntegration:
    """Integration tests for the full execution pipeline."""
    
    def test_full_pipeline_executes(self) -> None:
        """Test that the full pipeline executes without errors."""
        config = create_default_asi_config()
        result = run_full_qratum_asi_on_aion(config=config)
        
        assert result.success is True
        
        # Verify all phases completed
        phases = {t.phase for t in result.traces}
        assert ExecutionPhase.HARDWARE_DETECTION in phases
        assert ExecutionPhase.COMPILATION in phases
        assert ExecutionPhase.VERIFICATION in phases
        assert ExecutionPhase.OPTIMIZATION in phases
        assert ExecutionPhase.SCHEDULING in phases
        assert ExecutionPhase.EXECUTION in phases
    
    def test_metrics_are_collected(self) -> None:
        """Test that metrics are properly collected."""
        config = create_default_asi_config()
        result = run_full_qratum_asi_on_aion(config=config)
        
        assert result.metrics is not None
        assert result.metrics.tasks_completed >= 0
        assert result.metrics.latency_ms >= 0
    
    def test_proofs_are_generated(self) -> None:
        """Test that proofs are generated during compilation."""
        config = create_default_asi_config()
        result = run_full_qratum_asi_on_aion(config=config)
        
        assert result.compilation_result is not None
        assert len(result.compilation_result.proofs) > 0
