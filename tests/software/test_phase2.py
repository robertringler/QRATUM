"""Tests for Phase II enhancements."""
from __future__ import annotations

import asyncio
import time

from quasim import Phase2Config, Phase2Runtime
from quasim.adaptive_precision import AdaptivePrecisionManager, PrecisionConfig, PrecisionMode
from quasim.async_exec.executor import AsyncExecutor, ExecutionGraph
from quasim.async_exec.pipeline import Pipeline, PipelineStage
from quasim.autotune import BayesianTuner, EnergyMonitor, TuningConfig
from quasim.hetero import DeviceType, HeteroScheduler, Workload, WorkloadType
from quasim.ir import Backend, IRBuilder, IRType, lower_to_backend
from quasim.meta_cache import CacheManager, CacheEntry, FusionEngine, KernelGraph
from quasim.verification import KernelVerifier
from quasim.visualization import BenchmarkResult, DashboardGenerator


def test_ir_builder():
    """Test IR builder functionality."""
    builder = IRBuilder()
    
    # Add some operations
    node1 = builder.add_tensor_op("add", [], dtype=IRType.FP32, shape=(100,))
    node2 = builder.add_tensor_op("mul", [node1], dtype=IRType.FP32, shape=(100,))
    node3 = builder.add_tensor_op("relu", [node2], dtype=IRType.FP32, shape=(100,))
    
    assert len(builder.nodes) == 3
    assert node2 in node1.outputs
    
    # Test optimization
    builder.optimize()
    
    # Test MLIR export
    mlir = builder.to_mlir()
    assert "module {" in mlir
    assert "add" in mlir


def test_ir_lowering():
    """Test IR lowering to different backends."""
    builder = IRBuilder()
    node1 = builder.add_tensor_op("add", [], dtype=IRType.FP32)
    node2 = builder.add_tensor_op("relu", [node1], dtype=IRType.FP32)
    
    # Test CUDA lowering
    cuda_code = lower_to_backend(builder.nodes, Backend.CUDA)
    assert "#include <cuda_runtime.h>" in cuda_code
    assert "__global__" in cuda_code
    
    # Test HIP lowering
    hip_code = lower_to_backend(builder.nodes, Backend.HIP)
    assert "#include <hip/hip_runtime.h>" in hip_code
    
    # Test Triton lowering
    triton_code = lower_to_backend(builder.nodes, Backend.TRITON)
    assert "import triton" in triton_code
    assert "@triton.jit" in triton_code
    
    # Test JAX lowering
    jax_code = lower_to_backend(builder.nodes, Backend.JAX)
    assert "import jax" in jax_code


def test_cache_manager():
    """Test meta-compilation cache."""
    cache = CacheManager()
    
    # Test hash computation
    source = "kernel void test() {}"
    hash1 = cache.compute_hash(source, "cuda")
    hash2 = cache.compute_hash(source, "cuda")
    assert hash1 == hash2
    
    # Test cache operations
    entry = CacheEntry(
        kernel_hash=hash1,
        backend="cuda",
        version="1.0.0",
        compilation_time=0.5,
        source_code=source,
        metadata={"test": "data"},
    )
    
    cache.put(entry)
    retrieved = cache.get(hash1)
    assert retrieved is not None
    assert retrieved.source_code == source
    
    # Test cache listing
    entries = cache.list_entries()
    assert len(entries) >= 1


def test_fusion_engine():
    """Test neural kernel fusion."""
    engine = FusionEngine()
    
    # Test cost estimation
    cost1 = engine.estimate_cost("matmul", size=1000)
    cost2 = engine.estimate_cost("add", size=1000)
    assert cost1 > cost2
    
    # Build a kernel graph
    graph = KernelGraph()
    node1 = graph.add_node("op1", "add")
    node2 = graph.add_node("op2", "mul", dependencies=[node1])
    node3 = graph.add_node("op3", "relu", dependencies=[node2])
    
    # Test fusion optimization
    optimized = engine.optimize_graph(graph)
    assert isinstance(optimized, list)
    
    # Test cost model update
    engine.update_cost_model("custom_op", 50.0)


def test_adaptive_precision():
    """Test adaptive precision switching."""
    config = PrecisionConfig(
        mode=PrecisionMode.FP8,
        accumulator_mode=PrecisionMode.FP32,
        tolerance=1e-5,
        auto_fallback=True,
    )
    manager = AdaptivePrecisionManager(config)
    
    # Test precision selection
    precision = manager.select_precision("add", (-1.0, 1.0))
    assert isinstance(precision, PrecisionMode)
    
    # Test quantization
    value = 3.14159
    quantized = manager.quantize(value, PrecisionMode.FP8)
    assert isinstance(quantized, float)
    
    # Test calibration
    reference = [1.0, 2.0, 3.0]
    quantized_out = [1.01, 2.01, 3.01]
    error = manager.calibrate(reference, quantized_out)
    assert error >= 0.0
    
    # Test statistics
    stats = manager.get_statistics()
    assert "current_mode" in stats


def test_async_executor():
    """Test async execution."""
    executor = AsyncExecutor(max_concurrent=2)
    
    # Create execution graph
    graph = ExecutionGraph()
    
    def task_func(x: int) -> int:
        time.sleep(0.01)
        return x * 2
    
    task1 = graph.add_task("task1", task_func, None, 1)
    task2 = graph.add_task("task2", task_func, None, 2)
    task3 = graph.add_task("task3", task_func, [task1, task2], 3)
    
    # Execute graph
    results = asyncio.run(executor.execute_graph(graph))
    
    assert results["task1"] == 2
    assert results["task2"] == 4
    assert results["task3"] == 6
    
    # Check statistics
    stats = executor.get_statistics()
    assert stats["total_tasks"] >= 3


def test_pipeline():
    """Test pipeline execution."""
    pipeline = Pipeline()
    
    # Add stages
    pipeline.add_stage("stage1", lambda x: x * 2)
    pipeline.add_stage("stage2", lambda x: x + 1)
    pipeline.add_stage("stage3", lambda x: x ** 2)
    
    # Execute pipeline
    input_data = [1, 2, 3, 4]
    results = asyncio.run(pipeline.execute(input_data))
    
    # Verify: (((x * 2) + 1) ** 2)
    expected = [((x * 2 + 1) ** 2) for x in input_data]
    assert results == expected


def test_bayesian_tuner():
    """Test Bayesian optimizer."""
    config = TuningConfig(
        name="test_kernel",
        param_ranges={
            "block_size": (32.0, 1024.0),
            "tile_size": (8.0, 64.0),
        },
        objectives=["latency"],
        max_iterations=10,
    )
    
    tuner = BayesianTuner(config)
    
    # Define objective function
    def objective(params: dict[str, float]) -> dict[str, float]:
        # Simulate kernel execution
        latency = 1000.0 / params["block_size"] + params["tile_size"] * 0.1
        return {"latency": latency}
    
    # Run tuning
    best_config = tuner.tune(objective, verbose=False)
    
    assert "block_size" in best_config
    assert "tile_size" in best_config
    
    # Check history
    history = tuner.get_optimization_history()
    assert len(history) == 10


def test_energy_monitor():
    """Test energy monitoring."""
    monitor = EnergyMonitor(backend="cuda")
    
    # Test monitoring session
    monitor.start_monitoring()
    
    # Simulate some work
    for _ in range(5):
        monitor.sample()
        time.sleep(0.01)
    
    metrics = monitor.stop_monitoring()
    
    assert metrics.power_watts > 0
    assert metrics.energy_joules > 0
    
    # Test efficiency computation
    efficiency = monitor.compute_efficiency(1000.0, 150.0)
    assert efficiency > 0


def test_hetero_scheduler():
    """Test heterogeneous scheduler."""
    scheduler = HeteroScheduler()
    
    # Register devices
    gpu = scheduler.register_device(
        DeviceType.GPU,
        compute_units=108,
        memory_gb=80.0,
        peak_gflops=19500.0,
    )
    cpu = scheduler.register_device(
        DeviceType.CPU,
        compute_units=64,
        memory_gb=256.0,
        peak_gflops=2000.0,
    )
    
    # Schedule workloads
    decision1 = scheduler.schedule(workload_size=0.1, workload_type="compute")
    assert decision1 is not None
    assert decision1.device in (gpu, cpu)
    
    decision2 = scheduler.schedule(workload_size=0.2, workload_type="compute")
    assert decision2 is not None
    
    # Get statistics
    stats = scheduler.get_statistics()
    assert stats["total_scheduled"] == 2


def test_workload_characterization():
    """Test workload characterization."""
    workload = Workload(
        name="test_workload",
        workload_type=WorkloadType.DENSE_LINEAR_ALGEBRA,
        size_gflops=500.0,
        memory_footprint_gb=2.0,
        arithmetic_intensity=50.0,
    )
    
    assert workload.is_compute_bound
    assert not workload.is_memory_bound
    
    # Test suitability scores
    cpu_score = workload.estimate_cpu_suitability()
    gpu_score = workload.estimate_gpu_suitability()
    
    assert 0.0 <= cpu_score <= 1.0
    assert 0.0 <= gpu_score <= 1.0
    
    # Get backend hint
    hint = workload.get_optimal_backend_hint()
    assert hint in ("cpu", "gpu", "tpu")


def test_kernel_verifier():
    """Test kernel verification."""
    verifier = KernelVerifier()
    
    # Test determinism verification
    def deterministic_func(x: list[int]) -> list[int]:
        return [val * 2 for val in x]
    
    result = verifier.verify_determinism(
        deterministic_func,
        ([1, 2, 3],),
        iterations=5,
    )
    assert result.passed
    
    # Test conservation law
    def conserving_func(x: list[float]) -> list[float]:
        return x  # Identity preserves sum
    
    result = verifier.verify_conservation_law(
        conserving_func,
        [1.0, 2.0, 3.0],
        conservation_property="sum",
    )
    assert result.passed


def test_dashboard_generator():
    """Test dashboard generation."""
    dashboard = DashboardGenerator()
    
    # Add benchmark results
    for i in range(5):
        result = BenchmarkResult(
            name=f"benchmark_{i}",
            latency_ms=10.0 + i,
            throughput_gflops=1000.0 + i * 100,
            energy_joules=5.0 + i * 0.5,
            efficiency_gflops_per_watt=50.0 + i * 5,
            backend="cuda",
            timestamp=time.time(),
        )
        dashboard.add_result(result)
    
    # Generate outputs
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = f"{tmpdir}/test_dashboard.html"
        json_path = f"{tmpdir}/test_benchmarks.json"
        
        dashboard.generate_html(html_path)
        dashboard.export_json(json_path)
        
        import pathlib
        assert pathlib.Path(html_path).exists()
        assert pathlib.Path(json_path).exists()


def test_phase2_runtime_integration():
    """Test integrated Phase II runtime."""
    config = Phase2Config(
        simulation_precision="fp8",
        max_workspace_mb=1024,
        enable_fusion=True,
        enable_async=False,
        enable_energy_monitoring=True,
        backend="cuda",
        target_device="gpu",
    )
    
    runtime = Phase2Runtime(config)
    
    # Test simulation
    circuit = [
        [1 + 0j, 1 + 0j, 1 + 0j, 1 + 0j],
        [1 + 0j, 0 + 0j, 0 + 0j, 1 + 0j],
    ]
    
    result = runtime.simulate(circuit)
    
    assert len(result) == len(circuit)
    assert all(isinstance(val, complex) for val in result)
    
    # Get statistics
    stats = runtime.get_statistics()
    assert stats["execution_count"] == 1
    assert "precision" in stats
    assert "hetero" in stats
    
    # Generate dashboard
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        dashboard_path = f"{tmpdir}/dashboard.html"
        runtime.generate_dashboard(dashboard_path)
        
        import pathlib
        assert pathlib.Path(dashboard_path).exists()


def test_phase2_runtime_correctness():
    """Test Phase II runtime produces correct results."""
    config = Phase2Config(
        simulation_precision="fp32",
        enable_fusion=False,
        enable_async=False,
        enable_energy_monitoring=False,
    )
    
    runtime = Phase2Runtime(config)
    
    # Simple test case
    circuit = [
        [1 + 0j, 2 + 0j, 3 + 0j],
        [4 + 0j, 5 + 0j, 6 + 0j],
    ]
    
    result = runtime.simulate(circuit)
    
    # Expected: sum of each tensor
    expected = [6 + 0j, 15 + 0j]
    
    for res, exp in zip(result, expected):
        assert abs(res - exp) < 1e-5
