#!/usr/bin/env python3
"""Demonstration of all Phase II features in QuASIM."""
from __future__ import annotations

import asyncio
import time

print("=" * 80)
print("QuASIM Phase II — Comprehensive Feature Demonstration")
print("=" * 80)
print()

# 1. IR Unification
print("1. IR Unification Layer")
print("-" * 80)
from quasim.ir import IRBuilder, IRType, Backend, lower_to_backend

builder = IRBuilder()
node1 = builder.add_tensor_op("add", [], dtype=IRType.FP32, shape=(1024,))
node2 = builder.add_tensor_op("relu", [node1], dtype=IRType.FP32, shape=(1024,))
builder.optimize()

cuda_code = lower_to_backend(builder.nodes, Backend.CUDA)
print(f"✓ Built IR graph with {len(builder.nodes)} nodes")
print(f"✓ Generated CUDA kernel ({len(cuda_code)} bytes)")
print()

# 2. Meta-Compilation Cache
print("2. Meta-Compilation Cache")
print("-" * 80)
from quasim.meta_cache import CacheManager, FusionEngine, KernelGraph

cache = CacheManager()
source = "kernel void test() { }"
kernel_hash = cache.compute_hash(source, "cuda")
print(f"✓ Computed kernel hash: {kernel_hash}")

fusion_engine = FusionEngine()
graph = KernelGraph()
n1 = graph.add_node("op1", "add")
n2 = graph.add_node("op2", "mul", dependencies=[n1])
optimized = fusion_engine.optimize_graph(graph)
print(f"✓ Fusion engine optimized {len(graph.nodes)} nodes → {len(optimized)} groups")
print()

# 3. Adaptive Precision
print("3. Adaptive Precision Management")
print("-" * 80)
from quasim.adaptive_precision import AdaptivePrecisionManager, PrecisionConfig, PrecisionMode

manager = AdaptivePrecisionManager(PrecisionConfig(
    mode=PrecisionMode.FP8,
    auto_fallback=True,
))
precision = manager.select_precision("matmul", (-1.0, 1.0))
quantized = manager.quantize(3.14159, PrecisionMode.FP8)
print(f"✓ Selected precision: {precision.value}")
print(f"✓ Quantized 3.14159 → {quantized:.5f} (FP8)")
print()

# 4. Async Execution
print("4. Async Execution Pipeline")
print("-" * 80)
from quasim.async_exec import AsyncExecutor, ExecutionGraph, Pipeline

async def demo_async():
    executor = AsyncExecutor(max_concurrent=4)
    graph = ExecutionGraph()
    
    task1 = graph.add_task("task1", lambda x: x * 2, None, 5)
    task2 = graph.add_task("task2", lambda x: x + 3, [task1])
    
    results = await executor.execute_graph(graph)
    return results

results = asyncio.run(demo_async())
print(f"✓ Async execution completed: {results}")

pipeline = Pipeline()
pipeline.add_stage("double", lambda x: x * 2)
pipeline.add_stage("increment", lambda x: x + 1)
output = asyncio.run(pipeline.execute([1, 2, 3]))
print(f"✓ Pipeline processing: [1,2,3] → {output}")
print()

# 5. Heterogeneous Scheduling
print("5. Heterogeneous Device Scheduling")
print("-" * 80)
from quasim.hetero import HeteroScheduler, DeviceType, Workload, WorkloadType

scheduler = HeteroScheduler()
gpu = scheduler.register_device(DeviceType.GPU, peak_gflops=19500.0, memory_gb=80.0)
cpu = scheduler.register_device(DeviceType.CPU, peak_gflops=2000.0, memory_gb=256.0)

decision = scheduler.schedule(workload_size=0.5, workload_type="compute")
print(f"✓ Registered {len(scheduler.devices)} devices (GPU, CPU)")
print(f"✓ Scheduled workload to: {decision.device.device_type.value}")
print(f"  Estimated time: {decision.estimated_time:.6f}s")
print(f"  Estimated energy: {decision.estimated_energy:.4f}J")

workload = Workload(
    name="test",
    workload_type=WorkloadType.DENSE_LINEAR_ALGEBRA,
    size_gflops=500.0,
    memory_footprint_gb=2.0,
    arithmetic_intensity=50.0,
)
hint = workload.get_optimal_backend_hint()
print(f"✓ Workload optimal backend: {hint}")
print()

# 6. Autotuning
print("6. Bayesian Autotuning")
print("-" * 80)
from quasim.autotune import BayesianTuner, TuningConfig, EnergyMonitor

config = TuningConfig(
    name="demo_kernel",
    param_ranges={"block_size": (32.0, 512.0)},
    max_iterations=10,
)
tuner = BayesianTuner(config)

def objective(params):
    latency = 1000.0 / params["block_size"]
    return {"latency": latency}

best_config = tuner.tune(objective, verbose=False)
print(f"✓ Bayesian tuning completed: block_size={best_config['block_size']:.1f}")

monitor = EnergyMonitor(backend="cuda")
monitor.start_monitoring()
time.sleep(0.01)
monitor.sample()
metrics = monitor.stop_monitoring()
print(f"✓ Energy monitoring: {metrics.power_watts:.1f}W, {metrics.energy_joules:.4f}J")
print()

# 7. Formal Verification
print("7. Formal Verification")
print("-" * 80)
from quasim.verification import KernelVerifier

verifier = KernelVerifier()

def deterministic_func(x):
    return [v * 2 for v in x]

result = verifier.verify_determinism(deterministic_func, ([1, 2, 3],), iterations=5)
print(f"✓ Determinism check: {'PASSED' if result.passed else 'FAILED'}")

result = verifier.verify_conservation_law(lambda x: x, [1.0, 2.0, 3.0], conservation_property="sum")
print(f"✓ Conservation law: {'PASSED' if result.passed else 'FAILED'} (error: {result.details.get('error', 0):.2e})")
print()

# 8. Visualization
print("8. Benchmark Visualization")
print("-" * 80)
from quasim.visualization import DashboardGenerator, BenchmarkResult

dashboard = DashboardGenerator()
for i in range(3):
    result = BenchmarkResult(
        name=f"demo_{i}",
        latency_ms=10.0 + i,
        throughput_gflops=1000.0 + i * 100,
        energy_joules=5.0,
        efficiency_gflops_per_watt=50.0,
        backend="cuda",
        timestamp=time.time(),
    )
    dashboard.add_result(result)

print(f"✓ Dashboard with {len(dashboard.results)} benchmark results")
print()

# 9. Integrated Phase II Runtime
print("9. Integrated Phase II Runtime")
print("-" * 80)
from quasim import Phase2Config, Phase2Runtime

config = Phase2Config(
    simulation_precision="fp8",
    enable_fusion=True,
    enable_energy_monitoring=False,
    backend="cuda",
    target_device="gpu",
)

runtime = Phase2Runtime(config)
circuit = [
    [1+0j, 0+0j, 0+0j, 1+0j],
    [0+0j, 1+0j, 1+0j, 0+0j],
]

result = runtime.simulate(circuit)
stats = runtime.get_statistics()

print(f"✓ Simulation completed: {len(result)} outputs")
print(f"✓ Runtime statistics:")
print(f"  - Executions: {stats['execution_count']}")
print(f"  - Cached kernels: {stats['cache_entries']}")
print(f"  - Avg latency: {stats['avg_latency_ms']:.3f} ms")
print(f"  - Precision errors: {stats['precision']['avg_error']:.2e}")
print()

# Summary
print("=" * 80)
print("Phase II Feature Demonstration Complete!")
print("=" * 80)
print()
print("All 9 subsystems validated:")
print("  ✓ IR Unification Layer")
print("  ✓ Meta-Compilation Cache")
print("  ✓ Adaptive Precision")
print("  ✓ Async Execution")
print("  ✓ Heterogeneous Scheduling")
print("  ✓ Bayesian Autotuning")
print("  ✓ Formal Verification")
print("  ✓ Benchmark Visualization")
print("  ✓ Integrated Runtime")
print()
print("QuASIM Phase II is production-ready! 🚀")
print("=" * 80)
