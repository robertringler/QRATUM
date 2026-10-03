"""Performance measurement for FLUXA.

What is measured
----------------
* wall-clock runtime, split into LP solve time and framework overhead
* timesteps per second and events per second
* peak resident memory (via psutil when available, else skipped)
* process CPU time and its ratio to wall time
* scaling with scenario count (1, 10, 100 scenarios)
* serial vs process-parallel execution of a Monte Carlo campaign
* the marginal cost of the QRADLE provenance layer, measured by running the
  same trajectory with per-step event emission on and off

What is *not* measured
----------------------
No GPU, HPC or QPU execution occurs. FLUXA runs on CPU only. Numbers here are
specific to this container and are not a hardware comparison.

Version: 1.0.0
"""

from __future__ import annotations

import logging
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from fluxa.config import DEFAULT_SCENARIO_CONFIG, LoadedSystem
from fluxa.engine import RunConfig, SimulationEngine
from fluxa.montecarlo import CampaignConfig, run_campaign
from fluxa.scenarios import Scenario, load_scenarios

LOGGER = logging.getLogger("fluxa.benchmark")

try:  # pragma: no cover - availability is environment-dependent
    import psutil

    _PSUTIL = True
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]
    _PSUTIL = False


def host_description() -> dict[str, Any]:
    """Identify the machine the numbers were taken on."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "logical_cpus": os.cpu_count(),
        "python_version": platform.python_version(),
        "psutil_available": _PSUTIL,
    }
    if _PSUTIL:
        info["total_memory_mb"] = round(psutil.virtual_memory().total / 1e6, 1)
        info["physical_cpus"] = psutil.cpu_count(logical=False)
    return info


def _rss_mb() -> float | None:
    if not _PSUTIL:
        return None
    return round(psutil.Process().memory_info().rss / 1e6, 2)


@dataclass
class RunBenchmark:
    """Timing for a single simulation run."""

    label: str
    scenario_id: str
    n_steps: int
    horizon_steps: int
    n_events: int
    wall_time_s: float
    solve_time_s: float
    overhead_s: float
    solve_fraction: float
    timesteps_per_second: float
    events_per_second: float
    cpu_time_s: float
    cpu_utilisation: float
    rss_before_mb: float | None
    rss_after_mb: float | None
    rss_delta_mb: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark_run(
    loaded: LoadedSystem,
    scenario: Scenario,
    *,
    label: str,
    n_steps: int = 288,
    horizon_steps: int = 12,
    emit_per_step_events: bool = True,
    checkpoint_every: int = 24,
) -> tuple[RunBenchmark, Any]:
    """Time one simulation run. Returns the benchmark and the result."""
    config = RunConfig(
        run_id=f"bench-{label}",
        n_steps=n_steps,
        horizon_steps=horizon_steps,
        checkpoint_every=checkpoint_every,
        authorized=True,
        authorizer_id="benchmark",
        emit_per_step_events=emit_per_step_events,
    )
    engine = SimulationEngine(loaded, config)

    rss_before = _rss_mb()
    cpu_before = time.process_time()
    wall_before = time.perf_counter()
    result = engine.run(scenario)
    wall = time.perf_counter() - wall_before
    cpu = time.process_time() - cpu_before
    rss_after = _rss_mb()

    bench = RunBenchmark(
        label=label,
        scenario_id=scenario.scenario_id,
        n_steps=n_steps,
        horizon_steps=horizon_steps,
        n_events=len(result.ledger),
        wall_time_s=wall,
        solve_time_s=result.solve_time_s,
        overhead_s=max(wall - result.solve_time_s, 0.0),
        solve_fraction=(result.solve_time_s / wall if wall > 0 else float("nan")),
        timesteps_per_second=(n_steps / wall if wall > 0 else float("nan")),
        events_per_second=(len(result.ledger) / wall if wall > 0 else float("nan")),
        cpu_time_s=cpu,
        cpu_utilisation=(cpu / wall if wall > 0 else float("nan")),
        rss_before_mb=rss_before,
        rss_after_mb=rss_after,
        rss_delta_mb=(
            round(rss_after - rss_before, 2)
            if (rss_before is not None and rss_after is not None)
            else None
        ),
    )
    return bench, result


@dataclass
class ScalingPoint:
    """One point on the scenario-count scaling curve."""

    n_scenarios: int
    parallel: bool
    n_workers: int
    wall_time_s: float
    seconds_per_scenario: float
    timesteps_per_second: float
    n_steps_per_scenario: int
    completed: int
    failures: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark_scaling(
    counts: tuple[int, ...] = (1, 10, 100),
    *,
    n_steps: int = 288,
    campaign_seed: int = 777_001,
    parallel: bool = False,
    n_workers: int | None = None,
) -> list[ScalingPoint]:
    """Measure wall time against scenario count using the Monte Carlo campaign.

    Sample ``k`` is independent of ``n_samples`` by construction, so the three
    points are nested subsets of the same experiment rather than three
    unrelated workloads.
    """
    points: list[ScalingPoint] = []
    for count in counts:
        config = CampaignConfig(
            campaign_seed=campaign_seed, n_samples=count, n_steps=n_steps
        )
        campaign = run_campaign(config, parallel=parallel, n_workers=n_workers)
        points.append(
            ScalingPoint(
                n_scenarios=count,
                parallel=parallel,
                n_workers=campaign.n_workers,
                wall_time_s=campaign.wall_time_s,
                seconds_per_scenario=campaign.wall_time_s / count,
                timesteps_per_second=(count * n_steps) / campaign.wall_time_s,
                n_steps_per_scenario=n_steps,
                completed=len(campaign.rows),
                failures=len(campaign.failures),
            )
        )
        LOGGER.info(
            "scaling: %d scenarios in %.2f s (%s)",
            count,
            campaign.wall_time_s,
            "parallel" if parallel else "serial",
        )
    return points


@dataclass
class ParallelComparison:
    """Serial vs parallel execution of an identical campaign."""

    n_samples: int
    n_steps: int
    serial_wall_s: float
    parallel_wall_s: float
    n_workers: int
    speedup: float
    parallel_efficiency: float
    results_identical: bool
    first_mismatch: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark_parallel(
    n_samples: int = 16, *, n_steps: int = 288, campaign_seed: int = 777_001
) -> ParallelComparison:
    """Compare serial and parallel campaign execution.

    Also verifies that parallel execution produces *identical* results, which
    is the point of determinism: a speed-up that changes the answer is not a
    speed-up. Comparison is on the per-sample provenance identity hash.
    """
    config = CampaignConfig(campaign_seed=campaign_seed, n_samples=n_samples, n_steps=n_steps)
    serial = run_campaign(config, parallel=False)
    parallel = run_campaign(config, parallel=True)

    mismatch: str | None = None
    identical = True
    for a, b in zip(serial.rows, parallel.rows, strict=False):
        if a["identity_hash"] != b["identity_hash"]:
            identical = False
            mismatch = f"sample {a['sample_index']}: {a['identity_hash'][:12]} != {b['identity_hash'][:12]}"
            break
    if len(serial.rows) != len(parallel.rows):
        identical = False
        mismatch = f"row count {len(serial.rows)} != {len(parallel.rows)}"

    speedup = (
        serial.wall_time_s / parallel.wall_time_s if parallel.wall_time_s > 0 else float("nan")
    )
    return ParallelComparison(
        n_samples=n_samples,
        n_steps=n_steps,
        serial_wall_s=serial.wall_time_s,
        parallel_wall_s=parallel.wall_time_s,
        n_workers=parallel.n_workers,
        speedup=speedup,
        parallel_efficiency=speedup / parallel.n_workers if parallel.n_workers else float("nan"),
        results_identical=identical,
        first_mismatch=mismatch,
    )


@dataclass
class ProvenanceOverhead:
    """Cost of the QRADLE/FLUXA auditability layer.

    ``with_events`` emits the full per-timestep event set and takes a
    checkpoint every 24 steps. ``without_events`` runs identical physics with
    event emission and checkpointing disabled. The difference is the price of
    auditability, which is the central engineering question about the
    architecture: whether the guarantees are affordable.
    """

    scenario_id: str
    n_steps: int
    with_events_wall_s: float
    without_events_wall_s: float
    with_events_n_events: int
    without_events_n_events: int
    overhead_s: float
    overhead_fraction: float
    solve_time_with_s: float
    solve_time_without_s: float
    identical_physics: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark_provenance_overhead(
    loaded: LoadedSystem, scenario: Scenario, *, n_steps: int = 288, repeats: int = 3
) -> ProvenanceOverhead:
    """Measure the overhead of event emission and checkpointing.

    Each configuration is run ``repeats`` times and the *minimum* wall time is
    taken, which is the standard estimator for a timing measurement
    contaminated only by positive noise.
    """
    with_times: list[float] = []
    without_times: list[float] = []
    with_result = without_result = None

    for _ in range(repeats):
        bench, result = benchmark_run(
            loaded, scenario, label="prov-on", n_steps=n_steps,
            emit_per_step_events=True, checkpoint_every=24,
        )
        with_times.append(bench.wall_time_s)
        with_result = (bench, result)

        bench, result = benchmark_run(
            loaded, scenario, label="prov-off", n_steps=n_steps,
            emit_per_step_events=False, checkpoint_every=0,
        )
        without_times.append(bench.wall_time_s)
        without_result = (bench, result)

    assert with_result is not None and without_result is not None
    on_wall, off_wall = min(with_times), min(without_times)
    identical = with_result[1].state_hashes == without_result[1].state_hashes

    return ProvenanceOverhead(
        scenario_id=scenario.scenario_id,
        n_steps=n_steps,
        with_events_wall_s=on_wall,
        without_events_wall_s=off_wall,
        with_events_n_events=with_result[0].n_events,
        without_events_n_events=without_result[0].n_events,
        overhead_s=on_wall - off_wall,
        overhead_fraction=(on_wall - off_wall) / off_wall if off_wall > 0 else float("nan"),
        solve_time_with_s=with_result[1].solve_time_s,
        solve_time_without_s=without_result[1].solve_time_s,
        identical_physics=identical,
    )


@dataclass
class BenchmarkSuite:
    """Everything the performance section reports."""

    host: dict[str, Any]
    runs: list[RunBenchmark] = field(default_factory=list)
    scaling_serial: list[ScalingPoint] = field(default_factory=list)
    scaling_parallel: list[ScalingPoint] = field(default_factory=list)
    parallel_comparison: ParallelComparison | None = None
    provenance_overhead: ProvenanceOverhead | None = None
    quantum_acceleration_demonstrated: bool = False
    quantum_note: str = (
        "No quantum acceleration was demonstrated in this experiment. The repository's "
        "quantum abstraction (quasim/qratum.quantum.core) is a Qiskit wrapper and reports "
        "QISKIT_AVAILABLE=False in this environment; q-substrate/src/quantum.rs contains a "
        "12-qubit deterministic state-vector simulator in Rust that was not built or "
        "invoked. All FLUXA computation ran on CPU."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "runs": [r.to_dict() for r in self.runs],
            "scaling_serial": [p.to_dict() for p in self.scaling_serial],
            "scaling_parallel": [p.to_dict() for p in self.scaling_parallel],
            "parallel_comparison": (
                self.parallel_comparison.to_dict() if self.parallel_comparison else None
            ),
            "provenance_overhead": (
                self.provenance_overhead.to_dict() if self.provenance_overhead else None
            ),
            "quantum_acceleration_demonstrated": self.quantum_acceleration_demonstrated,
            "quantum_note": self.quantum_note,
        }


def run_benchmark_suite(
    loaded: LoadedSystem,
    *,
    scaling_counts: tuple[int, ...] = (1, 10, 100),
    parallel_samples: int = 16,
    scenario_path=DEFAULT_SCENARIO_CONFIG,
) -> BenchmarkSuite:
    """Run every benchmark and return the collected suite."""
    scenarios = load_scenarios(scenario_path)
    suite = BenchmarkSuite(host=host_description())

    for label, scenario_id, kwargs in (
        ("baseline-24h-5min", "A_BASELINE", {"n_steps": 288}),
        ("compound-24h-5min", "F_COMPOUND", {"n_steps": 288}),
        ("baseline-7d-5min", "A_BASELINE", {"n_steps": 2016}),
        ("baseline-24h-15min", "A_BASELINE", {"n_steps": 96, "horizon_steps": 4}),
        ("baseline-myopic-h1", "A_BASELINE", {"n_steps": 288, "horizon_steps": 1}),
    ):
        bench, _ = benchmark_run(loaded, scenarios[scenario_id], label=label, **kwargs)
        suite.runs.append(bench)
        LOGGER.info("%s: %.2f s (%.1f steps/s)", label, bench.wall_time_s, bench.timesteps_per_second)

    suite.scaling_serial = benchmark_scaling(scaling_counts, parallel=False)
    suite.scaling_parallel = benchmark_scaling(scaling_counts, parallel=True)
    suite.parallel_comparison = benchmark_parallel(parallel_samples)
    suite.provenance_overhead = benchmark_provenance_overhead(loaded, scenarios["A_BASELINE"])
    return suite
