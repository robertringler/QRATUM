"""
Planetary Simulation Module

Comprehensive simulation system for planetary-scale infrastructure with
visualization, metrics collection, and scenario management.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ScenarioType(Enum):
    """Simulation scenario types."""

    GLOBAL_ADOPTION = "global_adoption"
    SECTOR_EXPANSION = "sector_expansion"
    STRESS_TEST = "stress_test"
    DISASTER_RECOVERY = "disaster_recovery"
    ECONOMIC_SHOCK = "economic_shock"
    SCALING_TEST = "scaling_test"


class SimulationStatus(Enum):
    """Simulation run status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SimulationScenario:
    """Defines a simulation scenario.

    Attributes:
        scenario_id: Unique scenario identifier
        name: Scenario name
        scenario_type: Type of scenario
        description: Scenario description
        parameters: Scenario parameters
        duration_ticks: Simulation duration in ticks
        created_at: Scenario creation timestamp
    """

    scenario_id: str
    name: str
    scenario_type: ScenarioType
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    duration_ticks: int = 1000
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialize scenario to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "scenario_type": self.scenario_type.value,
            "description": self.description,
            "parameters": self.parameters,
            "duration_ticks": self.duration_ticks,
            "created_at": self.created_at,
        }


@dataclass
class SimulationMetric:
    """A single simulation metric data point.

    Attributes:
        metric_name: Name of the metric
        tick: Simulation tick
        value: Metric value
        metadata: Additional metadata
    """

    metric_name: str
    tick: int
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metric to dictionary."""
        return {
            "metric_name": self.metric_name,
            "tick": self.tick,
            "value": self.value,
            "metadata": self.metadata,
        }


@dataclass
class MetricsCollector:
    """Collects and aggregates simulation metrics.

    Attributes:
        collector_id: Unique collector identifier
        metrics: Dictionary of metric time series
        aggregations: Pre-computed aggregations
    """

    collector_id: str
    metrics: dict[str, list[SimulationMetric]] = field(default_factory=dict)
    aggregations: dict[str, dict[str, float]] = field(default_factory=dict)

    def record(self, metric_name: str, tick: int, value: float) -> None:
        """Record a metric value.

        Args:
            metric_name: Name of the metric
            tick: Simulation tick
            value: Metric value
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []

        metric = SimulationMetric(
            metric_name=metric_name,
            tick=tick,
            value=value,
        )
        self.metrics[metric_name].append(metric)

    def get_time_series(self, metric_name: str) -> list[tuple[int, float]]:
        """Get time series for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            List of (tick, value) tuples
        """
        if metric_name not in self.metrics:
            return []

        return [(m.tick, m.value) for m in self.metrics[metric_name]]

    def compute_aggregations(self) -> dict[str, dict[str, float]]:
        """Compute aggregations for all metrics.

        Returns:
            Aggregations by metric name
        """
        for metric_name, values in self.metrics.items():
            if not values:
                continue

            vals = [v.value for v in values]
            self.aggregations[metric_name] = {
                "min": min(vals),
                "max": max(vals),
                "mean": sum(vals) / len(vals),
                "first": vals[0],
                "last": vals[-1],
                "count": len(vals),
            }

        return self.aggregations

    def to_dict(self) -> dict[str, Any]:
        """Serialize collector to dictionary."""
        return {
            "collector_id": self.collector_id,
            "metric_names": list(self.metrics.keys()),
            "total_data_points": sum(len(v) for v in self.metrics.values()),
            "aggregations": self.compute_aggregations(),
        }


@dataclass
class VisualizationData:
    """Data for visualization rendering.

    Attributes:
        viz_type: Type of visualization
        title: Visualization title
        data: Visualization data
        config: Visualization configuration
    """

    viz_type: str
    title: str
    data: dict[str, Any]
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize visualization data to dictionary."""
        return {
            "viz_type": self.viz_type,
            "title": self.title,
            "data": self.data,
            "config": self.config,
        }


class VisualizationEngine:
    """Engine for generating visualizations from simulation data.

    Attributes:
        engine_id: Unique engine identifier
        visualizations: Generated visualizations
    """

    def __init__(self, engine_id: str | None = None) -> None:
        """Initialize visualization engine.

        Args:
            engine_id: Optional engine ID
        """
        self.engine_id = (
            engine_id or f"viz_{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:8]}"
        )
        self.visualizations: list[VisualizationData] = []

    def create_time_series_chart(
        self,
        title: str,
        metrics: MetricsCollector,
        metric_names: list[str],
    ) -> VisualizationData:
        """Create a time series chart.

        Args:
            title: Chart title
            metrics: Metrics collector with data
            metric_names: Names of metrics to plot

        Returns:
            Visualization data
        """
        series_data = {}
        for name in metric_names:
            series_data[name] = metrics.get_time_series(name)

        viz = VisualizationData(
            viz_type="time_series",
            title=title,
            data={"series": series_data},
            config={
                "x_axis": "tick",
                "y_axis": "value",
                "legend": metric_names,
            },
        )
        self.visualizations.append(viz)
        return viz

    def create_network_topology(
        self,
        title: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> VisualizationData:
        """Create a network topology visualization.

        Args:
            title: Visualization title
            nodes: List of node definitions
            edges: List of edge definitions

        Returns:
            Visualization data
        """
        viz = VisualizationData(
            viz_type="network_topology",
            title=title,
            data={
                "nodes": nodes,
                "edges": edges,
            },
            config={
                "layout": "force_directed",
                "node_size": "capacity",
                "edge_color": "throughput",
            },
        )
        self.visualizations.append(viz)
        return viz

    def create_heatmap(
        self,
        title: str,
        data: list[list[float]],
        x_labels: list[str],
        y_labels: list[str],
    ) -> VisualizationData:
        """Create a heatmap visualization.

        Args:
            title: Visualization title
            data: 2D data array
            x_labels: X-axis labels
            y_labels: Y-axis labels

        Returns:
            Visualization data
        """
        viz = VisualizationData(
            viz_type="heatmap",
            title=title,
            data={
                "matrix": data,
                "x_labels": x_labels,
                "y_labels": y_labels,
            },
            config={
                "color_scale": "viridis",
                "show_values": True,
            },
        )
        self.visualizations.append(viz)
        return viz

    def create_geographic_map(
        self,
        title: str,
        points: list[dict[str, Any]],
    ) -> VisualizationData:
        """Create a geographic map visualization.

        Args:
            title: Visualization title
            points: List of geographic points with lat/lon

        Returns:
            Visualization data
        """
        viz = VisualizationData(
            viz_type="geographic_map",
            title=title,
            data={"points": points},
            config={
                "projection": "mercator",
                "show_labels": True,
                "cluster_nearby": True,
            },
        )
        self.visualizations.append(viz)
        return viz

    def create_dashboard(
        self,
        title: str,
        metrics: dict[str, float],
        charts: list[str],
    ) -> VisualizationData:
        """Create a dashboard visualization.

        Args:
            title: Dashboard title
            metrics: Key metrics to display
            charts: List of chart IDs to include

        Returns:
            Visualization data
        """
        viz = VisualizationData(
            viz_type="dashboard",
            title=title,
            data={
                "metrics": metrics,
                "charts": charts,
            },
            config={
                "layout": "grid",
                "refresh_interval": 60,
            },
        )
        self.visualizations.append(viz)
        return viz

    def export_to_json(self) -> str:
        """Export all visualizations to JSON.

        Returns:
            JSON string of all visualizations
        """
        return json.dumps(
            {
                "engine_id": self.engine_id,
                "visualizations": [v.to_dict() for v in self.visualizations],
            },
            indent=2,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize engine to dictionary."""
        return {
            "engine_id": self.engine_id,
            "visualization_count": len(self.visualizations),
            "visualization_types": list(set(v.viz_type for v in self.visualizations)),
        }


@dataclass
class SimulationRun:
    """A single simulation run.

    Attributes:
        run_id: Unique run identifier
        scenario: Simulation scenario
        status: Current status
        start_time: Run start time
        end_time: Run end time
        current_tick: Current simulation tick
        metrics: Metrics collector
        events: Simulation events
        results: Final results
    """

    run_id: str
    scenario: SimulationScenario
    status: SimulationStatus = SimulationStatus.PENDING
    start_time: str = ""
    end_time: str = ""
    current_tick: int = 0
    metrics: MetricsCollector = field(default_factory=lambda: MetricsCollector(""))
    events: list[dict[str, Any]] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.metrics.collector_id:
            self.metrics = MetricsCollector(collector_id=f"metrics_{self.run_id}")

    def start(self) -> None:
        """Start the simulation run."""
        self.status = SimulationStatus.RUNNING
        self.start_time = datetime.now(timezone.utc).isoformat()
        self._log_event("simulation_started", {})

    def complete(self, results: dict[str, Any]) -> None:
        """Complete the simulation run.

        Args:
            results: Final simulation results
        """
        self.status = SimulationStatus.COMPLETED
        self.end_time = datetime.now(timezone.utc).isoformat()
        self.results = results
        self._log_event("simulation_completed", {"results_summary": list(results.keys())})

    def fail(self, error: str) -> None:
        """Mark simulation as failed.

        Args:
            error: Error message
        """
        self.status = SimulationStatus.FAILED
        self.end_time = datetime.now(timezone.utc).isoformat()
        self.results = {"error": error}
        self._log_event("simulation_failed", {"error": error})

    def _log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log a simulation event.

        Args:
            event_type: Type of event
            data: Event data
        """
        self.events.append(
            {
                "event_type": event_type,
                "tick": self.current_tick,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize run to dictionary."""
        return {
            "run_id": self.run_id,
            "scenario": self.scenario.to_dict(),
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "current_tick": self.current_tick,
            "metrics": self.metrics.to_dict(),
            "events": self.events[-50:],
            "results": self.results,
        }


class PlanetarySimulation:
    """Main planetary simulation engine.

    Orchestrates comprehensive simulations of the planetary infrastructure.

    Attributes:
        simulation_id: Unique simulation identifier
        scenarios: Registered scenarios
        runs: Simulation runs
        viz_engine: Visualization engine
        created_at: Simulation creation timestamp
    """

    def __init__(self, simulation_id: str | None = None) -> None:
        """Initialize planetary simulation.

        Args:
            simulation_id: Optional simulation ID
        """
        self.simulation_id = simulation_id or self._generate_id()
        self.scenarios: dict[str, SimulationScenario] = {}
        self.runs: list[SimulationRun] = []
        self.viz_engine = VisualizationEngine(engine_id=f"viz_{self.simulation_id}")
        self.created_at = datetime.now(timezone.utc).isoformat()

        # Register default scenarios
        self._register_default_scenarios()

    def _generate_id(self) -> str:
        """Generate unique simulation ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        return f"sim_{hashlib.sha256(timestamp.encode()).hexdigest()[:12]}"

    def _register_default_scenarios(self) -> None:
        """Register default simulation scenarios."""
        scenarios = [
            SimulationScenario(
                scenario_id="sc_global_adoption",
                name="Global Adoption",
                scenario_type=ScenarioType.GLOBAL_ADOPTION,
                description="Simulate global adoption across all regions and sectors",
                parameters={
                    "initial_users": 1000,
                    "growth_rate": 0.1,
                    "regions": ["americas", "europe", "asia", "africa", "oceania"],
                },
                duration_ticks=365,  # One year in days
            ),
            SimulationScenario(
                scenario_id="sc_sector_expansion",
                name="Sector Expansion",
                scenario_type=ScenarioType.SECTOR_EXPANSION,
                description="Simulate expansion into new industry sectors",
                parameters={
                    "sectors": ["energy", "healthcare", "finance", "logistics"],
                    "adoption_rate": 0.05,
                },
                duration_ticks=180,
            ),
            SimulationScenario(
                scenario_id="sc_stress_test",
                name="Infrastructure Stress Test",
                scenario_type=ScenarioType.STRESS_TEST,
                description="Simulate high-load conditions and system resilience",
                parameters={
                    "peak_load_multiplier": 10,
                    "duration_hours": 24,
                    "failure_injection": True,
                },
                duration_ticks=100,
            ),
            SimulationScenario(
                scenario_id="sc_disaster_recovery",
                name="Disaster Recovery Test",
                scenario_type=ScenarioType.DISASTER_RECOVERY,
                description="Simulate disaster scenarios and recovery procedures",
                parameters={
                    "disaster_type": "regional_outage",
                    "affected_regions": 2,
                    "recovery_target_hours": 4,
                },
                duration_ticks=50,
            ),
            SimulationScenario(
                scenario_id="sc_economic_shock",
                name="Economic Shock",
                scenario_type=ScenarioType.ECONOMIC_SHOCK,
                description="Simulate economic shock and token stability",
                parameters={
                    "shock_magnitude": 0.3,
                    "shock_duration_days": 30,
                    "recovery_mechanisms": ["liquidity_injection", "fee_adjustment"],
                },
                duration_ticks=90,
            ),
            SimulationScenario(
                scenario_id="sc_scaling_test",
                name="Scaling Test",
                scenario_type=ScenarioType.SCALING_TEST,
                description="Simulate infrastructure scaling under growth",
                parameters={
                    "target_scale_factor": 100,
                    "scaling_strategy": "horizontal",
                    "budget_constraint": 1_000_000,
                },
                duration_ticks=200,
            ),
        ]

        for scenario in scenarios:
            self.scenarios[scenario.scenario_id] = scenario

    def create_scenario(
        self,
        name: str,
        scenario_type: ScenarioType,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        duration_ticks: int = 1000,
    ) -> SimulationScenario:
        """Create a new simulation scenario.

        Args:
            name: Scenario name
            scenario_type: Type of scenario
            description: Scenario description
            parameters: Scenario parameters
            duration_ticks: Simulation duration

        Returns:
            Created scenario
        """
        scenario_id = f"sc_{hashlib.sha256(name.encode()).hexdigest()[:8]}"
        scenario = SimulationScenario(
            scenario_id=scenario_id,
            name=name,
            scenario_type=scenario_type,
            description=description,
            parameters=parameters or {},
            duration_ticks=duration_ticks,
        )
        self.scenarios[scenario_id] = scenario
        return scenario

    def run_simulation(
        self,
        scenario_id: str,
        seed: int = 42,
    ) -> SimulationRun:
        """Run a simulation scenario.

        Args:
            scenario_id: Scenario to run
            seed: Random seed for reproducibility

        Returns:
            Simulation run object
        """
        if scenario_id not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_id}")

        scenario = self.scenarios[scenario_id]
        run_id = f"run_{hashlib.sha256(f'{scenario_id}{datetime.now()}'.encode()).hexdigest()[:8]}"

        run = SimulationRun(
            run_id=run_id,
            scenario=scenario,
        )
        run.start()

        # Execute simulation based on scenario type
        try:
            if scenario.scenario_type == ScenarioType.GLOBAL_ADOPTION:
                results = self._simulate_global_adoption(run, seed)
            elif scenario.scenario_type == ScenarioType.SECTOR_EXPANSION:
                results = self._simulate_sector_expansion(run, seed)
            elif scenario.scenario_type == ScenarioType.STRESS_TEST:
                results = self._simulate_stress_test(run, seed)
            elif scenario.scenario_type == ScenarioType.DISASTER_RECOVERY:
                results = self._simulate_disaster_recovery(run, seed)
            elif scenario.scenario_type == ScenarioType.ECONOMIC_SHOCK:
                results = self._simulate_economic_shock(run, seed)
            elif scenario.scenario_type == ScenarioType.SCALING_TEST:
                results = self._simulate_scaling_test(run, seed)
            else:
                results = self._simulate_generic(run, seed)

            run.complete(results)
        except Exception as e:
            run.fail(str(e))

        self.runs.append(run)
        return run

    def _simulate_global_adoption(self, run: SimulationRun, seed: int) -> dict[str, Any]:
        """Simulate global adoption scenario.

        Args:
            run: Simulation run object
            seed: Random seed

        Returns:
            Simulation results
        """
        # random already imported at module level
        random.seed(seed)

        params = run.scenario.parameters
        users = params.get("initial_users", 1000)
        growth_rate = params.get("growth_rate", 0.1)
        regions = params.get("regions", ["global"])

        regional_users = {r: users // len(regions) for r in regions}
        total_revenue = 0.0

        for tick in range(run.scenario.duration_ticks):
            run.current_tick = tick

            # Growth with regional variation
            for region in regions:
                region_growth = growth_rate * (1 + random.gauss(0, 0.1))
                regional_users[region] = int(regional_users[region] * (1 + region_growth / 365))

            total_users = sum(regional_users.values())
            daily_revenue = total_users * 0.1  # $0.10 per user per day
            total_revenue += daily_revenue

            # Record metrics
            run.metrics.record("total_users", tick, total_users)
            run.metrics.record("daily_revenue", tick, daily_revenue)
            run.metrics.record("cumulative_revenue", tick, total_revenue)

            for region, count in regional_users.items():
                run.metrics.record(f"users_{region}", tick, count)

        return {
            "final_users": sum(regional_users.values()),
            "total_revenue": total_revenue,
            "regional_distribution": regional_users,
            "growth_achieved": sum(regional_users.values()) / params.get("initial_users", 1),
        }

    def _simulate_sector_expansion(self, run: SimulationRun, seed: int) -> dict[str, Any]:
        """Simulate sector expansion scenario.

        Args:
            run: Simulation run object
            seed: Random seed

        Returns:
            Simulation results
        """
        # random already imported at module level
        random.seed(seed)

        params = run.scenario.parameters
        sectors = params.get("sectors", ["general"])
        adoption_rate = params.get("adoption_rate", 0.05)

        sector_adoption = dict.fromkeys(sectors, 0.0)
        integrations = dict.fromkeys(sectors, 0)

        for tick in range(run.scenario.duration_ticks):
            run.current_tick = tick

            for sector in sectors:
                # Adoption follows S-curve
                current = sector_adoption[sector]
                growth = adoption_rate * current * (1 - current) + 0.01
                sector_adoption[sector] = min(1.0, current + growth)

                # New integrations
                if random.random() < sector_adoption[sector] * 0.1:
                    integrations[sector] += 1

                run.metrics.record(f"adoption_{sector}", tick, sector_adoption[sector] * 100)
                run.metrics.record(f"integrations_{sector}", tick, integrations[sector])

        return {
            "final_adoption": {s: a * 100 for s, a in sector_adoption.items()},
            "total_integrations": integrations,
            "average_adoption": sum(sector_adoption.values()) / len(sectors) * 100,
        }

    def _simulate_stress_test(self, run: SimulationRun, seed: int) -> dict[str, Any]:
        """Simulate stress test scenario.

        Args:
            run: Simulation run object
            seed: Random seed

        Returns:
            Simulation results
        """
        # random already imported at module level
        random.seed(seed)

        params = run.scenario.parameters
        peak_multiplier = params.get("peak_load_multiplier", 10)

        base_load = 0.3
        failures = 0
        max_latency = 0.0

        for tick in range(run.scenario.duration_ticks):
            run.current_tick = tick

            # Load ramps up to peak and back down
            progress = tick / run.scenario.duration_ticks
            if progress < 0.3:
                load = base_load + (peak_multiplier * base_load - base_load) * (progress / 0.3)
            elif progress < 0.7:
                load = base_load * peak_multiplier
            else:
                load = base_load * peak_multiplier - (peak_multiplier * base_load - base_load) * (
                    (progress - 0.7) / 0.3
                )

            load = min(1.0, load)

            # Latency increases with load
            latency = 10 + (load**2) * 990  # 10ms to 1000ms
            max_latency = max(max_latency, latency)

            # Failure probability increases with load
            if random.random() < (load - 0.8) * 0.5:
                failures += 1

            run.metrics.record("system_load", tick, load * 100)
            run.metrics.record("latency_ms", tick, latency)
            run.metrics.record("cumulative_failures", tick, failures)

        return {
            "peak_load_achieved": peak_multiplier * base_load * 100,
            "max_latency_ms": max_latency,
            "total_failures": failures,
            "availability": (run.scenario.duration_ticks - failures)
            / run.scenario.duration_ticks
            * 100,
        }

    def _simulate_disaster_recovery(self, run: SimulationRun, seed: int) -> dict[str, Any]:
        """Simulate disaster recovery scenario.

        Args:
            run: Simulation run object
            seed: Random seed

        Returns:
            Simulation results
        """
        # random already imported at module level
        random.seed(seed)

        params = run.scenario.parameters
        affected_regions = params.get("affected_regions", 1)
        recovery_target = params.get("recovery_target_hours", 4)

        system_health = 1.0
        disaster_tick = run.scenario.duration_ticks // 4
        recovery_complete_tick = None

        for tick in range(run.scenario.duration_ticks):
            run.current_tick = tick

            if tick == disaster_tick:
                # Disaster strikes
                system_health = 1.0 - (affected_regions * 0.25)

            if tick > disaster_tick and system_health < 1.0:
                # Recovery process
                recovery_rate = 0.1
                system_health = min(1.0, system_health + recovery_rate)

                if system_health >= 0.99 and recovery_complete_tick is None:
                    recovery_complete_tick = tick

            run.metrics.record("system_health", tick, system_health * 100)
            run.metrics.record(
                "regions_affected",
                tick,
                affected_regions if tick >= disaster_tick and system_health < 1.0 else 0,
            )

        recovery_time = (
            (recovery_complete_tick - disaster_tick)
            if recovery_complete_tick
            else run.scenario.duration_ticks
        )
        rto_met = recovery_time <= recovery_target

        return {
            "disaster_tick": disaster_tick,
            "recovery_complete_tick": recovery_complete_tick,
            "recovery_time_ticks": recovery_time,
            "recovery_target_met": rto_met,
            "min_health_during_disaster": (1.0 - affected_regions * 0.25) * 100,
        }

    def _simulate_economic_shock(self, run: SimulationRun, seed: int) -> dict[str, Any]:
        """Simulate economic shock scenario.

        Args:
            run: Simulation run object
            seed: Random seed

        Returns:
            Simulation results
        """
        # random already imported at module level
        random.seed(seed)

        params = run.scenario.parameters
        shock_magnitude = params.get("shock_magnitude", 0.3)
        shock_duration = params.get("shock_duration_days", 30)

        token_price = 1.0
        liquidity = 1.0
        shock_start = run.scenario.duration_ticks // 4
        min_price = token_price

        for tick in range(run.scenario.duration_ticks):
            run.current_tick = tick

            if shock_start <= tick < shock_start + shock_duration:
                # During shock
                shock_effect = shock_magnitude * (1 - (tick - shock_start) / shock_duration)
                token_price *= 1 - shock_effect * random.gauss(0.1, 0.05)
                liquidity *= 0.99
            else:
                # Recovery
                token_price *= 1 + random.gauss(0.01, 0.02)
                liquidity = min(1.0, liquidity * 1.01)

            token_price = max(0.1, min(2.0, token_price))
            min_price = min(min_price, token_price)

            run.metrics.record("token_price", tick, token_price)
            run.metrics.record("liquidity", tick, liquidity * 100)
            run.metrics.record("volatility", tick, abs(random.gauss(0, shock_magnitude)) * 100)

        return {
            "initial_price": 1.0,
            "final_price": token_price,
            "min_price_during_shock": min_price,
            "price_recovery": token_price >= 0.9,
            "final_liquidity": liquidity * 100,
        }

    def _simulate_scaling_test(self, run: SimulationRun, seed: int) -> dict[str, Any]:
        """Simulate scaling test scenario.

        Args:
            run: Simulation run object
            seed: Random seed

        Returns:
            Simulation results
        """
        # random already imported at module level
        random.seed(seed)

        params = run.scenario.parameters
        target_scale = params.get("target_scale_factor", 100)
        budget = params.get("budget_constraint", 1_000_000)

        nodes = 10
        capacity = 100
        cost_spent = 0
        demand = 100

        for tick in range(run.scenario.duration_ticks):
            run.current_tick = tick

            # Demand grows
            demand *= 1.02
            utilization = min(1.0, demand / capacity)

            # Auto-scale if needed and budget allows
            if utilization > 0.8 and cost_spent < budget:
                new_nodes = max(1, int(nodes * 0.1))
                node_cost = new_nodes * 1000
                if cost_spent + node_cost <= budget:
                    nodes += new_nodes
                    capacity = nodes * 10
                    cost_spent += node_cost

            scale_achieved = capacity / 100

            run.metrics.record("nodes", tick, nodes)
            run.metrics.record("capacity", tick, capacity)
            run.metrics.record("utilization", tick, utilization * 100)
            run.metrics.record("scale_factor", tick, scale_achieved)
            run.metrics.record("cost_spent", tick, cost_spent)

        return {
            "final_nodes": nodes,
            "final_capacity": capacity,
            "scale_achieved": capacity / 100,
            "target_met": (capacity / 100) >= target_scale,
            "total_cost": cost_spent,
            "cost_efficiency": capacity / cost_spent if cost_spent > 0 else 0,
        }

    def _simulate_generic(self, run: SimulationRun, seed: int) -> dict[str, Any]:
        """Generic simulation for custom scenarios.

        Args:
            run: Simulation run object
            seed: Random seed

        Returns:
            Simulation results
        """
        # random already imported at module level
        random.seed(seed)

        for tick in range(run.scenario.duration_ticks):
            run.current_tick = tick
            run.metrics.record("generic_metric", tick, random.random() * 100)

        return {"status": "completed", "ticks_executed": run.scenario.duration_ticks}

    def generate_visualizations(self, run_id: str) -> list[VisualizationData]:
        """Generate visualizations for a simulation run.

        Args:
            run_id: Run to visualize

        Returns:
            List of generated visualizations
        """
        run = next((r for r in self.runs if r.run_id == run_id), None)
        if not run:
            return []

        visualizations = []

        # Time series of all metrics
        metric_names = list(run.metrics.metrics.keys())
        if metric_names:
            viz = self.viz_engine.create_time_series_chart(
                title=f"Simulation Metrics: {run.scenario.name}",
                metrics=run.metrics,
                metric_names=metric_names[:5],  # Top 5 metrics
            )
            visualizations.append(viz)

        # Dashboard with key results
        if run.results:
            dashboard = self.viz_engine.create_dashboard(
                title=f"Dashboard: {run.scenario.name}",
                metrics=run.results,
                charts=[],
            )
            visualizations.append(dashboard)

        return visualizations

    def get_statistics(self) -> dict[str, Any]:
        """Get simulation statistics.

        Returns:
            Simulation statistics
        """
        runs_by_status: dict[str, int] = {s.value: 0 for s in SimulationStatus}
        for run in self.runs:
            runs_by_status[run.status.value] += 1

        return {
            "simulation_id": self.simulation_id,
            "total_scenarios": len(self.scenarios),
            "total_runs": len(self.runs),
            "runs_by_status": runs_by_status,
            "visualizations_generated": len(self.viz_engine.visualizations),
            "created_at": self.created_at,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize simulation to dictionary."""
        return {
            "simulation_id": self.simulation_id,
            "scenarios": {k: v.to_dict() for k, v in self.scenarios.items()},
            "runs": [r.to_dict() for r in self.runs[-10:]],
            "viz_engine": self.viz_engine.to_dict(),
            "statistics": self.get_statistics(),
            "created_at": self.created_at,
        }
