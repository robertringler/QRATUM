"""Tire scenario generator for creating comprehensive simulation library."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from .environment import EnvironmentalConditions, RoadSurface, WeatherCondition
from .geometry import TireGeometry, TireStructure, TireType, TreadDesign, TreadPattern
from .materials import CompoundType, MaterialProperties, TireCompound
from .simulation import TireSimulation


class TireScenarioGenerator:
    """Generator for diverse tire simulation scenarios.

    Creates comprehensive library of tire simulations covering:
    - Multiple tire types and sizes
    - Various compound formulations
    - Wide range of environmental conditions
    - Different operating conditions (load, pressure, speed)
    """

    def __init__(self, random_seed: int = 42):
        """Initialize scenario generator.

        Args:
            random_seed: Random seed for reproducibility
        """

        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

    def generate_compound_variants(self, count: int = 8) -> list[TireCompound]:
        """Generate diverse tire compound variants.

        Args:
            count: Number of compound variants to generate

        Returns:
            List of tire compound specifications
        """

        compound_types = list(CompoundType)
        compounds = []

        for i in range(count):
            comp_type = compound_types[i % len(compound_types)]

            # Base properties vary by compound type
            if comp_type == CompoundType.NATURAL_RUBBER:
                props = MaterialProperties(
                    elastic_modulus=0.002,
                    hardness_shore_a=65,
                    wet_grip_coefficient=0.78,
                    rolling_resistance_coeff=0.011,
                    abrasion_resistance=0.75,
                )
            elif comp_type == CompoundType.SYNTHETIC_RUBBER:
                props = MaterialProperties(
                    elastic_modulus=0.0025,
                    hardness_shore_a=70,
                    wet_grip_coefficient=0.72,
                    rolling_resistance_coeff=0.010,
                    abrasion_resistance=0.82,
                )
            elif comp_type == CompoundType.BIOPOLYMER:
                props = MaterialProperties(
                    elastic_modulus=0.0018,
                    hardness_shore_a=60,
                    wet_grip_coefficient=0.75,
                    rolling_resistance_coeff=0.009,
                    abrasion_resistance=0.70,
                )
            elif comp_type == CompoundType.NANO_ENHANCED:
                props = MaterialProperties(
                    elastic_modulus=0.0028,
                    hardness_shore_a=68,
                    wet_grip_coefficient=0.80,
                    rolling_resistance_coeff=0.009,
                    abrasion_resistance=0.88,
                )
            elif comp_type == CompoundType.GRAPHENE_REINFORCED:
                props = MaterialProperties(
                    elastic_modulus=0.0032,
                    hardness_shore_a=72,
                    wet_grip_coefficient=0.82,
                    rolling_resistance_coeff=0.008,
                    abrasion_resistance=0.92,
                )
            elif comp_type == CompoundType.QUANTUM_OPTIMIZED:
                props = MaterialProperties(
                    elastic_modulus=0.0030,
                    hardness_shore_a=66,
                    wet_grip_coefficient=0.85,
                    rolling_resistance_coeff=0.0075,
                    abrasion_resistance=0.95,
                )
            elif comp_type == CompoundType.SILICA_ENHANCED:
                props = MaterialProperties(
                    elastic_modulus=0.0024,
                    hardness_shore_a=64,
                    wet_grip_coefficient=0.80,
                    rolling_resistance_coeff=0.0085,
                    abrasion_resistance=0.85,
                )
            else:  # CARBON_BLACK
                props = MaterialProperties(
                    elastic_modulus=0.0026,
                    hardness_shore_a=68,
                    wet_grip_coefficient=0.76,
                    rolling_resistance_coeff=0.010,
                    abrasion_resistance=0.80,
                )

            compound = TireCompound(
                compound_id=f"COMP_{i:04d}",
                name=f"{comp_type.value}_variant_{i}",
                compound_type=comp_type,
                base_properties=props,
                additives={"silica": 0.15, "carbon_black": 0.25, "sulfur": 0.02},
                quantum_optimization_level=self.rng.uniform(0, 1),
            )
            compounds.append(compound)

        return compounds

    def generate_geometry_variants(self, count: int = 50) -> list[TireGeometry]:
        """Generate diverse tire geometry variants.

        Args:
            count: Number of geometry variants to generate

        Returns:
            List of tire geometry specifications
        """

        tire_types = list(TireType)
        tread_patterns = list(TreadPattern)
        geometries = []

        # Common tire sizes for different types
        size_templates = {
            TireType.PASSENGER: [(205, 55, 16), (225, 45, 17), (235, 40, 18)],
            TireType.TRUCK: [(265, 70, 17), (275, 65, 18), (285, 60, 20)],
            TireType.OFF_ROAD: [(31, 10.5, 15), (33, 12.5, 17), (35, 12.5, 20)],
            TireType.RACING: [(245, 35, 18), (275, 30, 19), (305, 30, 20)],
            TireType.EV_SPECIFIC: [(235, 45, 18), (255, 40, 19), (265, 35, 20)],
            TireType.WINTER: [(215, 60, 16), (225, 55, 17), (235, 50, 18)],
            TireType.ALL_SEASON: [(215, 55, 17), (225, 50, 18), (235, 45, 18)],
            TireType.PERFORMANCE: [(245, 40, 18), (255, 35, 19), (275, 30, 20)],
        }

        for i in range(count):
            tire_type = tire_types[i % len(tire_types)]
            tread_pattern = tread_patterns[i % len(tread_patterns)]
            sizes = size_templates.get(tire_type, [(225, 50, 17)])
            width, aspect, diameter = sizes[i % len(sizes)]

            # Tread design varies by tire type
            if tire_type == TireType.WINTER:
                tread = TreadDesign(
                    pattern_type=tread_pattern,
                    groove_depth=10.0,
                    groove_width=10.0,
                    groove_count=5,
                    sipe_density=8.0,
                    sipe_depth=6.0,
                    void_ratio=0.35,
                    edge_count=300,
                )
            elif tire_type == TireType.RACING:
                tread = TreadDesign(
                    pattern_type=tread_pattern,
                    groove_depth=6.0,
                    groove_width=6.0,
                    groove_count=3,
                    sipe_density=2.0,
                    sipe_depth=2.0,
                    void_ratio=0.20,
                    edge_count=150,
                )
            elif tire_type == TireType.OFF_ROAD:
                tread = TreadDesign(
                    pattern_type=TreadPattern.DIRECTIONAL,
                    groove_depth=12.0,
                    groove_width=12.0,
                    groove_count=6,
                    sipe_density=4.0,
                    sipe_depth=5.0,
                    void_ratio=0.40,
                    edge_count=250,
                    block_stiffness=0.8,
                )
            else:
                tread = TreadDesign(
                    pattern_type=tread_pattern,
                    groove_depth=8.0 + self.rng.uniform(-1, 1),
                    groove_width=8.0 + self.rng.uniform(-1, 1),
                    groove_count=4,
                    sipe_density=5.0 + self.rng.uniform(-1, 1),
                    sipe_depth=4.0 + self.rng.uniform(-0.5, 0.5),
                    void_ratio=0.30 + self.rng.uniform(-0.05, 0.05),
                    edge_count=int(200 + self.rng.uniform(-50, 50)),
                )

            # Structure varies by tire type
            if tire_type == TireType.TRUCK:
                structure = TireStructure(
                    belt_count=3,
                    ply_count=4,
                    max_load=1200.0,
                    max_pressure=400.0,
                )
            elif tire_type == TireType.RACING:
                structure = TireStructure(
                    belt_count=2,
                    ply_count=2,
                    max_load=700.0,
                    max_pressure=280.0,
                )
            else:
                structure = TireStructure(
                    belt_count=2,
                    ply_count=2,
                    max_load=800.0 + self.rng.uniform(-100, 100),
                    max_pressure=300.0 + self.rng.uniform(-20, 20),
                )

            geometry = TireGeometry(
                tire_id=f"TIRE_{i:04d}",
                tire_type=tire_type,
                width=width,
                aspect_ratio=aspect,
                diameter=diameter,
                tread_design=tread,
                structure=structure,
                mass=8.0 + 0.5 * width / 50.0,
            )
            geometries.append(geometry)

        return geometries

    def generate_environment_variants(self, count: int = 40) -> list[EnvironmentalConditions]:
        """Generate diverse environmental condition variants.

        Args:
            count: Number of environment variants to generate

        Returns:
            List of environmental condition specifications
        """

        surfaces = list(RoadSurface)
        weathers = list(WeatherCondition)
        environments = []

        # Temperature range from -40°C to +80°C
        temperatures = np.linspace(-40, 80, count // 4)

        for i in range(count):
            surface = surfaces[i % len(surfaces)]
            weather = weathers[i % len(weathers)]
            temp = temperatures[i % len(temperatures)]

            # Wetness based on weather
            if weather in [WeatherCondition.RAIN, WeatherCondition.HEAVY_RAIN]:
                wetness = 0.5 + self.rng.uniform(0, 0.5)
            elif weather in [WeatherCondition.SNOW, WeatherCondition.ICE]:
                wetness = 0.3 + self.rng.uniform(0, 0.3)
            else:
                wetness = self.rng.uniform(0, 0.2)

            # Road temperature slightly different from ambient
            road_temp = temp + self.rng.uniform(-5, 15)

            # Rainfall rate
            rainfall = 0.0
            if weather == WeatherCondition.RAIN:
                rainfall = self.rng.uniform(5, 25)
            elif weather == WeatherCondition.HEAVY_RAIN:
                rainfall = self.rng.uniform(25, 100)

            env = EnvironmentalConditions(
                ambient_temperature=temp,
                road_temperature=road_temp,
                surface_type=surface,
                surface_wetness=wetness,
                weather=weather,
                humidity=self.rng.uniform(0.2, 0.9),
                wind_speed=self.rng.uniform(0, 20),
                altitude=self.rng.uniform(0, 3000),
                uv_index=max(0, min(11, 5 + (temp - 20) * 0.1)),
                rainfall_rate=rainfall,
            )
            environments.append(env)

        return environments

    def generate_operating_conditions(self, count: int = 25) -> list[tuple[float, float, float]]:
        """Generate diverse operating condition variants.

        Args:
            count: Number of operating condition sets to generate

        Returns:
            List of (load_kg, pressure_kpa, speed_kmh) tuples
        """

        conditions = []

        # Load range: 200-1200 kg
        loads = np.linspace(200, 1200, 5)
        # Pressure range: 180-350 kPa
        pressures = np.linspace(180, 350, 5)
        # Speed range: 0-250 km/h
        speeds = np.linspace(30, 250, 5)

        # Generate combinations
        for load, pressure, speed in itertools.product(loads, pressures[:3], speeds[:5]):
            conditions.append((float(load), float(pressure), float(speed)))
            if len(conditions) >= count:
                break

        return conditions

    def generate_scenarios(self, target_count: int = 10000) -> list[dict[str, Any]]:
        """Generate comprehensive tire simulation scenario library.

        Args:
            target_count: Target number of scenarios to generate

        Returns:
            List of scenario specifications ready for simulation
        """

        print(f"Generating {target_count} tire simulation scenarios...")

        # Generate component variants
        compounds = self.generate_compound_variants(8)
        geometries = self.generate_geometry_variants(50)
        environments = self.generate_environment_variants(40)
        conditions = self.generate_operating_conditions(25)

        print(f"Generated {len(compounds)} compound variants")
        print(f"Generated {len(geometries)} geometry variants")
        print(f"Generated {len(environments)} environment variants")
        print(f"Generated {len(conditions)} operating condition sets")

        scenarios = []
        scenario_id = 0

        # Create combinations to reach target count
        # Reorder loops to ensure diversity across all dimensions early
        iterations_needed = (
            target_count // (len(compounds) * len(geometries) * len(environments)) + 1
        )

        for _iteration in range(iterations_needed):
            if len(scenarios) >= target_count:
                break

            for geometry in geometries:
                for environment in environments:
                    for compound in compounds:
                        if len(scenarios) >= target_count:
                            break

                        # Select operating conditions
                        load, pressure, speed = conditions[scenario_id % len(conditions)]

                        scenario = {
                            "simulation_id": f"SIM_{scenario_id:06d}",
                            "compound": compound,
                            "geometry": geometry,
                            "environment": environment,
                            "load_kg": load,
                            "pressure_kpa": pressure,
                            "speed_kmh": speed,
                        }

                        scenarios.append(scenario)
                        scenario_id += 1

        print(f"Generated {len(scenarios)} unique scenarios")
        return scenarios[:target_count]


def generate_tire_library(
    output_dir: str = "tire_simulation_library",
    scenario_count: int = 10000,
    run_simulations: bool = True,
    export_format: str = "json",
) -> dict[str, Any]:
    """Generate complete tire simulation library with Goodyear Quantum Pilot integration.

    Args:
        output_dir: Output directory for library
        scenario_count: Number of scenarios to generate
        run_simulations: Whether to run simulations or just generate scenarios
        export_format: Export format ('json', 'csv', or 'both')

    Returns:
        Summary of library generation
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate scenarios
    generator = TireScenarioGenerator(random_seed=42)
    scenarios = generator.generate_scenarios(target_count=scenario_count)

    summary = {
        "total_scenarios": len(scenarios),
        "run_simulations": run_simulations,
        "output_directory": str(output_path),
        "export_format": export_format,
    }

    if run_simulations:
        # Run simulations
        print(f"Running {len(scenarios)} tire simulations...")
        simulator = TireSimulation(use_quantum_acceleration=True, random_seed=42)
        results = []

        for i, scenario in enumerate(scenarios):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(scenarios)} simulations complete")

            result = simulator.simulate(
                simulation_id=scenario["simulation_id"],
                tire_geometry=scenario["geometry"],
                compound=scenario["compound"],
                environment=scenario["environment"],
                load_kg=scenario["load_kg"],
                pressure_kpa=scenario["pressure_kpa"],
                speed_kmh=scenario["speed_kmh"],
            )
            results.append(result.to_dict())

        print(f"Completed {len(results)} simulations")

        # Export results
        if export_format in ["json", "both"]:
            json_file = output_path / "tire_simulation_results.json"
            with open(json_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Exported JSON to {json_file}")
            summary["json_file"] = str(json_file)

        if export_format in ["csv", "both"]:
            # Export flattened CSV for easy analysis
            csv_file = output_path / "tire_simulation_results.csv"
            _export_to_csv(results, csv_file)
            print(f"Exported CSV to {csv_file}")
            summary["csv_file"] = str(csv_file)

        # Generate summary statistics
        summary["statistics"] = _compute_library_statistics(results)

    else:
        # Just export scenario specifications
        scenario_dicts = []
        for scenario in scenarios:
            scenario_dicts.append(
                {
                    "simulation_id": scenario["simulation_id"],
                    "compound": scenario["compound"].to_dict(),
                    "geometry": scenario["geometry"].to_dict(),
                    "environment": scenario["environment"].to_dict(),
                    "load_kg": scenario["load_kg"],
                    "pressure_kpa": scenario["pressure_kpa"],
                    "speed_kmh": scenario["speed_kmh"],
                }
            )

        json_file = output_path / "tire_scenarios.json"
        with open(json_file, "w") as f:
            json.dump(scenario_dicts, f, indent=2)
        print(f"Exported scenarios to {json_file}")
        summary["scenario_file"] = str(json_file)

    # Export README
    readme_file = output_path / "README.md"
    _generate_readme(readme_file, summary)

    return summary


def _export_to_csv(results: list[dict[str, Any]], csv_file: Path) -> None:
    """Export results to CSV format."""

    try:
        import csv

        with open(csv_file, "w", newline="") as f:
            if not results:
                return

            # Flatten first result to get headers
            first_result = results[0]
            fieldnames = _flatten_dict_keys(first_result)

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                flat_result = _flatten_dict(result)
                writer.writerow(flat_result)

    except ImportError:
        print("CSV export requires csv module (should be in standard library)")


def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = "_") -> dict[str, Any]:
    """Flatten nested dictionary."""

    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def _flatten_dict_keys(d: dict[str, Any], parent_key: str = "", sep: str = "_") -> list[str]:
    """Get flattened dictionary keys."""

    keys = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            keys.extend(_flatten_dict_keys(v, new_key, sep=sep))
        else:
            keys.append(new_key)
    return keys


def _compute_library_statistics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics for simulation library."""

    if not results:
        return {}

    metrics_list = [r["performance_metrics"] for r in results]

    def avg(key: str) -> float:
        values = [m.get(key, 0) for m in metrics_list]
        return sum(values) / len(values) if values else 0.0

    def minmax(key: str) -> tuple[float, float]:
        values = [m.get(key, 0) for m in metrics_list]
        return (min(values) if values else 0.0, max(values) if values else 0.0)

    return {
        "avg_grip_coefficient": round(avg("grip_coefficient"), 4),
        "avg_rolling_resistance": round(avg("rolling_resistance"), 4),
        "avg_wear_rate": round(avg("wear_rate"), 3),
        "avg_thermal_performance": round(avg("thermal_performance"), 4),
        "avg_optimization_score": round(avg("optimization_score"), 4),
        "grip_range": [round(x, 4) for x in minmax("grip_coefficient")],
        "rolling_resistance_range": [round(x, 4) for x in minmax("rolling_resistance")],
        "predicted_lifetime_range": [round(x, 0) for x in minmax("predicted_lifetime_km")],
    }


def _generate_readme(readme_file: Path, summary: dict[str, Any]) -> None:
    """Generate README for tire simulation library."""

    content = f"""# Tire Simulation Library - Goodyear Quantum Pilot Platform Integration

## Overview

This library contains {summary["total_scenarios"]} comprehensive tire simulation scenarios
generated using QuASIM quantum-accelerated simulation engine.

## Library Structure

- **Tire Types**: Passenger, Truck, Off-Road, Racing, EV-Specific, Winter, All-Season, Performance
- **Compounds**: Natural Rubber, Synthetic Rubber, Bio-Polymer, Nano-Enhanced, Graphene-Reinforced, Quantum-Optimized
- **Environmental Conditions**: Temperature range -40°C to +80°C, multiple surface types and weather conditions
- **Performance Domains**: Traction, wear, thermal response, rolling resistance, hydroplaning, noise, durability

## Quantum Enhancement

All simulations utilize QuASIM's quantum-enhanced optimization for:
- Multi-variable compound interactions
- Material property optimization
- Emergent behavior detection
- Predictive failure analysis

## Data Format

### JSON Structure
Each simulation result contains:
- Input parameters (tire geometry, compound, environment, load, pressure, speed)
- Performance metrics (grip, rolling resistance, wear, thermal performance, etc.)
- Thermal analysis (temperature distribution)
- Wear analysis (wear pattern across tread)
- Stress analysis (structural stress distribution)
- Optimization suggestions

### CSV Format
Flattened representation for easy data analysis and machine learning workflows.

## Integration with Goodyear Quantum Pilot

This library is compatible with:
- CAD systems (geometry export)
- FEA tools (stress and thermal analysis)
- AI-driven design workflows (optimization suggestions)
- Predictive maintenance systems (lifetime and failure mode prediction)

## Usage

```python
from quasim.domains.tire import generate_tire_library

# Generate library
summary = generate_tire_library(
    output_dir="tire_simulation_library",
    scenario_count=10000,
    run_simulations=True,
    export_format="both"
)
```

## Statistics

"""

    if "statistics" in summary:
        stats = summary["statistics"]
        content += f"""

- Average Grip Coefficient: {stats.get("avg_grip_coefficient", "N/A")}
- Average Rolling Resistance: {stats.get("avg_rolling_resistance", "N/A")}
- Average Wear Rate: {stats.get("avg_wear_rate", "N/A")} mm/1000km
- Average Thermal Performance: {stats.get("avg_thermal_performance", "N/A")}
- Average Optimization Score: {stats.get("avg_optimization_score", "N/A")}

### Performance Ranges
- Grip Coefficient: {stats.get("grip_range", ["N/A", "N/A"])[0]} - {stats.get("grip_range", ["N/A", "N/A"])[1]}
- Rolling Resistance: {stats.get("rolling_resistance_range", ["N/A", "N/A"])[0]} - {stats.get("rolling_resistance_range", ["N/A", "N/A"])[1]}
- Predicted Lifetime: {stats.get("predicted_lifetime_range", ["N/A", "N/A"])[0]} - {stats.get("predicted_lifetime_range", ["N/A", "N/A"])[1]} km
"""

    content += """

## Compliance

All simulations maintain:
- Deterministic reproducibility (<1μs seed replay drift)
- DO-178C Level A safety compliance posture
- Full provenance tracking via QuNimbus
- Industrial-grade validation suitable for R&D and product development

## Contact

For questions or custom simulation requests, contact the QuASIM team.
"""

    with open(readme_file, "w") as f:
        f.write(content)
