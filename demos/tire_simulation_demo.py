"""Demonstration of comprehensive tire simulation library generation.

This demo showcases the QuASIM tire simulation capabilities integrated with
the Goodyear Quantum Pilot platform, generating 10,000+ unique tire simulation
scenarios with quantum-enhanced optimization.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_basic_tire_library():
    """Demonstrate basic tire library generation with synthetic compounds."""
    print("=" * 80)
    print("DEMO 1: Basic Tire Simulation Library")
    print("=" * 80)
    print()

    from quasim.domains.tire import generate_tire_library

    print("Generating 1,000 tire simulation scenarios with synthetic compounds...")
    print()

    summary = generate_tire_library(
        output_dir="demo_tire_library_basic",
        scenario_count=1000,
        run_simulations=True,
        export_format="both",
    )

    print()
    print("Summary:")
    print(f"  Total Scenarios: {summary['total_scenarios']}")
    print(f"  Output Directory: {summary['output_directory']}")

    if "statistics" in summary:
        stats = summary["statistics"]
        print()
        print("Performance Statistics:")
        print(f"  Average Grip Coefficient: {stats['avg_grip_coefficient']}")
        print(f"  Average Rolling Resistance: {stats['avg_rolling_resistance']}")
        print(f"  Average Wear Rate: {stats['avg_wear_rate']} mm/1000km")
        print(f"  Average Optimization Score: {stats['avg_optimization_score']}")
        print(
            f"  Grip Range: {stats['grip_range'][0]} - {stats['grip_range'][1]}"
        )
        print(
            f"  Lifetime Range: {stats['predicted_lifetime_range'][0]} - {stats['predicted_lifetime_range'][1]} km"
        )

    print()


def demo_goodyear_integration():
    """Demonstrate Goodyear Quantum Pilot integration with 1,000+ materials."""
    print("=" * 80)
    print("DEMO 2: Goodyear Quantum Pilot Integration")
    print("=" * 80)
    print()

    try:
        from integrations.goodyear import GoodyearQuantumPilot

        # Initialize Goodyear Quantum Pilot
        gqp = GoodyearQuantumPilot()

        # Show database statistics
        db_stats = gqp.materials_db.get_statistics()
        print("Goodyear Materials Database:")
        print(f"  Total Materials: {db_stats['total_materials']}")
        print(f"  Quantum Validated: {db_stats['quantum_validated']} ({db_stats['quantum_validated_percentage']}%)")
        print()
        print("Materials by Family:")
        for family, count in db_stats["by_family"].items():
            print(f"  {family.replace('_', ' ').title()}: {count} materials")
        print()

        # Generate library with quantum-validated materials
        print("Generating simulation library with quantum-validated Goodyear materials...")
        print("(Using subset for demo - full library can generate 10,000+ scenarios)")
        print()

        summary = gqp.generate_comprehensive_library(
            output_dir="demo_tire_library_goodyear",
            scenarios_per_material=5,
            use_all_materials=False,
            material_filters={"quantum_validated": True, "certification_status": "certified"},
        )

        print()
        print("Summary:")
        print(f"  Materials Used: {summary['total_materials']}")
        print(f"  Total Scenarios: {summary['total_scenarios']}")
        print(f"  Scenarios per Material: {summary['scenarios_per_material']}")

        if "statistics" in summary:
            stats = summary["statistics"]
            print()
            print("Performance Statistics:")
            print(f"  Average Grip Coefficient: {stats['avg_grip_coefficient']}")
            print(f"  Average Rolling Resistance: {stats['avg_rolling_resistance']}")
            print(f"  Average Optimization Score: {stats['avg_optimization_score']}")

    except ImportError:
        print("Goodyear integration not available")

    print()


def demo_material_search():
    """Demonstrate material database search capabilities."""
    print("=" * 80)
    print("DEMO 3: Material Database Search")
    print("=" * 80)
    print()

    try:
        from integrations.goodyear import GoodyearMaterialsDatabase

        db = GoodyearMaterialsDatabase()

        # Search for high-performance materials
        print("Searching for high-performance materials (wet grip > 0.80)...")
        results = db.search_materials(
            min_wet_grip=0.80, certification_status="certified", quantum_validated=True
        )

        print(f"Found {len(results)} materials matching criteria")
        print()
        print("Top 5 materials:")
        for i, material in enumerate(results[:5]):
            print(f"{i+1}. {material.name}")
            print(f"   Family: {material.family}")
            print(f"   Wet Grip: {material.properties.get('wet_grip_coefficient', 0):.3f}")
            print(
                f"   Rolling Resistance: {material.properties.get('rolling_resistance_coeff', 0):.4f}"
            )
            print(
                f"   Abrasion Resistance: {material.properties.get('abrasion_resistance', 0):.3f}"
            )
            print()

        # Search by family
        print("Searching for graphene-reinforced materials...")
        graphene_materials = db.search_materials(family="graphene_reinforced")
        print(f"Found {len(graphene_materials)} graphene-reinforced materials")
        print()

    except ImportError:
        print("Goodyear integration not available")

    print()


def demo_quantum_optimization():
    """Demonstrate quantum optimization of tire compounds."""
    print("=" * 80)
    print("DEMO 4: Quantum Optimization")
    print("=" * 80)
    print()

    from quasim.domains.tire.materials import (CompoundType,
                                               MaterialProperties,
                                               TireCompound)

    # Create a base compound
    props = MaterialProperties(
        wet_grip_coefficient=0.70,
        rolling_resistance_coeff=0.012,
        abrasion_resistance=0.75,
    )

    compound = TireCompound(
        compound_id="DEMO_001",
        name="Demo Compound",
        compound_type=CompoundType.SYNTHETIC_RUBBER,
        base_properties=props,
        additives={"silica": 0.15, "carbon_black": 0.25, "sulfur": 0.02},
        quantum_optimization_level=0.0,
    )

    print("Initial Compound Properties:")
    print(f"  Wet Grip: {props.wet_grip_coefficient:.3f}")
    print(f"  Rolling Resistance: {props.rolling_resistance_coeff:.4f}")
    print(f"  Abrasion Resistance: {props.abrasion_resistance:.3f}")
    print(f"  Quantum Optimization Level: {compound.quantum_optimization_level:.1f}")
    print()

    # Apply quantum optimization
    print("Applying quantum optimization...")
    target_props = {"wet_grip": 0.85, "rolling_resistance": 0.008, "abrasion": 0.90}

    result = compound.apply_quantum_optimization(
        target_properties=target_props, optimization_iterations=50
    )

    print()
    print("Optimization Results:")
    print(f"  Algorithm: {result['algorithm']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Converged: {result['convergence']}")
    print(f"  Quantum Optimization Level: {compound.quantum_optimization_level:.1f}")
    print()


def demo_scenario_diversity():
    """Demonstrate scenario diversity and coverage."""
    print("=" * 80)
    print("DEMO 5: Scenario Diversity")
    print("=" * 80)
    print()

    from quasim.domains.tire.generator import TireScenarioGenerator

    gen = TireScenarioGenerator(random_seed=42)

    # Generate diverse scenarios
    print("Generating diverse scenarios...")
    scenarios = gen.generate_scenarios(target_count=1000)

    # Analyze diversity
    tire_types = set(s["geometry"].tire_type.value for s in scenarios)
    compound_types = set(s["compound"].compound_type.value for s in scenarios)
    surface_types = set(s["environment"].surface_type.value for s in scenarios)
    weather_types = set(s["environment"].weather.value for s in scenarios)

    print(f"Generated {len(scenarios)} scenarios with:")
    print(f"  {len(tire_types)} tire types: {', '.join(tire_types)}")
    print(f"  {len(compound_types)} compound types: {', '.join(compound_types)}")
    print(f"  {len(surface_types)} surface types: {', '.join(sorted(surface_types))}")
    print(f"  {len(weather_types)} weather conditions: {', '.join(weather_types)}")
    print()

    # Temperature range
    temps = [s["environment"].ambient_temperature for s in scenarios]
    print(f"Temperature Range: {min(temps):.1f}°C to {max(temps):.1f}°C")

    # Speed range
    speeds = [s["speed_kmh"] for s in scenarios]
    print(f"Speed Range: {min(speeds):.0f} km/h to {max(speeds):.0f} km/h")

    # Load range
    loads = [s["load_kg"] for s in scenarios]
    print(f"Load Range: {min(loads):.0f} kg to {max(loads):.0f} kg")
    print()


def main():
    """Run all demonstrations."""
    print()
    print("=" * 80)
    print("QuASIM Tire Simulation Library - Comprehensive Demonstration")
    print("Goodyear Quantum Pilot Platform Integration")
    print("=" * 80)
    print()

    # Run demonstrations
    demo_scenario_diversity()
    demo_quantum_optimization()
    demo_material_search()
    demo_basic_tire_library()
    demo_goodyear_integration()

    print("=" * 80)
    print("All Demonstrations Complete!")
    print("=" * 80)
    print()
    print("The tire simulation library is ready for:")
    print("  - CAD system integration")
    print("  - FEA tool compatibility")
    print("  - AI/ML training datasets")
    print("  - Digital twin platforms")
    print("  - Predictive maintenance systems")
    print()
    print("For production use with 10,000+ scenarios, use:")
    print("  quasim-tire goodyear --use-all --scenarios-per-material 10")
    print()


if __name__ == "__main__":
    main()
