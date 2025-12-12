#!/usr/bin/env python3
"""QuASIM-SpaceX-NASA Integration Roadmap Demo.

This example demonstrates the key components of the 90-day integration roadmap:
1. Deterministic simulation with seed management
2. Telemetry ingestion from SpaceX and NASA
3. Monte-Carlo fidelity validation
4. Certification artifact generation
"""

from __future__ import annotations

import json

from quasim import Config, runtime
from seed_management import SeedManager
from telemetry_api import NASATelemetryAdapter, SpaceXTelemetryAdapter


def demo_deterministic_simulation():
    """Demonstrate deterministic simulation with QuASIM."""
    print("\n" + "=" * 70)
    print("Demo 1: Deterministic Simulation with Seed Management")
    print("=" * 70 + "\n")

    # Create seed manager
    seed_mgr = SeedManager(base_seed=42, environment="demo")

    # Generate seed for simulation
    seed_record = seed_mgr.generate_seed(replay_id="demo_sim_001")
    print(f"Generated seed: {seed_record.seed_value}")
    print(f"Timestamp: {seed_record.timestamp}")
    print(f"Hash: {seed_record.hash_digest[:16]}...")

    # Run simulation with seed
    cfg = Config(
        simulation_precision="fp64",
        max_workspace_mb=128,
        seed=seed_record.seed_value,
    )

    circuit = [
        [1 + 0j, 1 + 0j, 1 + 0j, 1 + 0j],
        [1 + 0j, 0 + 0j, 0 + 0j, 1 + 0j],
    ]

    with runtime(cfg) as rt:
        result = rt.simulate(circuit)
        print("\n✓ Simulation completed")
        print(f"  Result length: {len(result)}")
        print(f"  Average latency: {rt.average_latency * 1000:.3f}ms")

    # Verify seed record integrity
    is_valid = seed_record.verify()
    print(f"  Seed integrity: {'✓ Valid' if is_valid else '✗ Invalid'}")


def demo_spacex_telemetry():
    """Demonstrate SpaceX telemetry ingestion."""
    print("\n" + "=" * 70)
    print("Demo 2: SpaceX Falcon 9 Telemetry Ingestion")
    print("=" * 70 + "\n")

    adapter = SpaceXTelemetryAdapter(endpoint="localhost:8001")
    adapter.connect()

    # Simulate SpaceX telemetry data
    raw_data = {
        "timestamp": 125.5,
        "vehicle_id": "Falcon9_B1067",
        "altitude": 45000.0,
        "velocity": 2500.0,
        "downrange": 50.0,
        "thrust": 7607.0,  # kN
        "isp": 282.0,
        "throttle": 92.0,
        "attitude_q": [0.707, 0.707, 0.0, 0.0],
        "angular_rates": [0.1, 0.05, 0.02],
        "guidance_mode": "NOMINAL",
    }

    print("Raw telemetry data:")
    print(f"  T+{raw_data['timestamp']:.1f}s")
    print(f"  Altitude: {raw_data['altitude'] / 1000:.1f} km")
    print(f"  Velocity: {raw_data['velocity']:.1f} m/s")

    # Parse and validate
    telemetry = adapter.parse_telemetry(raw_data)
    is_valid, errors = adapter.validate_schema(telemetry)

    print("\n✓ Telemetry parsed successfully")
    print(f"  Vehicle: {telemetry.vehicle_id}")
    print(f"  GNC Mode: {telemetry.gnc_loops['guidance_mode']}")
    print(f"  Validation: {'✓ Valid' if is_valid else '✗ Invalid'}")

    if not is_valid:
        print(f"  Errors: {errors}")

    adapter.disconnect()


def demo_nasa_telemetry():
    """Demonstrate NASA telemetry ingestion."""
    print("\n" + "=" * 70)
    print("Demo 3: NASA Orion/SLS Telemetry Ingestion")
    print("=" * 70 + "\n")

    adapter = NASATelemetryAdapter(log_format="NASA_CSV_V2")

    # Simulate NASA CSV log entry
    csv_line = "125.5,Orion,6500000,0,0,7500,0,0,NOMINAL"

    print("CSV log entry:")
    print(f"  {csv_line}")

    # Parse and validate
    telemetry = adapter.parse_csv_log(csv_line)
    is_valid, errors = adapter.validate_schema(telemetry)

    print("\n✓ Telemetry parsed successfully")
    print(f"  Vehicle: {telemetry.vehicle_system}")
    print(f"  MET: {telemetry.met:.1f}s")
    print(
        f"  Position: [{telemetry.state_vector[0] / 1000:.0f}, "
        f"{telemetry.state_vector[1] / 1000:.0f}, "
        f"{telemetry.state_vector[2] / 1000:.0f}] km"
    )
    print(f"  GNC Mode: {telemetry.gnc_mode}")
    print(f"  Validation: {'✓ Valid' if is_valid else '✗ Invalid'}")

    if not is_valid:
        print(f"  Errors: {errors}")

    # Export to QuASIM format
    quasim_data = adapter.export_quasim_format(telemetry)
    print("\n✓ Exported to QuASIM format")
    print(f"  Source: {quasim_data['source']}")


def demo_montecarlo_validation():
    """Demonstrate Monte-Carlo fidelity validation."""
    print("\n" + "=" * 70)
    print("Demo 4: Monte-Carlo Fidelity Validation")
    print("=" * 70 + "\n")

    # Load Monte-Carlo results
    try:
        with open("montecarlo_campaigns/MC_Results_1024.json") as f:
            data = json.load(f)

        stats = data["statistics"]

        print("Monte-Carlo Campaign Statistics:")
        print(f"  Trajectories: {data['metadata']['num_trajectories']}")
        print(f"  Vehicles: {', '.join(data['metadata']['vehicles'])}")
        print(f"  Mean Fidelity: {stats['mean_fidelity']:.4f}")
        print(f"  Std Dev: {stats['fidelity_std']:.4f}")
        print(f"  Convergence Rate: {stats['convergence_rate']:.2%}")
        print(f"  Target: {stats['target_fidelity']} ± {stats['target_tolerance']}")

        if stats["acceptance_criteria_met"]:
            print("\n✓ Acceptance criteria MET")
        else:
            print("\n✗ Acceptance criteria NOT MET")

        # Check first few trajectories
        print("\nSample Trajectories:")
        for traj in data["trajectories"][:3]:
            status = "✓" if traj["converged"] else "✗"
            print(
                f"  {status} Trajectory {traj['trajectory_id']:04d}: "
                f"fidelity={traj['fidelity']:.4f}, "
                f"deviation={traj['nominal_deviation_pct']:+.2f}%"
            )

    except FileNotFoundError:
        print("✗ Monte-Carlo results not found. Run generate_quasim_jsons.py first.")


def demo_certification_package():
    """Demonstrate certification package validation."""
    print("\n" + "=" * 70)
    print("Demo 5: Certification Data Package Status")
    print("=" * 70 + "\n")

    try:
        with open("cdp_artifacts/CDP_v1.0.json") as f:
            data = json.load(f)

        package = data["package"]

        print("Certification Data Package:")
        print(f"  Package ID: {package['package_id']}")
        print(f"  Revision: {package['revision']}")
        print(f"  Status: {package['verification_status']}")
        print(f"  Open Anomalies: {package['open_anomalies']}")
        print(f"  Standard: {package['standard']}")

        print("\nVerification Evidence:")
        for evidence in data["verification_evidence"]:
            status_symbol = "✓" if evidence["status"] in ["Verified", "Submitted"] else "✗"
            print(f"  {status_symbol} {evidence['id']}: {evidence['description']}")
            print(f"     Status: {evidence['status']}")

        if package["open_anomalies"] == 0:
            print("\n✓ Package ready for audit (zero open anomalies)")
        else:
            print(f"\n✗ Package has {package['open_anomalies']} open anomalies")

    except FileNotFoundError:
        print("✗ Certification package not found. Run generate_quasim_jsons.py first.")


def main():
    """Run all demos."""
    print("\n" + "#" * 70)
    print("# QuASIM-SpaceX-NASA Integration Roadmap Demonstration")
    print("# DO-178C Level A / ECSS-Q-ST-80C / NASA E-HBK-4008")
    print("#" * 70)

    demo_deterministic_simulation()
    demo_spacex_telemetry()
    demo_nasa_telemetry()
    demo_montecarlo_validation()
    demo_certification_package()

    print("\n" + "=" * 70)
    print("✓ All demonstrations completed successfully")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
