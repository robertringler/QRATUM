"""
QRATUM Platform Examples - All Verticals Demo

Demonstrates all 14 vertical modules in action.
"""

from qratum.platform import PlatformIntent, PlatformOrchestrator
from qratum.verticals import (
    CapraModule,
    ChronaModule,
    CohoraModule,
    EcoraModule,
    FluxaModule,
    FusiaModule,
    GeonaModule,
    JurisModule,
    NeuraModule,
    OrbiaModule,
    SentraModule,
    StrataModule,
    VexorModule,
    VitraModule,
)


def main():
    """Run comprehensive demo of all 14 verticals"""

    print("=" * 80)
    print("QRATUM Platform - All 14 Verticals Demonstration")
    print("=" * 80)
    print()

    # Initialize orchestrator and register all verticals
    orchestrator = PlatformOrchestrator()

    modules = [
        ("JURIS", JurisModule()),
        ("VITRA", VitraModule()),
        ("ECORA", EcoraModule()),
        ("CAPRA", CapraModule()),
        ("SENTRA", SentraModule()),
        ("NEURA", NeuraModule()),
        ("FLUXA", FluxaModule()),
        ("CHRONA", ChronaModule()),
        ("GEONA", GeonaModule()),
        ("FUSIA", FusiaModule()),
        ("STRATA", StrataModule()),
        ("VEXOR", VexorModule()),
        ("COHORA", CohoraModule()),
        ("ORBIA", OrbiaModule()),
    ]

    for name, module in modules:
        orchestrator.register_vertical(name, module)

    print(f"✓ Registered {len(modules)} vertical modules")
    print()

    # Execute one task from each vertical
    demo_tasks = [
        ("JURIS", "analyze_contract", {"contract_text": "Sample legal contract"}),
        ("VITRA", "analyze_sequence", {"sequence": "ATCGATCG", "type": "dna"}),
        ("ECORA", "model_climate", {"scenario": "SSP2-4.5"}),
        ("CAPRA", "price_derivative", {"option_type": "call", "spot_price": 100}),
        ("SENTRA", "simulate_trajectory", {"initial_velocity_ms": 1000, "launch_angle_deg": 45}),
        ("NEURA", "decode_bci", {"signal_type": "EEG"}),
        ("FLUXA", "optimize_routes", {"num_stops": 10}),
        ("CHRONA", "simulate_circuit", {"node": "7nm"}),
        ("GEONA", "analyze_satellite_imagery", {"resolution": 10}),
        ("FUSIA", "simulate_plasma", {"temperature_kev": 15}),
        ("STRATA", "model_economy", {"country": "USA"}),
        ("VEXOR", "detect_threats", {"network": "corporate"}),
        ("COHORA", "coordinate_swarm", {"size": 50}),
        ("ORBIA", "propagate_orbit", {"altitude_km": 400}),
    ]

    results = []

    for vertical, task, params in demo_tasks:
        print(f"Executing: {vertical}.{task}")

        intent = PlatformIntent(
            vertical=vertical,
            task=task,
            parameters=params,
            requester_id="demo_user",
        )

        contract = orchestrator.submit_intent(intent)
        result = orchestrator.execute_contract(contract)
        results.append((vertical, task, result))

        print(f"  ✓ Completed with substrate recommendations: {result['recommended_substrates']}")

    print()
    print("=" * 80)
    print("Execution Summary")
    print("=" * 80)

    status = orchestrator.get_platform_status()
    print(f"Total contracts executed: {status['contracts_executed']}")
    print(f"Event chain length: {status['event_chain_length']}")
    print(
        f"Event chain integrity: {'✓ VERIFIED' if status['event_chain_integrity'] else '✗ FAILED'}"
    )
    print()

    # Show sample results from each vertical
    print("=" * 80)
    print("Sample Results from Each Vertical")
    print("=" * 80)
    print()

    for vertical, task, result in results:
        print(f"{vertical} ({task}):")
        output = result["output"]
        # Print first few keys from output
        for key in list(output.keys())[:3]:
            print(f"  {key}: {output[key]}")
        print()

    print("=" * 80)
    print("All verticals demonstrated successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
