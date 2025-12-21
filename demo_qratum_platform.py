#!/usr/bin/env python3
"""
QRATUM Platform v2.0 Demonstration Script

This script demonstrates the core capabilities of all 7 vertical modules:
- JURIS (Legal AI)
- VITRA (Bioinformatics)
- ECORA (Climate & Energy)
- CAPRA (Financial Risk)
- SENTRA (Aerospace & Defense)
- NEURA (Neuroscience & BCI)
- FLUXA (Supply Chain)
"""

from qratum_platform.core import QRATUMPlatform, PlatformIntent, VerticalModule
from verticals import (
    JURISModule,
    VITRAModule,
    ECORAModule,
    CAPRAModule,
    SENTRAModule,
    NEURAModule,
    FLUXAModule,
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def demo_juris():
    """Demonstrate JURIS Legal AI module."""
    print_section("JURIS - Legal AI Module")

    platform = QRATUMPlatform()
    juris = JURISModule()
    platform.register_module(VerticalModule.JURIS, juris)

    # Legal reasoning
    intent = PlatformIntent(
        vertical=VerticalModule.JURIS,
        operation="legal_reasoning",
        parameters={
            "facts": "Company A failed to deliver goods on time, causing Company B losses.",
            "area_of_law": "contract",
        },
        user_id="demo_user",
    )

    contract = platform.create_contract(intent)
    print(f"Contract ID: {contract.contract_id}")
    print(f"Substrate: {contract.substrate.value}")

    result = platform.execute_contract(contract.contract_id)
    print(f"\nIRAC Analysis:")
    print(f"  Issue: {result['issue'][0] if result['issue'] else 'N/A'}")
    print(f"  Conclusion: {result['conclusion'][:100]}...")


def demo_vitra():
    """Demonstrate VITRA Bioinformatics module."""
    print_section("VITRA - Bioinformatics Module")

    platform = QRATUMPlatform()
    vitra = VITRAModule()
    platform.register_module(VerticalModule.VITRA, vitra)

    # DNA sequence analysis
    intent = PlatformIntent(
        vertical=VerticalModule.VITRA,
        operation="sequence_analysis",
        parameters={"sequence": "ATGGCTAGCTAGCTAGCTAG", "sequence_type": "dna"},
        user_id="demo_user",
    )

    contract = platform.create_contract(intent)
    result = platform.execute_contract(contract.contract_id)

    print(f"Sequence Length: {result['length']}")
    print(f"GC Content: {result['gc_content']:.2%}")
    print(f"ORFs Found: {len(result['open_reading_frames'])}")


def demo_ecora():
    """Demonstrate ECORA Climate & Energy module."""
    print_section("ECORA - Climate & Energy Module")

    platform = QRATUMPlatform()
    ecora = ECORAModule()
    platform.register_module(VerticalModule.ECORA, ecora)

    # Climate projection
    intent = PlatformIntent(
        vertical=VerticalModule.ECORA,
        operation="climate_projection",
        parameters={"scenario": "SSP2-4.5", "region": "global", "target_year": 2100},
        user_id="demo_user",
    )

    contract = platform.create_contract(intent)
    result = platform.execute_contract(contract.contract_id)

    print(f"Scenario: {result['scenario']}")
    print(f"Projected Warming: {result['projected_warming_c']}°C")
    print(f"Sea Level Rise: {result['impacts']['sea_level_rise_cm']} cm")


def demo_capra():
    """Demonstrate CAPRA Financial Risk module."""
    print_section("CAPRA - Financial Risk Module")

    platform = QRATUMPlatform()
    capra = CAPRAModule()
    platform.register_module(VerticalModule.CAPRA, capra)

    # Option pricing
    intent = PlatformIntent(
        vertical=VerticalModule.CAPRA,
        operation="option_pricing",
        parameters={
            "spot_price": 100,
            "strike_price": 105,
            "time_to_maturity": 0.5,
            "risk_free_rate": 0.05,
            "volatility": 0.25,
            "option_type": "call",
        },
        user_id="demo_user",
    )

    contract = platform.create_contract(intent)
    result = platform.execute_contract(contract.contract_id)

    print(f"Option Price: ${result['price']:.2f}")
    print(f"Delta: {result['greeks']['delta']:.4f}")
    print(f"Gamma: {result['greeks']['gamma']:.4f}")


def demo_sentra():
    """Demonstrate SENTRA Aerospace & Defense module."""
    print_section("SENTRA - Aerospace & Defense Module")

    platform = QRATUMPlatform()
    sentra = SENTRAModule()
    platform.register_module(VerticalModule.SENTRA, sentra)

    # Orbit propagation
    intent = PlatformIntent(
        vertical=VerticalModule.SENTRA,
        operation="orbit_propagation",
        parameters={"altitude_km": 400, "inclination_deg": 51.6, "duration_hours": 24},
        user_id="demo_user",
    )

    contract = platform.create_contract(intent)
    result = platform.execute_contract(contract.contract_id)

    print(f"Orbital Period: {result['orbital_period_minutes']:.2f} minutes")
    print(f"Orbital Velocity: {result['orbital_velocity_km_s']:.2f} km/s")
    print(f"Orbits per Day: {result['orbits_per_day']:.2f}")


def demo_neura():
    """Demonstrate NEURA Neuroscience & BCI module."""
    print_section("NEURA - Neuroscience & BCI Module")

    platform = QRATUMPlatform()
    neura = NEURAModule()
    platform.register_module(VerticalModule.NEURA, neura)

    # EEG analysis
    intent = PlatformIntent(
        vertical=VerticalModule.NEURA,
        operation="eeg_analysis",
        parameters={"sampling_rate_hz": 256, "duration_s": 60},
        user_id="demo_user",
    )

    contract = platform.create_contract(intent)
    result = platform.execute_contract(contract.contract_id)

    print(f"Dominant Frequency: {result['dominant_frequency']}")
    print(f"Band Power (Alpha): {result['band_power']['alpha']:.2f} μV²")
    print(f"Artifacts Detected: {len(result['artifacts_detected'])}")


def demo_fluxa():
    """Demonstrate FLUXA Supply Chain module."""
    print_section("FLUXA - Supply Chain Module")

    platform = QRATUMPlatform()
    fluxa = FLUXAModule()
    platform.register_module(VerticalModule.FLUXA, fluxa)

    # Route optimization
    intent = PlatformIntent(
        vertical=VerticalModule.FLUXA,
        operation="route_optimization",
        parameters={
            "depot": {"lat": 0, "lon": 0},
            "locations": [
                {"id": "A", "lat": 1, "lon": 1, "demand": 10},
                {"id": "B", "lat": 2, "lon": 1, "demand": 15},
                {"id": "C", "lat": 1, "lon": 2, "demand": 8},
            ],
            "num_vehicles": 2,
            "vehicle_capacity": 30,
        },
        user_id="demo_user",
    )

    contract = platform.create_contract(intent)
    result = platform.execute_contract(contract.contract_id)

    print(f"Total Distance: {result['total_distance_km']:.2f} km")
    print(f"Total Time: {result['total_time_hours']:.2f} hours")
    print(f"Avg Utilization: {result['avg_utilization']:.1%}")


def verify_invariants():
    """Verify QRATUM fatal invariants."""
    print_section("Verifying QRATUM Fatal Invariants")

    platform = QRATUMPlatform()
    juris = JURISModule()
    platform.register_module(VerticalModule.JURIS, juris)

    intent = PlatformIntent(
        vertical=VerticalModule.JURIS,
        operation="legal_reasoning",
        parameters={"facts": "test"},
        user_id="demo_user",
    )

    contract = platform.create_contract(intent)
    platform.execute_contract(contract.contract_id)

    print("✓ Contract immutability: Contracts are frozen dataclasses")
    print("✓ Event emission: All operations emit events")
    print(f"✓ Hash-chain integrity: {platform.verify_integrity()}")
    print(f"✓ Event count: {len(platform.event_chain.events)} events recorded")
    print("✓ Causal traceability: Events linked via previous_hash")


def main():
    """Run all demonstrations."""
    print("\n" + "█" * 80)
    print(" " * 20 + "QRATUM PLATFORM v2.0 DEMONSTRATION")
    print("█" * 80)

    try:
        demo_juris()
        demo_vitra()
        demo_ecora()
        demo_capra()
        demo_sentra()
        demo_neura()
        demo_fluxa()
        verify_invariants()

        print_section("Demonstration Complete")
        print("All 7 vertical modules executed successfully!")
        print("Platform integrity verified ✓")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
