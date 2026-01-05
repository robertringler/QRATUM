"""
QRATUM Platform Examples - Simple Demo

Demonstrates basic usage of the QRATUM platform with multiple verticals.
"""

from qratum.platform import PlatformIntent, PlatformOrchestrator
from qratum.verticals import (
    CapraModule,
    EcoraModule,
    JurisModule,
    VitraModule,
)


def main():
    """Run a simple demo showcasing the platform"""

    print("=" * 80)
    print("QRATUM Sovereign AI Platform v2.0 - Demo")
    print("=" * 80)
    print()

    # Initialize orchestrator
    orchestrator = PlatformOrchestrator()

    # Register vertical modules
    print("Registering vertical modules...")
    orchestrator.register_vertical("JURIS", JurisModule())
    orchestrator.register_vertical("VITRA", VitraModule())
    orchestrator.register_vertical("ECORA", EcoraModule())
    orchestrator.register_vertical("CAPRA", CapraModule())
    print(f"✓ Registered {len(orchestrator.vertical_registry)} verticals")
    print()

    # Example 1: Legal contract analysis
    print("-" * 80)
    print("Example 1: JURIS - Legal Contract Analysis")
    print("-" * 80)

    intent1 = PlatformIntent(
        vertical="JURIS",
        task="analyze_contract",
        parameters={
            "contract_text": "This agreement shall indemnify the party against all liability. "
            "The party must deliver goods within 30 days. "
            "Failure to comply will result in termination and penalties."
        },
        requester_id="demo_user",
    )

    contract1 = orchestrator.submit_intent(intent1)
    print(f"✓ Contract created: {contract1.contract_id}")

    result1 = orchestrator.execute_contract(contract1)
    print("✓ Execution completed")
    print(f"\nRisks identified: {len(result1['output']['risks_identified'])}")
    for risk in result1["output"]["risks_identified"]:
        print(f"  - {risk}")
    print(f"\nSafety Disclaimer: {result1['safety_disclaimer'][:100]}...")
    print()

    # Example 2: Bioinformatics sequence analysis
    print("-" * 80)
    print("Example 2: VITRA - Genomic Sequence Analysis")
    print("-" * 80)

    intent2 = PlatformIntent(
        vertical="VITRA",
        task="analyze_sequence",
        parameters={
            "sequence": "ATCGATCGATCGATCGTAGCTAGCTAGCT",
            "type": "dna",
        },
        requester_id="demo_user",
    )

    contract2 = orchestrator.submit_intent(intent2)
    print(f"✓ Contract created: {contract2.contract_id}")

    result2 = orchestrator.execute_contract(contract2)
    print("✓ Execution completed")
    print(f"\nSequence length: {result2['output']['sequence_length']}")
    print(f"GC content: {result2['output']['gc_content']:.2%}")
    print(f"Predicted genes: {result2['output']['predicted_genes']}")
    print()

    # Example 3: Climate modeling
    print("-" * 80)
    print("Example 3: ECORA - Climate Modeling")
    print("-" * 80)

    intent3 = PlatformIntent(
        vertical="ECORA",
        task="model_climate",
        parameters={
            "scenario": "SSP2-4.5",
            "time_horizon_years": 30,
        },
        requester_id="demo_user",
    )

    contract3 = orchestrator.submit_intent(intent3)
    print(f"✓ Contract created: {contract3.contract_id}")

    result3 = orchestrator.execute_contract(contract3)
    print("✓ Execution completed")
    print(f"\nClimate scenario: {result3['output']['scenario']}")
    print(f"Projected temperature increase: {result3['output']['projected_temp_increase_c']}°C")
    print(f"Sea level rise: {result3['output']['sea_level_rise_cm']} cm")
    print()

    # Example 4: Financial derivatives pricing
    print("-" * 80)
    print("Example 4: CAPRA - Options Pricing")
    print("-" * 80)

    intent4 = PlatformIntent(
        vertical="CAPRA",
        task="price_derivative",
        parameters={
            "option_type": "call",
            "spot_price": 100,
            "strike_price": 105,
            "time_to_maturity": 0.5,
            "risk_free_rate": 0.05,
            "volatility": 0.25,
        },
        requester_id="demo_user",
    )

    contract4 = orchestrator.submit_intent(intent4)
    print(f"✓ Contract created: {contract4.contract_id}")

    result4 = orchestrator.execute_contract(contract4)
    print("✓ Execution completed")
    print(f"\nOption price: ${result4['output']['price']:.2f}")
    print(f"Delta: {result4['output']['delta']:.4f}")
    print(f"Gamma: {result4['output']['gamma']:.4f}")
    print()

    # Platform status
    print("=" * 80)
    print("Platform Status")
    print("=" * 80)

    status = orchestrator.get_platform_status()
    print(f"Contracts created: {status['contracts_created']}")
    print(f"Contracts executed: {status['contracts_executed']}")
    print(f"Event chain length: {status['event_chain_length']}")
    print(
        f"Event chain integrity: {'✓ VERIFIED' if status['event_chain_integrity'] else '✗ FAILED'}"
    )
    print()

    # Demonstrate replay capability
    print("=" * 80)
    print("Deterministic Replay Demonstration")
    print("=" * 80)

    replay = orchestrator.replay_contract(contract1.contract_id)
    print(f"Replayed contract: {replay['contract_id']}")
    print(f"Event count: {replay['event_count']}")
    print(f"Replay verified: {'✓ YES' if replay['replay_verified'] else '✗ NO'}")
    print()

    print("=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
