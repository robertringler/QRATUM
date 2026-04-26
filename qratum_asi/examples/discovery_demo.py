"""Discovery demonstration for QRATUM-ASI.

This example demonstrates Q-FORGE's cross-domain hypothesis
generation and novel synthesis capabilities.
"""

from qratum_asi import QRATUMASI
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.types import ASISafetyLevel, AuthorizationType


def main():
    """Run discovery demonstration."""
    print("=" * 80)
    print("QRATUM-ASI: Q-FORGE Discovery Demonstration")
    print("=" * 80)
    print()

    # Initialize QRATUM-ASI
    print("Initializing QRATUM-ASI...")
    asi = QRATUMASI()
    print("✓ System initialized")
    print()

    # Populate Q-REALITY with cross-domain knowledge
    print("Populating Q-REALITY with cross-domain knowledge...")
    contract1 = ASIContract(
        contract_id="contract_001",
        operation_type="add_knowledge",
        safety_level=ASISafetyLevel.ROUTINE,
        authorization_type=AuthorizationType.NONE,
        payload={},
    )

    # Add knowledge from multiple domains
    asi.q_reality.add_knowledge_node(
        node_id="materials_001",
        content={"property": "quantum tunneling in polymers"},
        source_vertical="QUASIM",
        confidence=0.85,
        provenance=["materials_research"],
        contract=contract1,
    )

    asi.q_reality.add_knowledge_node(
        node_id="bio_001",
        content={"property": "protein folding mechanisms"},
        source_vertical="XENON",
        confidence=0.90,
        provenance=["molecular_dynamics"],
        contract=contract1,
    )

    asi.q_reality.add_knowledge_node(
        node_id="quantum_001",
        content={"property": "quantum coherence in biological systems"},
        source_vertical="QUBIC",
        confidence=0.75,
        provenance=["quantum_biology"],
        contract=contract1,
    )
    print(f"✓ Added {len(asi.q_reality.knowledge_nodes)} knowledge nodes")
    print()

    # Generate cross-domain hypothesis
    print("Generating CROSS-DOMAIN HYPOTHESIS...")
    contract2 = ASIContract(
        contract_id="contract_002",
        operation_type="generate_hypothesis",
        safety_level=ASISafetyLevel.ELEVATED,
        authorization_type=AuthorizationType.SINGLE_HUMAN,
        payload={},
        authorized=True,
        authorized_by="demo_user",
    )

    hypothesis = asi.q_forge.generate_hypothesis(
        hypothesis_id="hyp_001",
        description="Quantum effects in biological polymers may enable novel materials",
        domains=["QUASIM", "XENON", "QUBIC"],
        premises=[
            "Quantum tunneling observed in synthetic polymers",
            "Protein folding uses quantum coherence",
            "Biological systems maintain quantum states",
        ],
        contract=contract2,
    )

    print(f"  Hypothesis: {hypothesis.description}")
    print(f"  Domains: {', '.join(hypothesis.domains)}")
    print(f"  Confidence: {hypothesis.confidence:.2f}")
    print(f"  Novelty Score: {hypothesis.novelty_score:.2f}")
    print("  Predictions:")
    for pred in hypothesis.predictions:
        print(f"    - {pred}")
    print()

    # Make a discovery
    print("Making DISCOVERY...")
    contract3 = ASIContract(
        contract_id="contract_003",
        operation_type="make_discovery",
        safety_level=ASISafetyLevel.ELEVATED,
        authorization_type=AuthorizationType.SINGLE_HUMAN,
        payload={},
        authorized=True,
        authorized_by="demo_user",
    )

    discovery = asi.q_forge.make_discovery(
        discovery_id="disc_001",
        title="Bio-Inspired Quantum Materials",
        description="Novel material design based on quantum-biological principles",
        domains=["QUASIM", "XENON", "QUBIC"],
        supporting_evidence=[
            "Hypothesis hyp_001",
            "Experimental validation needed",
            "Cross-domain synthesis",
        ],
        contract=contract3,
    )

    print(f"  Discovery: {discovery.title}")
    print(f"  Description: {discovery.description}")
    print(f"  Domains: {', '.join(discovery.domains)}")
    print(f"  Confidence: {discovery.confidence:.2f}")
    print(f"  Novelty Score: {discovery.novelty_score:.2f}")
    print(f"  Validation Status: {discovery.validation_status}")
    print()

    # Validate discovery
    print("Validating DISCOVERY...")
    contract4 = ASIContract(
        contract_id="contract_004",
        operation_type="validate_discovery",
        safety_level=ASISafetyLevel.SENSITIVE,
        authorization_type=AuthorizationType.SINGLE_HUMAN,
        payload={},
        authorized=True,
        authorized_by="demo_user",
    )

    validated = asi.q_forge.validate_discovery(
        discovery_id="disc_001",
        validation_results={"passed": True, "score": 0.82},
        contract=contract4,
    )

    print(f"  Discovery: {validated.title}")
    print(f"  Validation Status: {validated.validation_status}")
    print()

    # Synthesize discoveries
    print("Synthesizing MULTIPLE DISCOVERIES...")
    contract5 = ASIContract(
        contract_id="contract_005",
        operation_type="synthesize_discoveries",
        safety_level=ASISafetyLevel.ELEVATED,
        authorization_type=AuthorizationType.SINGLE_HUMAN,
        payload={},
        authorized=True,
        authorized_by="demo_user",
    )

    synthesis = asi.q_forge.synthesize_discoveries(
        discovery_ids=["disc_001"],
        synthesis_goal="Identify novel research directions",
        contract=contract5,
    )

    print(f"  Goal: {synthesis['goal']}")
    print("  Novel Insights:")
    for insight in synthesis["novel_insights"]:
        print(f"    - {insight}")
    print(f"  Confidence: {synthesis['confidence']:.2f}")
    print(f"  Novelty Score: {synthesis['novelty_score']:.2f}")
    print()

    # Display system status
    print("=" * 80)
    print("System Status:")
    print("=" * 80)
    status = asi.get_system_status()
    print(f"  Hypotheses Generated: {len(asi.q_forge.hypotheses)}")
    print(f"  Discoveries Made: {len(asi.q_forge.discoveries)}")
    print(f"  Merkle Chain Length: {status['merkle_chain_length']}")
    print(f"  Merkle Chain Integrity: {status['merkle_chain_integrity']}")
    print()

    print("✓ Discovery demonstration complete!")


if __name__ == "__main__":
    main()
