"""Reasoning demonstration for QRATUM-ASI.

This example demonstrates Q-MIND's multi-strategy reasoning
capabilities with deterministic, auditable inference chains.
"""

from qratum_asi import QRATUMASI
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.types import (ASISafetyLevel, AuthorizationType,
                                   ReasoningStrategy)


def main():
    """Run reasoning demonstration."""
    print("=" * 80)
    print("QRATUM-ASI: Q-MIND Reasoning Demonstration")
    print("=" * 80)
    print()

    # Initialize QRATUM-ASI
    print("Initializing QRATUM-ASI...")
    asi = QRATUMASI()
    print("✓ System initialized")
    print(f"  Merkle chain length: {asi.merkle_chain.get_chain_length()}")
    print()

    # Add some knowledge to Q-REALITY
    print("Populating Q-REALITY with knowledge...")
    contract1 = ASIContract(
        contract_id="contract_001",
        operation_type="add_knowledge",
        safety_level=ASISafetyLevel.ROUTINE,
        authorization_type=AuthorizationType.NONE,
        payload={},
    )

    asi.q_reality.add_knowledge_node(
        node_id="node_001",
        content={"statement": "All AI systems require oversight"},
        source_vertical="JURIS",
        confidence=0.95,
        provenance=["safety_principles"],
        contract=contract1,
    )

    asi.q_reality.add_knowledge_node(
        node_id="node_002",
        content={"statement": "QRATUM-ASI is an AI system"},
        source_vertical="AGI",
        confidence=1.0,
        provenance=["system_definition"],
        contract=contract1,
    )
    print(f"✓ Added {len(asi.q_reality.knowledge_nodes)} knowledge nodes")
    print()

    # Perform deductive reasoning
    print("Performing DEDUCTIVE reasoning...")
    contract2 = ASIContract(
        contract_id="contract_002",
        operation_type="reasoning",
        safety_level=ASISafetyLevel.ROUTINE,
        authorization_type=AuthorizationType.NONE,
        payload={},
    )

    chain = asi.q_mind.reason(
        query="Does QRATUM-ASI require oversight?",
        strategy=ReasoningStrategy.DEDUCTIVE,
        context={},
        contract=contract2,
    )

    print(f"  Query: {chain.query}")
    print(f"  Strategy: {chain.steps[0].strategy.value}")
    print(f"  Conclusion: {chain.final_conclusion}")
    print(f"  Confidence: {chain.overall_confidence:.2f}")
    print()

    # Perform causal reasoning
    print("Performing CAUSAL reasoning...")
    contract3 = ASIContract(
        contract_id="contract_003",
        operation_type="reasoning",
        safety_level=ASISafetyLevel.ROUTINE,
        authorization_type=AuthorizationType.NONE,
        payload={},
    )

    chain2 = asi.q_mind.reason(
        query="What is the effect of removing oversight?",
        strategy=ReasoningStrategy.CAUSAL,
        context={},
        contract=contract3,
    )

    print(f"  Query: {chain2.query}")
    print(f"  Strategy: {chain2.steps[0].strategy.value}")
    print(f"  Conclusion: {chain2.final_conclusion}")
    print(f"  Confidence: {chain2.overall_confidence:.2f}")
    print()

    # Cross-domain synthesis
    print("Performing CROSS-DOMAIN synthesis...")
    contract4 = ASIContract(
        contract_id="contract_004",
        operation_type="synthesis",
        safety_level=ASISafetyLevel.ELEVATED,
        authorization_type=AuthorizationType.SINGLE_HUMAN,
        payload={},
        authorized=True,
        authorized_by="demo_user",
    )

    synthesis = asi.q_mind.cross_domain_synthesis(
        domains=["JURIS", "AGI", "XENON"],
        synthesis_goal="Understand multi-domain safety requirements",
        contract=contract4,
    )

    print(f"  Goal: {synthesis['goal']}")
    print(f"  Domains: {', '.join(synthesis['domains'])}")
    print(f"  Insights: {len(synthesis['insights'])}")
    for insight in synthesis['insights']:
        print(f"    - {insight}")
    print()

    # Display system status
    print("=" * 80)
    print("System Status:")
    print("=" * 80)
    status = asi.get_system_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    print()

    print("✓ Reasoning demonstration complete!")


if __name__ == "__main__":
    main()
