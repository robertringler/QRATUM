#!/usr/bin/env python3
"""
QRATUM-OMNILEX Example: Contract Breach Analysis

This example demonstrates how to use OMNILEX to analyze a contract breach
scenario using the IRAC legal reasoning framework.
"""

import time

from omnilex import QRATUMOmniLexEngine, LegalQILIntent
from omnilex.qil_legal import generate_intent_id


def main():
    """Run a sample contract breach analysis."""

    print("=" * 80)
    print("QRATUM-OMNILEX v1.0 - Contract Breach Analysis Example")
    print("=" * 80)
    print()

    # Initialize the OMNILEX engine
    print("Initializing OMNILEX engine...")
    engine = QRATUMOmniLexEngine()
    print("✓ Engine initialized")
    print()

    # Define the factual scenario
    facts = """
    AlphaCorp and BetaServices entered into a written service agreement on
    January 1, 2024. The agreement required BetaServices to provide IT support
    services for 12 months with a response time of 4 hours for critical issues.

    On March 15, 2024, AlphaCorp experienced a critical server outage affecting
    production. AlphaCorp contacted BetaServices at 9:00 AM requesting immediate
    assistance. BetaServices did not respond until 6:00 PM the same day
    (9 hours later).

    The delay caused AlphaCorp to lose $50,000 in revenue from inability to
    process orders. AlphaCorp had previously notified BetaServices that timely
    response was essential due to e-commerce operations.
    """

    legal_question = """
    Did BetaServices breach the service agreement, and if so, what damages may
    AlphaCorp recover under California law?
    """

    print("Scenario: Service Agreement Breach")
    print("-" * 80)
    print(f"Facts: {facts.strip()}")
    print()
    print(f"Question: {legal_question.strip()}")
    print("-" * 80)
    print()

    # Create legal analysis intent
    print("Creating legal analysis intent...")
    intent = LegalQILIntent(
        intent_id=generate_intent_id("irac_analysis", "US-CA", time.time()),
        compute_task="irac_analysis",
        jurisdiction_primary="US-CA",
        jurisdictions_secondary=(),
        legal_domain="contract",
        reasoning_framework="irac",
        attorney_supervised=True,
        raw_facts=facts,
        legal_question=legal_question
    )
    print(f"✓ Intent created: {intent.intent_id}")
    print(f"  Intent hash: {intent.compute_hash()[:16]}...")
    print()

    # Submit for analysis
    print("Submitting intent to OMNILEX for analysis...")
    response = engine.submit_legal_intent(intent)
    print("✓ Analysis complete")
    print()

    # Display results
    result = response["result"]

    print("=" * 80)
    print("ANALYSIS RESULTS (IRAC)")
    print("=" * 80)
    print()

    print("ISSUE:")
    print("-" * 80)
    print(result["issue"])
    print()

    print("RULE:")
    print("-" * 80)
    print(result["rule"])
    print()
    print("Sources:", ", ".join(result["rule_sources"]))
    print()

    print("APPLICATION:")
    print("-" * 80)
    print(result["application"])
    print()

    print("CONCLUSION:")
    print("-" * 80)
    print(result["conclusion"])
    print()
    print(f"Confidence: {result['confidence']:.1%}")
    print()

    print("CAVEATS:")
    print("-" * 80)
    for caveat in result["caveats"]:
        print(f"  • {caveat}")
    print()

    # Display metadata
    print("=" * 80)
    print("ANALYSIS METADATA")
    print("=" * 80)
    print(f"Intent ID: {response['intent_id']}")
    print(f"Intent Hash: {response['intent_hash'][:32]}...")
    print(f"Result Hash: {response['result_hash'][:32]}...")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(response['timestamp']))}")
    print(f"Attorney Supervised: {response['attorney_supervised']}")
    print(f"OMNILEX Version: {response['version']}")
    print()

    # Display disclaimer
    print(response["disclaimer"])
    print()

    # Demonstrate replay capability
    print("=" * 80)
    print("DEMONSTRATING DETERMINISTIC REPLAY")
    print("=" * 80)
    print()
    print(f"Replaying analysis for intent: {intent.intent_id}")
    replayed = engine.replay_analysis(intent.intent_id)
    print("✓ Analysis replayed successfully")
    print(f"  Original hash: {response['result_hash'][:32]}...")
    print(f"  Replayed hash: {replayed['result_hash'][:32]}...")
    print(f"  Hashes match: {response['result_hash'] == replayed['result_hash']}")
    print()

    # Demonstrate audit capability
    print("=" * 80)
    print("DEMONSTRATING AUDIT TRAIL")
    print("=" * 80)
    print()
    print(f"Auditing analysis for intent: {intent.intent_id}")
    audit = engine.audit_analysis(intent.intent_id)
    print("✓ Audit complete")
    print(f"  Hash integrity valid: {audit['hash_integrity']['valid']}")
    print(f"  Contract immutability: {audit['invariants_satisfied']['contract_immutability']}")
    print(f"  Hash chain integrity: {audit['invariants_satisfied']['hash_chain_integrity']}")
    print(f"  Causal traceability: {audit['invariants_satisfied']['causal_traceability']}")
    print()

    print("=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print()
    print("This example demonstrated:")
    print("  ✓ Creating a legal analysis intent")
    print("  ✓ Submitting to OMNILEX for IRAC analysis")
    print("  ✓ Retrieving structured results")
    print("  ✓ Deterministic replay capability")
    print("  ✓ Audit trail verification")
    print()
    print("Remember: This analysis is for informational purposes only.")
    print("Always consult a licensed attorney for legal advice.")
    print()


if __name__ == "__main__":
    main()
