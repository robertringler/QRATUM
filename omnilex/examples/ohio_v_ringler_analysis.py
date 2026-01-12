#!/usr/bin/env python3
"""
QRATUM-JURIS Example: STATE OF OHIO v. RINGLER, ROBERT
Holmes County Case No. 17CR052

This example demonstrates the full QRATUM-JURIS forensic legal analysis
for a real Ohio criminal case.

IMPORTANT: This analysis is for INFORMATIONAL PURPOSES ONLY and does NOT
constitute legal advice.
"""

import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from omnilex.juris import (
    QRATUMJurisEngine,
    create_ohio_v_ringler_intent,
)


def main():
    """Run QRATUM-JURIS analysis on Ohio v. Ringler case."""

    print("=" * 80)
    print("QRATUM-JURIS: Ultra-Detailed Legal Analysis")
    print("STATE OF OHIO v. RINGLER, ROBERT")
    print("Holmes County Case No. 17CR052")
    print("=" * 80)
    print()

    # Initialize QRATUM-JURIS engine
    engine = QRATUMJurisEngine()

    # Create intent for Ohio v. Ringler case
    intent = create_ohio_v_ringler_intent()

    print(f"Intent ID: {intent.intent_id}")
    print(f"Case: {intent.case_name}")
    print(f"Jurisdiction: {intent.jurisdiction}")
    print()

    # Execute full forensic analysis
    print("Executing forensic legal analysis...")
    print("-" * 80)

    result = engine.analyze_criminal_case(intent)

    # Display disclaimer
    print(result["disclaimer"])

    # Display Phase I: Public Record Exhaustion
    print("\n" + "=" * 80)
    print("PHASE I: PUBLIC RECORD EXHAUSTION")
    print("=" * 80)

    phase_i = result["phase_i_public_record_exhaustion"]

    print("\nA. Sources Enumerated:")
    for source in phase_i["sources_enumerated"]:
        print(f"   • {source}")

    print("\nB. Source-by-Source Disclosure Table:")
    print("-" * 80)
    print(f"{'Source':<25} {'Searched':<10} {'Accessible':<15} {'Information Found':<30}")
    print("-" * 80)
    for row in phase_i["source_disclosure_table"]:
        info = (
            row["information_found"][:30] + "..."
            if len(row["information_found"]) > 30
            else row["information_found"]
        )
        print(f"{row['source']:<25} {str(row['searched']):<10} {row['accessible']:<15} {info:<30}")

    print("\nC. Negative Certification:")
    print(f"   {phase_i['negative_certification']}")

    print("\nD. Information Completeness Rating:")
    print(f"   Rating: {phase_i['completeness_rating']}")
    print(f"   Justification: {phase_i['completeness_justification']}")

    # Display Phase II: Legal Analysis
    print("\n" + "=" * 80)
    print("PHASE II: LEGAL ANALYSIS")
    print("=" * 80)

    phase_ii = result["phase_ii_legal_analysis"]

    # 1. Procedural Posture
    print("\n1️⃣ PROCEDURAL POSTURE RECONSTRUCTION")
    print("-" * 60)
    posture = phase_ii["1_procedural_posture"]
    print("\nTimeline:")
    for event in posture["timeline"]:
        status = "✓" if event["status"] == "documented" else "?"
        print(f"   {status} {event['event']}: {event['date']}")

    if posture["gaps_identified"]:
        print("\nGaps Identified:")
        for gap in posture["gaps_identified"]:
            print(f"   ⚠ {gap['missing_event']}: {gap['legal_significance']}")

    if posture["unusual_timing"]:
        print("\nUnusual Timing:")
        for issue in posture["unusual_timing"]:
            print(f"   ⚠ {issue[:100]}...")

    # 2. Charge-by-Charge Analysis
    print("\n2️⃣ CHARGE-BY-CHARGE LEGAL ANALYSIS")
    print("-" * 60)
    for charge_analysis in phase_ii["2_charge_analysis"]:
        print(f"\n📋 {charge_analysis.get('charge', 'Unknown')}")
        print(f"   Statute: {charge_analysis.get('statute_code', 'N/A')}")
        print(f"   Offense Level: {charge_analysis.get('offense_level', 'N/A')}")
        print(f"   Mens Rea: {charge_analysis.get('mens_rea', 'N/A')}")
        if "elements" in charge_analysis:
            print("   Elements:")
            for elem in charge_analysis["elements"]:
                print(f"      • {elem}")

    # 3. Constitutional Issues
    print("\n3️⃣ CONSTITUTIONAL & STATUTORY ISSUE SPOTTING")
    print("-" * 60)
    for issue in phase_ii["3_constitutional_issues"]:
        print(f"\n⚖️ {issue['amendment']}")
        print(f"   Issue: {issue['issue']}")
        print(f"   Analysis: {issue['analysis'][:100]}...")
        print(f"   Potential Violation: {issue['potential_violation']}")

    # 4. Speedy Trial
    print("\n4️⃣ SPEEDY TRIAL COMPUTATION")
    print("-" * 60)
    st = phase_ii["4_speedy_trial_computation"]
    print(f"   Statutory Limit: {st['statutory_limit_days']} days")
    print(f"   Highest Charge: {st['highest_charge_level']}")
    print(f"   Filing Date: {st['filing_date']}")
    print(f"   Plea Date: {st['plea_date']}")
    print(f"   Waiver Analysis: {st['waiver_analysis'][:100]}...")

    # 5. Plea & Sentencing Forensics
    print("\n5️⃣ PLEA & SENTENCING FORENSICS")
    print("-" * 60)
    psf = phase_ii["5_plea_sentencing_forensics"]
    sent = psf["sentencing_legality"]
    print(f"   Charge: {sent['charge']}")
    print(f"   Statutory Range: {sent['statutory_range']}")
    print(f"   Sentence Imposed: {sent['sentence_imposed']}")
    print(f"   Within Range: {sent['within_range']}")

    jrc = psf["judicial_release_compliance"]
    print("\n   Judicial Release Analysis:")
    print(f"   Potential Issue: {jrc['potential_issue'][:100]}...")

    # 6. Adversarial Dual Analysis
    print("\n6️⃣ ADVERSARIAL DUAL ANALYSIS")
    print("-" * 60)
    for issue_analysis in phase_ii["6_adversarial_dual_analysis"]["issues_analyzed"]:
        print(f"\n⚔️ {issue_analysis['issue']}")
        print(f"   Defense Strength: {issue_analysis['defense_optimal']['strength']:.0%}")
        print(f"   Prosecution Strength: {issue_analysis['prosecution_optimal']['strength']:.0%}")
        print(f"   Stronger Position: {issue_analysis['stronger_position']}")

    # 7. Outcome Probability Matrix
    print("\n7️⃣ OUTCOME PROBABILITY MATRIX")
    print("-" * 60)
    print(f"{'Outcome':<25} {'Probability':<15} {'Legal Basis':<40}")
    print("-" * 80)
    for outcome in phase_ii["7_outcome_probability_matrix"]:
        print(
            f"{outcome['outcome']:<25} {outcome['probability_percent']}%{'':<10} {outcome['legal_basis'][:40]}"
        )

    # 8. Post-Conviction Review
    print("\n8️⃣ POST-CONVICTION & COLLATERAL REVIEW")
    print("-" * 60)
    pcr = phase_ii["8_post_conviction_review"]
    print(f"   Crim.R. 32.1 Viability: {pcr['crim_r_32_1']['viability']}")
    print(f"   R.C. 2953.21 Viability: {pcr['rc_2953_21']['viability']}")
    print(f"   IAC Viability: {pcr['ineffective_assistance']['viability']}")
    print("\n   Sealing Eligibility:")
    seal = pcr["sealing_eligibility"]
    print(f"      Assault (F4): {seal['ineligible']}")
    for eligible in seal["eligible_offenses"]:
        print(f"      ✓ {eligible}")

    # 9. Appellate Risk Profile
    print("\n9️⃣ APPELLATE RISK PROFILE")
    print("-" * 60)
    arp = phase_ii["9_appellate_risk_profile"]
    print("   De Novo Issues:")
    for issue in arp["de_novo_issues"][:3]:
        print(f"      • {issue}")
    print(f"\n   Likely Appellate Posture: {arp['likely_appellate_posture'][:100]}...")

    # 10. Final Audit Statement
    print("\n🔟 FINAL CONFIDENCE & AUDIT STATEMENT")
    print("-" * 60)
    audit = phase_ii["10_final_audit_statement"]
    print(f"   Overall Confidence: {audit['overall_confidence']}")
    print("\n   Top 3 Facts That Could Flip Outcome:")
    for i, fact in enumerate(audit["top_3_facts_that_could_flip_outcome"], 1):
        print(f"      {i}. {fact['fact']}")
        print(f"         Impact: {fact['impact']}")

    print("\n   Documents Critical to Obtain:")
    for doc in audit["documents_critical_to_obtain"]:
        print(f"      📄 {doc}")

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResult Hash: {result['result_hash'][:32]}...")
    print(f"Intent Hash: {result['intent_hash'][:32]}...")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Version: {result['version']}")

    # Save full report to JSON
    output_path = Path(__file__).parent / "ohio_v_ringler_report.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n✓ Full report saved to: {output_path}")

    print("\n" + "=" * 80)
    print("⚖️  IMPORTANT: This analysis is for INFORMATIONAL PURPOSES ONLY.")
    print("   It does NOT constitute legal advice.")
    print("   Consult qualified legal counsel for definitive legal advice.")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
