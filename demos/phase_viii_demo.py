#!/usr/bin/env python3
"""Phase VIII Autonomous Governance Demo.

Demonstrates the complete integration of:
- Meta-Controller Kernel (MCK)
- Policy Reasoner (PR)
- Quantum Ethical Governor (QEG)
"""

from quasim.meta import MetaControllerKernel, QuantumEthicalGovernor
from quasim.policy import ConfigurationMutation, PolicyDecision, PolicyReasoner


def main():
    """Run Phase VIII integration demo."""

    print("=" * 70)
    print("QuASIM Phase VIII: Autonomous Governance Demo")
    print("=" * 70)
    print()

    # Initialize components
    print("Initializing Phase VIII components...")
    mck = MetaControllerKernel(seed=42)
    pr = PolicyReasoner()
    qeg = QuantumEthicalGovernor(
        energy_budget=1000.0, equity_threshold=0.3, min_sustainability_score=75.0
    )
    print("✓ Meta-Controller Kernel initialized")
    print("✓ Policy Reasoner initialized")
    print("✓ Quantum Ethical Governor initialized")
    print()

    # Simulate autonomous control loop
    print("-" * 70)
    print("SIMULATION: Autonomous Control Loop (5 episodes)")
    print("-" * 70)
    print()

    for episode in range(1, 6):
        print(f"Episode {episode}:")

        # MCK observes system state
        phi_variance = 0.30 - (episode - 1) * 0.05
        compliance_score = 97.0 + episode * 0.2
        resource_utilization = 0.70 + episode * 0.02
        error_rate = 0.03 - episode * 0.005

        state = mck.observe_state(
            phi_variance=phi_variance,
            compliance_score=compliance_score,
            resource_utilization=resource_utilization,
            error_rate=error_rate,
        )

        print("  System State:")
        print(f"    Φ variance: {state.phi_variance:.3f}")
        print(f"    Compliance: {state.compliance_score:.1f}%")
        print(f"    Resources: {state.resource_utilization:.1%}")
        print(f"    Error rate: {state.error_rate:.3f}")

        # MCK selects action
        action = mck.select_action(state, epsilon=0.2)
        print(f"  MCK Action: {action.parameter_name} {action.adjustment:+.2f} ({action.reason})")

        # Policy Reasoner evaluates proposed change
        mutation = ConfigurationMutation(
            parameter=action.parameter_name,
            current_value=0.0,
            proposed_value=action.adjustment,
            rationale=f"MCK optimization (episode {episode})",
            requestor="meta_controller_kernel",
        )

        evaluation = pr.evaluate_mutation(mutation)
        print(f"  Policy Decision: {evaluation.decision.value.upper()}")

        if evaluation.decision == PolicyDecision.CONDITIONAL:
            print(f"    Required approvers: {', '.join(evaluation.approved_by)}")
        elif evaluation.decision == PolicyDecision.REJECTED:
            print(f"    Rejection reasons: {len(evaluation.violated_rules)} rule violations")

        # Simulate state transition
        next_phi_variance = max(0.05, phi_variance - 0.03)
        next_compliance_score = min(100.0, compliance_score + 0.3)
        next_resource_utilization = min(0.85, resource_utilization + 0.02)
        next_error_rate = max(0.001, error_rate - 0.005)

        next_state = mck.observe_state(
            phi_variance=next_phi_variance,
            compliance_score=next_compliance_score,
            resource_utilization=next_resource_utilization,
            error_rate=next_error_rate,
        )

        # MCK updates Q-values
        reward = mck.compute_reward(state, next_state)
        mck.update_q_value(state, action, reward, next_state)
        print(f"  Reward: {reward:+.2f}")

        # QEG monitors resources
        resource_metrics = qeg.monitor_resources(
            energy_consumption=50.0 + episode * 10,
            compute_time=120.0 + episode * 10,
            memory_usage=8.0 + episode * 0.5,
            network_bandwidth=100.0,
        )

        # QEG assesses fairness
        fairness_metrics = qeg.assess_fairness(
            resource_distribution=[100.0] * 4,
            access_counts=[10] * 4,
            priority_levels=[1] * 4,
        )

        # QEG computes ethical score
        assessment = qeg.compute_ethical_score(resource_metrics, fairness_metrics)
        print(f"  Ethics Score: {assessment.ethics_score:.1f}/100.0")
        print(f"    Energy efficiency: {assessment.energy_efficiency:.1f}/100.0")
        print(f"    Equity balance: {assessment.equity_balance:.1f}/100.0")

        if assessment.violations:
            print(f"    ⚠️  Violations: {len(assessment.violations)}")

        print()

    # Display performance summary
    print("-" * 70)
    print("PERFORMANCE SUMMARY")
    print("-" * 70)
    print()

    mck_metrics = mck.get_performance_metrics()
    print("Meta-Controller Kernel:")
    print(f"  Episodes: {mck_metrics['episodes']}")
    print(f"  Average reward: {mck_metrics['avg_reward']:.2f}")
    print(f"  Φ variance reduction: {mck_metrics['phi_variance_reduction']:.3f}")
    print(f"  Compliance maintained: {mck_metrics['compliance_maintained']}")
    print(f"  Q-table size: {mck_metrics['q_table_size']}")
    print()

    pr_stats = pr.get_statistics()
    print("Policy Reasoner:")
    print(f"  Total rules: {pr_stats['total_rules']}")
    print("  Rules by framework:")
    for framework, count in pr_stats["rules_by_framework"].items():
        print(f"    {framework}: {count} rules")
    print()

    qeg_summary = qeg.get_performance_summary()
    print("Quantum Ethical Governor:")
    print(f"  Assessments: {qeg_summary['assessments_count']}")
    print(f"  Average ethics score: {qeg_summary['avg_ethics_score']:.1f}/100.0")
    print(f"  Latest ethics score: {qeg_summary['latest_ethics_score']:.1f}/100.0")
    print(f"  Violations: {qeg_summary['violations_count']}")
    print()

    # Emit to DVL
    print("-" * 70)
    print("DVL EMISSION")
    print("-" * 70)
    print()

    latest_assessment = qeg.assessment_history[-1]
    dvl_record = qeg.emit_to_dvl(latest_assessment)
    print(f"Record Type: {dvl_record['record_type']}")
    print(f"Ethics Score: {dvl_record['ethics_score']:.1f}/100.0")
    print(f"Attestation: {dvl_record['attestation']}")
    print(f"Timestamp: {dvl_record['timestamp']}")
    print()

    print("=" * 70)
    print("✅ Phase VIII Autonomous Governance Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
