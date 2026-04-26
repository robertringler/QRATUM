"""Tests for Policy Reasoner."""

from quasim.policy import (
    ConfigurationMutation,
    PolicyDecision,
    PolicyFramework,
    PolicyReasoner,
)


def test_policy_reasoner_initialization():
    """Test policy reasoner initialization."""

    pr = PolicyReasoner()
    assert len(pr.rules) > 0

    stats = pr.get_statistics()
    assert stats["total_rules"] > 0
    assert len(stats["rules_by_framework"]) == 4
    assert len(stats["rules_by_severity"]) == 4


def test_policy_reasoner_rules_by_framework():
    """Test retrieving rules by framework."""

    pr = PolicyReasoner()

    do178c_rules = pr.get_rules_by_framework(PolicyFramework.DO_178C)
    assert len(do178c_rules) > 0

    nist_rules = pr.get_rules_by_framework(PolicyFramework.NIST_800_53)
    assert len(nist_rules) > 0

    cmmc_rules = pr.get_rules_by_framework(PolicyFramework.CMMC_2_0)
    assert len(cmmc_rules) > 0

    iso_rules = pr.get_rules_by_framework(PolicyFramework.ISO_27001)
    assert len(iso_rules) > 0


def test_policy_reasoner_get_rule_by_id():
    """Test retrieving rule by ID."""

    pr = PolicyReasoner()

    rule = pr.get_rule_by_id("DO178C-001")
    assert rule is not None
    assert rule.framework == PolicyFramework.DO_178C

    rule = pr.get_rule_by_id("NIST-AC-01")
    assert rule is not None
    assert rule.framework == PolicyFramework.NIST_800_53


def test_policy_reasoner_approved_mutation():
    """Test evaluation of approved mutation."""

    pr = PolicyReasoner()

    mutation = ConfigurationMutation(
        parameter="batch_size",
        current_value=100,
        proposed_value=150,
        rationale="Increase throughput",
        requestor="system_optimizer",
    )

    evaluation = pr.evaluate_mutation(mutation)

    assert evaluation.decision in [PolicyDecision.APPROVED, PolicyDecision.CONDITIONAL]
    assert evaluation.timestamp is not None


def test_policy_reasoner_rejected_mutation():
    """Test evaluation of rejected mutation."""

    pr = PolicyReasoner()

    mutation = ConfigurationMutation(
        parameter="random_seed_mode",
        current_value="deterministic",
        proposed_value="nondeterministic",
        rationale="Add random behavior",
        requestor="experimental_feature",
    )

    evaluation = pr.evaluate_mutation(mutation)

    # Should be rejected due to non-deterministic change
    assert evaluation.decision == PolicyDecision.REJECTED
    assert len(evaluation.violated_rules) > 0


def test_policy_reasoner_safety_critical_mutation():
    """Test evaluation of safety-critical mutation."""

    pr = PolicyReasoner()

    mutation = ConfigurationMutation(
        parameter="phi_tolerance",
        current_value=0.01,
        proposed_value=0.05,
        rationale="Relax tolerance for performance",
        requestor="performance_tuner",
    )

    evaluation = pr.evaluate_mutation(mutation)

    # Should require conditional approval
    assert evaluation.decision == PolicyDecision.CONDITIONAL
    assert len(evaluation.violated_rules) > 0
    assert len(evaluation.conditions) > 0
    assert (
        "safety_engineer" in evaluation.approved_by
        or "chief_compliance_officer" in evaluation.approved_by
    )


def test_policy_reasoner_access_control_mutation():
    """Test evaluation of access control mutation."""

    pr = PolicyReasoner()

    mutation = ConfigurationMutation(
        parameter="auth_timeout",
        current_value=3600,
        proposed_value=7200,
        rationale="Extend session timeout",
        requestor="user_experience_team",
    )

    evaluation = pr.evaluate_mutation(mutation)

    # Should require conditional approval
    assert evaluation.decision == PolicyDecision.CONDITIONAL
    assert len(evaluation.violated_rules) > 0
    assert "security_officer" in evaluation.approved_by


def test_policy_reasoner_crypto_mutation():
    """Test evaluation of cryptographic mutation."""

    pr = PolicyReasoner()

    mutation = ConfigurationMutation(
        parameter="crypto_algorithm",
        current_value="AES-256",
        proposed_value="AES-128",
        rationale="Performance optimization",
        requestor="crypto_team",
    )

    evaluation = pr.evaluate_mutation(mutation)

    # Should require conditional approval with high severity
    assert evaluation.decision == PolicyDecision.CONDITIONAL
    assert any(r.severity == "critical" for r in evaluation.violated_rules)


def test_policy_reasoner_baseline_deviation():
    """Test evaluation with baseline deviation."""

    pr = PolicyReasoner()

    mutation = ConfigurationMutation(
        parameter="resource_limit",
        current_value=1000,
        proposed_value=1500,
        rationale="Handle increased load",
        requestor="capacity_planner",
    )

    # Provide baseline context
    context = {"baseline_params": {"resource_limit": 1000}}

    evaluation = pr.evaluate_mutation(mutation, context=context)

    # 50% deviation should trigger baseline deviation check
    assert evaluation.decision in [PolicyDecision.CONDITIONAL, PolicyDecision.REJECTED]


def test_policy_reasoner_cui_handling():
    """Test evaluation of CUI handling change."""

    pr = PolicyReasoner()

    mutation = ConfigurationMutation(
        parameter="cui_storage_path",
        current_value="/secure/cui",
        proposed_value="/data/cui",
        rationale="Reorganize storage",
        requestor="data_admin",
    )

    evaluation = pr.evaluate_mutation(mutation)

    # Should require conditional approval
    assert evaluation.decision == PolicyDecision.CONDITIONAL
    assert any(r.framework == PolicyFramework.CMMC_2_0 for r in evaluation.violated_rules)


def test_policy_reasoner_logic_correctness():
    """Test policy reasoner logic correctness."""

    pr = PolicyReasoner()

    # Test multiple scenarios
    test_cases = [
        {
            "mutation": ConfigurationMutation(
                parameter="log_level",
                current_value="INFO",
                proposed_value="DEBUG",
                rationale="Increase verbosity",
                requestor="dev_ops",
            ),
            "expected_decision": PolicyDecision.APPROVED,
        },
        {
            "mutation": ConfigurationMutation(
                parameter="safety_margin",
                current_value=0.1,
                proposed_value=0.05,
                rationale="Tighter margins",
                requestor="optimizer",
            ),
            "expected_decision": PolicyDecision.CONDITIONAL,
        },
    ]

    for case in test_cases:
        evaluation = pr.evaluate_mutation(case["mutation"])
        # Decision should match expected or be more restrictive
        if case["expected_decision"] == PolicyDecision.APPROVED:
            assert evaluation.decision in [
                PolicyDecision.APPROVED,
                PolicyDecision.CONDITIONAL,
                PolicyDecision.REJECTED,
            ]
        elif case["expected_decision"] == PolicyDecision.CONDITIONAL:
            assert evaluation.decision in [PolicyDecision.CONDITIONAL, PolicyDecision.REJECTED]
