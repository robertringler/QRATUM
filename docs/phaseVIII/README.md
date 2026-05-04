# Phase VIII: Autonomous Governance & Meta-Control

## Overview

Phase VIII extends QuASIM with autonomous governance capabilities through three core components:
- Meta-Controller Kernel (MCK)
- Policy Reasoner (PR)
- Quantum Ethical Governor (QEG)

## Components

### 1. Meta-Controller Kernel (MCK)

**Purpose**: Reinforcement learning-based autonomous parameter tuning

**Features**:
- Q-learning for Φ_QEVF variance minimization
- Deterministic seed replay for auditability
- Checkpoint save/load for reproducibility
- Telemetry integration

**Usage**:
```python
from quasim.meta import MetaControllerKernel

mck = MetaControllerKernel(seed=42)

# Observe system state
state = mck.observe_state(
    phi_variance=0.20,
    compliance_score=98.0,
    resource_utilization=0.70,
    error_rate=0.03
)

# Select action
action = mck.select_action(state)

# Update based on reward
next_state = mck.observe_state(...)
reward = mck.compute_reward(state, next_state)
mck.update_q_value(state, action, reward, next_state)
```

### 2. Policy Reasoner (PR)

**Purpose**: Logic-based compliance rule engine

**Frameworks Supported**:
- DO-178C Level A (3 rules)
- NIST 800-53 (3 rules)
- CMMC 2.0 Level 2 (2 rules)
- ISO 27001 (2 rules)

**Usage**:
```python
from quasim.policy import PolicyReasoner, ConfigurationMutation

pr = PolicyReasoner()

mutation = ConfigurationMutation(
    parameter="phi_tolerance",
    current_value=0.01,
    proposed_value=0.05,
    rationale="Performance tuning",
    requestor="mck"
)

evaluation = pr.evaluate_mutation(mutation)
print(f"Decision: {evaluation.decision}")
print(f"Required approvers: {evaluation.approved_by}")
```

### 3. Quantum Ethical Governor (QEG)

**Purpose**: Energy-equity balance monitoring

**Features**:
- Resource usage tracking
- Fairness metrics (Gini coefficient)
- Ethical scoring (0-100)
- DVL ledger emission

**Usage**:
```python
from quasim.meta import QuantumEthicalGovernor

qeg = QuantumEthicalGovernor()

# Monitor resources
metrics = qeg.monitor_resources(
    energy_consumption=50.0,
    compute_time=120.0,
    memory_usage=8.0,
    network_bandwidth=100.0
)

# Assess fairness
fairness = qeg.assess_fairness(
    resource_distribution=[100, 100, 100],
    access_counts=[10, 10, 10],
    priority_levels=[1, 1, 1]
)

# Compute ethics score
assessment = qeg.compute_ethical_score()
dvl_record = qeg.emit_to_dvl(assessment)
```

## Integration Example

```python
# Complete Phase VIII workflow
from quasim.meta import MetaControllerKernel, QuantumEthicalGovernor
from quasim.policy import PolicyReasoner, ConfigurationMutation

# Initialize components
mck = MetaControllerKernel(seed=42)
pr = PolicyReasoner()
qeg = QuantumEthicalGovernor()

# MCK proposes change
state = mck.observe_state(...)
action = mck.select_action(state)

# PR evaluates change
mutation = ConfigurationMutation(
    parameter=action.parameter_name,
    current_value=0.0,
    proposed_value=action.adjustment,
    rationale="MCK optimization",
    requestor="mck"
)
evaluation = pr.evaluate_mutation(mutation)

if evaluation.decision == PolicyDecision.APPROVED:
    # Apply change and monitor with QEG
    resource_metrics = qeg.monitor_resources(...)
    fairness_metrics = qeg.assess_fairness(...)
    assessment = qeg.compute_ethical_score()
    qeg.emit_to_dvl(assessment)
```

## Testing

```bash
# Run all Phase VIII tests
make autonomy-test

# Or with pytest
pytest tests/phaseVIII/ -v
```

## CI/CD

Phase VIII CI runs:
- On push to main/develop
- Nightly simulation at 2 AM UTC
- Integration tests

See `.github/workflows/phaseVIII.yml`

## Metrics

Phase VIII exposes metrics:
- `phi_variance` - Φ_QEVF variance
- `policy_decision_rate` - PR decision throughput
- `ethics_score` - QEG ethical compliance

## References

- MCK: Reinforcement Learning for Parameter Optimization
- PR: Rule-Based Compliance Validation
- QEG: Energy-Equity Balance Monitoring
