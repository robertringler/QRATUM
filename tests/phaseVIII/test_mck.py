"""Tests for Meta-Controller Kernel (MCK)."""

from quasim.meta import MCKAction, MCKState, MetaControllerKernel


def test_mck_initialization():
    """Test MCK initialization with seed."""

    mck = MetaControllerKernel(seed=42)
    assert mck.seed == 42
    assert mck.learning_rate == 0.01
    assert mck.discount_factor == 0.95
    assert mck.exploration_rate == 0.1


def test_mck_state_observation():
    """Test state observation."""

    mck = MetaControllerKernel(seed=42)

    state = mck.observe_state(
        phi_variance=0.15, compliance_score=98.5, resource_utilization=0.65, error_rate=0.02
    )

    assert isinstance(state, MCKState)
    assert state.phi_variance == 0.15
    assert state.compliance_score == 98.5
    assert state.resource_utilization == 0.65
    assert state.error_rate == 0.02
    assert state.timestamp is not None


def test_mck_action_selection():
    """Test action selection with deterministic seed."""

    mck = MetaControllerKernel(seed=42)

    state = mck.observe_state(
        phi_variance=0.15, compliance_score=98.5, resource_utilization=0.65, error_rate=0.02
    )

    action = mck.select_action(state)

    assert isinstance(action, MCKAction)
    assert action.parameter_name is not None
    assert action.adjustment is not None
    assert action.reason in ["exploration", "exploitation"]


def test_mck_reward_computation():
    """Test reward computation."""

    mck = MetaControllerKernel(seed=42)

    prev_state = MCKState(
        phi_variance=0.20,
        compliance_score=98.0,
        resource_utilization=0.70,
        error_rate=0.03,
        timestamp="2025-01-01T00:00:00Z",
    )

    new_state = MCKState(
        phi_variance=0.15,
        compliance_score=98.5,
        resource_utilization=0.65,
        error_rate=0.02,
        timestamp="2025-01-01T00:01:00Z",
    )

    reward = mck.compute_reward(prev_state, new_state)

    # Reward should be positive due to variance reduction
    assert reward > 0


def test_mck_q_value_update():
    """Test Q-value update."""

    mck = MetaControllerKernel(seed=42)

    state = MCKState(
        phi_variance=0.20,
        compliance_score=98.0,
        resource_utilization=0.70,
        error_rate=0.03,
        timestamp="2025-01-01T00:00:00Z",
    )

    next_state = MCKState(
        phi_variance=0.15,
        compliance_score=98.5,
        resource_utilization=0.65,
        error_rate=0.02,
        timestamp="2025-01-01T00:01:00Z",
    )

    action = MCKAction(parameter_name="phi_tolerance", adjustment=0.05, reason="test")

    mck.update_q_value(state, action, reward=1.5, next_state=next_state)

    assert len(mck.experience_buffer) == 1
    assert len(mck.q_table) > 0


def test_mck_checkpoint_save_load(tmp_path):
    """Test checkpoint save and load for deterministic replay."""

    mck = MetaControllerKernel(seed=42)

    # Train for a few episodes
    for i in range(5):
        state = mck.observe_state(
            phi_variance=0.20 - i * 0.01,
            compliance_score=98.0 + i * 0.1,
            resource_utilization=0.70,
            error_rate=0.03,
        )
        action = mck.select_action(state)
        next_state = mck.observe_state(
            phi_variance=0.19 - i * 0.01,
            compliance_score=98.1 + i * 0.1,
            resource_utilization=0.68,
            error_rate=0.025,
        )
        reward = mck.compute_reward(state, next_state)
        mck.update_q_value(state, action, reward, next_state)

    # Save checkpoint
    checkpoint_path = tmp_path / "mck_checkpoint.json"
    mck.save_checkpoint(checkpoint_path)

    assert checkpoint_path.exists()

    # Load checkpoint
    mck2 = MetaControllerKernel()
    mck2.load_checkpoint(checkpoint_path)

    assert mck2.seed == mck.seed
    assert len(mck2.experience_buffer) == len(mck.experience_buffer)
    assert len(mck2.q_table) == len(mck.q_table)


def test_mck_convergence():
    """Test MCK convergence over multiple episodes."""

    mck = MetaControllerKernel(seed=42)

    initial_variance = 0.30
    target_variance = 0.10

    for episode in range(20):
        current_variance = initial_variance - (initial_variance - target_variance) * (episode / 20)

        state = mck.observe_state(
            phi_variance=current_variance,
            compliance_score=98.0,
            resource_utilization=0.65,
            error_rate=0.02,
        )

        action = mck.select_action(state, epsilon=0.1)

        # Simulate action effect
        next_variance = max(target_variance, current_variance - 0.01)
        next_state = mck.observe_state(
            phi_variance=next_variance,
            compliance_score=98.0,
            resource_utilization=0.65,
            error_rate=0.02,
        )

        reward = mck.compute_reward(state, next_state)
        mck.update_q_value(state, action, reward, next_state)

    # Check metrics
    metrics = mck.get_performance_metrics()
    assert metrics["episodes"] == 20
    assert metrics["phi_variance_reduction"] > 0
    assert metrics["compliance_maintained"] is True


def test_mck_performance_metrics():
    """Test performance metrics retrieval."""

    mck = MetaControllerKernel(seed=42)

    # Initial metrics
    metrics = mck.get_performance_metrics()
    assert metrics["episodes"] == 0
    assert metrics["avg_reward"] == 0.0

    # After some episodes
    state = mck.observe_state(
        phi_variance=0.20, compliance_score=98.0, resource_utilization=0.70, error_rate=0.03
    )
    action = mck.select_action(state)
    next_state = mck.observe_state(
        phi_variance=0.15, compliance_score=98.5, resource_utilization=0.65, error_rate=0.02
    )
    reward = mck.compute_reward(state, next_state)
    mck.update_q_value(state, action, reward, next_state)

    metrics = mck.get_performance_metrics()
    assert metrics["episodes"] == 1
    assert metrics["avg_reward"] > 0
