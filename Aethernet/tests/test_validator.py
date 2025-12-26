"""Tests for Aethernet Validator Module."""

import pytest
from Aethernet.core.validator import (
    Validator,
    ValidatorCredentials,
    ValidatorRegistry,
    ValidatorStake,
    ValidatorStatus,
    SlashingReason,
)


class TestValidatorCredentials:
    """Tests for ValidatorCredentials."""

    def test_credential_creation(self):
        """Test creating credentials."""
        creds = ValidatorCredentials(
            validator_id="val_001",
            public_key="abc123",
            node_address="127.0.0.1:26656",
            created_at="2025-01-01T00:00:00Z",
        )
        assert creds.validator_id == "val_001"
        assert creds.public_key == "abc123"

    def test_credential_hash(self):
        """Test credential hash is deterministic."""
        creds1 = ValidatorCredentials(
            validator_id="val_001",
            public_key="abc123",
            node_address="127.0.0.1:26656",
            created_at="2025-01-01T00:00:00Z",
        )
        creds2 = ValidatorCredentials(
            validator_id="val_001",
            public_key="abc123",
            node_address="127.0.0.1:26656",
            created_at="2025-01-01T00:00:00Z",
        )
        assert creds1.compute_hash() == creds2.compute_hash()

    def test_credential_serialization(self):
        """Test credential serialization."""
        creds = ValidatorCredentials(
            validator_id="val_001",
            public_key="abc123",
            node_address="127.0.0.1:26656",
            created_at="2025-01-01T00:00:00Z",
        )
        serialized = creds.serialize()
        assert "credential_hash" in serialized
        assert serialized["validator_id"] == "val_001"


class TestValidatorStake:
    """Tests for ValidatorStake."""

    def test_stake_creation(self):
        """Test creating stake."""
        stake = ValidatorStake(validator_id="val_001")
        assert stake.total_stake == 0
        assert stake.effective_stake() == 0

    def test_add_self_stake(self):
        """Test adding self stake."""
        stake = ValidatorStake(validator_id="val_001")
        stake.add_stake(10000, is_self=True)
        assert stake.total_stake == 10000
        assert stake.self_stake == 10000
        assert stake.delegated_stake == 0

    def test_add_delegated_stake(self):
        """Test adding delegated stake."""
        stake = ValidatorStake(validator_id="val_001")
        stake.add_stake(10000, is_self=True)
        stake.add_stake(5000, is_self=False)
        assert stake.total_stake == 15000
        assert stake.self_stake == 10000
        assert stake.delegated_stake == 5000

    def test_unbonding(self):
        """Test unbonding stake."""
        stake = ValidatorStake(validator_id="val_001")
        stake.add_stake(10000, is_self=True)
        stake.begin_unbonding(3000, "epoch_21")
        assert stake.unbonding_stake == 3000
        assert stake.effective_stake() == 7000

    def test_complete_unbonding(self):
        """Test completing unbonding."""
        stake = ValidatorStake(validator_id="val_001")
        stake.add_stake(10000, is_self=True)
        stake.begin_unbonding(3000, "epoch_21")
        unbonded = stake.complete_unbonding()
        assert unbonded == 3000
        assert stake.total_stake == 7000
        assert stake.unbonding_stake == 0

    def test_slash_stake(self):
        """Test slashing stake."""
        stake = ValidatorStake(validator_id="val_001")
        stake.add_stake(10000, is_self=True)
        slashed = stake.slash(0.10)  # 10% slash
        assert slashed == 1000
        assert stake.total_stake == 9000


class TestValidatorRegistry:
    """Tests for ValidatorRegistry."""

    def test_registry_creation(self):
        """Test creating registry."""
        registry = ValidatorRegistry()
        assert registry.current_epoch == 0
        assert len(registry.validators) == 0

    def test_register_validator(self):
        """Test registering a validator."""
        registry = ValidatorRegistry()
        validator = registry.register_validator(
            public_key="pk_abc123",
            node_address="127.0.0.1:26656",
            initial_stake=10000,
        )
        assert validator.status == ValidatorStatus.PENDING
        assert validator.stake.total_stake == 10000

    def test_register_validator_min_stake(self):
        """Test registration fails below minimum stake."""
        registry = ValidatorRegistry()
        with pytest.raises(ValueError):
            registry.register_validator(
                public_key="pk_abc123",
                node_address="127.0.0.1:26656",
                initial_stake=100,  # Below minimum
            )

    def test_activate_validator(self):
        """Test activating a validator."""
        registry = ValidatorRegistry()
        validator = registry.register_validator(
            public_key="pk_abc123",
            node_address="127.0.0.1:26656",
            initial_stake=10000,
        )
        result = registry.activate_validator(validator.credentials.validator_id)
        assert result is True
        assert validator.status == ValidatorStatus.ACTIVE

    def test_delegate_stake(self):
        """Test delegating stake."""
        registry = ValidatorRegistry()
        validator = registry.register_validator(
            public_key="pk_abc123",
            node_address="127.0.0.1:26656",
            initial_stake=10000,
        )
        registry.activate_validator(validator.credentials.validator_id)
        result = registry.delegate_stake(
            validator.credentials.validator_id,
            amount=5000,
            delegator_id="del_001",
        )
        assert result is True
        assert validator.stake.delegated_stake == 5000

    def test_slash_validator(self):
        """Test slashing a validator."""
        registry = ValidatorRegistry()
        validator = registry.register_validator(
            public_key="pk_abc123",
            node_address="127.0.0.1:26656",
            initial_stake=10000,
        )
        registry.activate_validator(validator.credentials.validator_id)
        event = registry.slash_validator(
            validator.credentials.validator_id,
            SlashingReason.DOWNTIME,
            "evidence_hash_123",
        )
        assert event is not None
        assert event.reason == SlashingReason.DOWNTIME
        assert len(registry.slashing_history) == 1

    def test_jail_validator(self):
        """Test jailing a validator."""
        registry = ValidatorRegistry()
        validator = registry.register_validator(
            public_key="pk_abc123",
            node_address="127.0.0.1:26656",
            initial_stake=10000,
        )
        registry.activate_validator(validator.credentials.validator_id)
        result = registry.jail_validator(
            validator.credentials.validator_id,
            "test_reason",
        )
        assert result is True
        assert validator.status == ValidatorStatus.JAILED

    def test_unjail_validator(self):
        """Test unjailing a validator."""
        registry = ValidatorRegistry()
        validator = registry.register_validator(
            public_key="pk_abc123",
            node_address="127.0.0.1:26656",
            initial_stake=10000,
        )
        registry.activate_validator(validator.credentials.validator_id)
        registry.jail_validator(validator.credentials.validator_id, "test")
        result = registry.unjail_validator(validator.credentials.validator_id)
        assert result is True
        assert validator.status == ValidatorStatus.ACTIVE

    def test_get_active_validators(self):
        """Test getting active validators."""
        registry = ValidatorRegistry()
        v1 = registry.register_validator("pk1", "addr1", 10000)
        v2 = registry.register_validator("pk2", "addr2", 10000)
        registry.activate_validator(v1.credentials.validator_id)
        # v2 stays pending
        active = registry.get_active_validators()
        assert len(active) == 1
        assert active[0].credentials.validator_id == v1.credentials.validator_id

    def test_total_voting_power(self):
        """Test total voting power calculation."""
        registry = ValidatorRegistry()
        v1 = registry.register_validator("pk1", "addr1", 10000)
        v2 = registry.register_validator("pk2", "addr2", 20000)
        registry.activate_validator(v1.credentials.validator_id)
        registry.activate_validator(v2.credentials.validator_id)
        total = registry.get_total_voting_power()
        assert total == 30000

    def test_rotation_schedule(self):
        """Test proposer rotation schedule."""
        registry = ValidatorRegistry()
        v1 = registry.register_validator("pk1", "addr1", 10000)
        v2 = registry.register_validator("pk2", "addr2", 20000)
        registry.activate_validator(v1.credentials.validator_id)
        registry.activate_validator(v2.credentials.validator_id)
        schedule = registry.compute_rotation_schedule()
        # Higher stake should be first
        assert len(schedule) == 2
        assert schedule[0] == v2.credentials.validator_id

    def test_proposer_for_slot(self):
        """Test getting proposer for slot."""
        registry = ValidatorRegistry()
        v1 = registry.register_validator("pk1", "addr1", 10000)
        registry.activate_validator(v1.credentials.validator_id)
        proposer = registry.get_proposer_for_slot(0)
        assert proposer == v1.credentials.validator_id

    def test_advance_epoch(self):
        """Test advancing epoch."""
        registry = ValidatorRegistry()
        registry.advance_epoch()
        assert registry.current_epoch == 1

    def test_registry_stats(self):
        """Test registry statistics."""
        registry = ValidatorRegistry()
        v1 = registry.register_validator("pk1", "addr1", 10000)
        registry.activate_validator(v1.credentials.validator_id)
        stats = registry.get_stats()
        assert stats["total_validators"] == 1
        assert stats["active_validators"] == 1


class TestValidator:
    """Tests for Validator class."""

    def test_validator_creation(self):
        """Test creating a validator."""
        creds = ValidatorCredentials(
            validator_id="val_001",
            public_key="pk",
            node_address="addr",
            created_at="2025-01-01T00:00:00Z",
        )
        validator = Validator(credentials=creds)
        assert validator.status == ValidatorStatus.PENDING
        assert validator.commission_rate == 0.10

    def test_validator_is_active(self):
        """Test is_active check."""
        creds = ValidatorCredentials(
            validator_id="val_001",
            public_key="pk",
            node_address="addr",
            created_at="2025-01-01T00:00:00Z",
        )
        validator = Validator(credentials=creds)
        assert validator.is_active() is False
        validator.status = ValidatorStatus.ACTIVE
        assert validator.is_active() is True

    def test_validator_voting_power(self):
        """Test voting power calculation."""
        creds = ValidatorCredentials(
            validator_id="val_001",
            public_key="pk",
            node_address="addr",
            created_at="2025-01-01T00:00:00Z",
        )
        validator = Validator(credentials=creds)
        validator.stake.add_stake(10000, is_self=True)
        # Not active - no voting power
        assert validator.voting_power() == 0
        validator.status = ValidatorStatus.ACTIVE
        assert validator.voting_power() == 10000

    def test_validator_serialization(self):
        """Test validator serialization."""
        creds = ValidatorCredentials(
            validator_id="val_001",
            public_key="pk",
            node_address="addr",
            created_at="2025-01-01T00:00:00Z",
        )
        validator = Validator(credentials=creds)
        serialized = validator.serialize()
        assert "credentials" in serialized
        assert "stake" in serialized
        assert serialized["status"] == "pending"
