import pytest

from qconstitution.validator import ValidationError, validate_node_config
from qnode.config import NodeConfig


def test_validation_rejects_missing_syscalls():
    cfg = NodeConfig(node_id="n1", identity_ref="i", allowed_syscalls=[])
    with pytest.raises(ValidationError):
        validate_node_config(cfg)


def test_validation_accepts_declared_syscalls():
    cfg = NodeConfig(node_id="n1", identity_ref="i", allowed_syscalls=["read"])
    validate_node_config(cfg)
