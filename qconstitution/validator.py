"""Validation utilities for Q-Stack constitutional compliance."""
from __future__ import annotations

from dataclasses import dataclass

from qconstitution.charter import Charter, default_charter
from qconstitution.interpreter import ConstitutionalInterpreter
from qnode.config import NodeConfig


class ValidationError(Exception):
    pass


@dataclass
class Validator:
    charter: Charter

    def validate_node(self, config: NodeConfig, ledger_enabled: bool = True) -> None:
        interpreter = ConstitutionalInterpreter(self.charter)
        violations = interpreter.evaluate_constraints(
            "node", {"allowed_syscalls": config.allowed_syscalls, "ledger_enabled": ledger_enabled}
        )
        if violations:
            raise ValidationError(
                f"Node {config.node_id} violates constitution: " + "; ".join(sorted(violations))
            )


def validate_node_config(config: NodeConfig, charter: Charter | None = None, ledger_enabled: bool = True) -> None:
    validator = Validator(charter or default_charter())
    validator.validate_node(config, ledger_enabled=ledger_enabled)
