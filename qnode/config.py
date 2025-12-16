"""Node configuration primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class NodeConfig:
    """Deterministic configuration for a sovereign node."""

    node_id: str
    identity_ref: str
    allowed_syscalls: List[str] = field(default_factory=list)
    policy_limits: Dict[str, int] = field(default_factory=dict)
    default_budget: int = 100
    annotations: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_dict(payload: Dict[str, object]) -> NodeConfig:
        return NodeConfig(
            node_id=str(payload.get("node_id", "qnode")),
            identity_ref=str(payload.get("identity_ref", "anonymous")),
            allowed_syscalls=list(payload.get("allowed_syscalls", [])),
            policy_limits=dict(payload.get("policy_limits", {})),
            default_budget=int(payload.get("default_budget", 100)),
            annotations={k: str(v) for k, v in payload.get("annotations", {}).items()},
        )

    def budget_for(self, syscall: str) -> int:
        return self.policy_limits.get(syscall, self.default_budget)

    def describe(self) -> Dict[str, object]:
        return {
            "node_id": self.node_id,
            "identity_ref": self.identity_ref,
            "allowed_syscalls": list(self.allowed_syscalls),
            "policy_limits": dict(self.policy_limits),
            "default_budget": self.default_budget,
            "annotations": dict(self.annotations),
        }
