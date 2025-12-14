"""High-level deterministic system calls."""
from __future__ import annotations

from typing import Callable

from qnode.config import NodeConfig
from qnode.policies import NodePolicy, compose_policies
from qnode.incident_log import IncidentLog


class SyscallError(Exception):
    pass


class SyscallRouter:
    def __init__(self, config: NodeConfig, policies: list[NodePolicy], incident_log: IncidentLog) -> None:
        self.config = config
        self.policies = policies
        self.incidents = incident_log
        self.registry: dict[str, Callable[[dict[str, object]], object]] = {}

    def register(self, name: str, handler: Callable[[dict[str, object]], object]) -> None:
        self.registry[name] = handler

    def dispatch(self, name: str, payload: dict[str, object]) -> object:
        if name not in self.registry:
            raise SyscallError(f"unknown syscall {name}")
        context = {"syscall": name, "cost": payload.get("cost", 0)}
        decision = compose_policies(self.policies, context)
        if not decision.allowed:
            self.incidents.record("policy_violation", {"syscall": name, "reason": decision.reason})
            raise SyscallError(decision.reason)
        result = self.registry[name](payload)
        self.incidents.record("syscall", {"name": name, "payload": payload, "result": result})
        return result

    def registered(self) -> list[str]:
        return sorted(self.registry.keys())
