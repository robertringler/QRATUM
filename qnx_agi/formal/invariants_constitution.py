"""System-wide constitutional invariants."""
from __future__ import annotations



def invariants() -> list[str]:
    return [
        "ledger_append_only",
        "constitution_version_monotonic",
        "node_syscalls_must_be_declared",
    ]


def check_invariants(state: dict[str, object]) -> list[str]:
    violations: list[str] = []
    if not state.get("ledger_append_only", True):
        violations.append("ledger_append_only")
    if not state.get("constitution_version_monotonic", True):
        violations.append("constitution_version_monotonic")
    if not state.get("node_syscalls_declared", True):
        violations.append("node_syscalls_must_be_declared")
    return violations
