"""Node-level policies for safety and alignment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str


@dataclass
class NodePolicy:
    name: str
    check: Callable[[dict[str, object]], bool]
    reason: str

    def evaluate(self, context: dict[str, object]) -> PolicyDecision:
        allowed = bool(self.check(context))
        detail = self.reason if allowed else f"blocked:{self.reason}"
        return PolicyDecision(allowed=allowed, reason=detail)


def syscall_allowlist_policy(allowed: list[str]) -> NodePolicy:
    def _check(ctx: dict[str, object]) -> bool:
        return ctx.get("syscall") in allowed

    return NodePolicy(name="allowlist", check=_check, reason="syscall-allowlist")


def budget_policy(limit_lookup: Callable[[str], int]) -> NodePolicy:
    def _check(ctx: dict[str, object]) -> bool:
        budget = limit_lookup(ctx.get("syscall", ""))
        cost = int(ctx.get("cost", 0))
        return cost <= budget

    return NodePolicy(name="budget", check=_check, reason="budget-bound")


def compose_policies(policies: list[NodePolicy], context: dict[str, object]) -> PolicyDecision:
    for policy in policies:
        decision = policy.evaluate(context)
        if not decision.allowed:
            return decision
    return PolicyDecision(True, "ok")
