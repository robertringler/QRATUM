"""Internal voting market for subagents."""
from __future__ import annotations


from .critic import critic_score


def aggregate(plans: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    scored = {name: critic_score(plan) for name, plan in plans.items()}
    winner = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return {"winner": winner[0], "score": winner[1], "scores": scored}
