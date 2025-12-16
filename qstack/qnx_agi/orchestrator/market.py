"""Internal voting market for subagents."""

from __future__ import annotations

from typing import Dict, List

from .critic import critic_score


def aggregate(plans: Dict[str, List[Dict[str, object]]]) -> Dict[str, object]:
    scored = {name: critic_score(plan) for name, plan in plans.items()}
    winner = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return {"winner": winner[0], "score": winner[1], "scores": scored}
