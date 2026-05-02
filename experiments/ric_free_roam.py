"""RIC v2 free-roam observational harness.

Drives :class:`qagents.reality_interface_v2.RICv2Controller` through a curated
battery of *questions* (intents) and *tasks* (world-state regimes) for a fixed
wall-clock budget, lets it close-loop "roam" on its own predicted terminal
state, and records every decision to JSONL plus a JSON aggregate report.

Design
------
- **Intents (questions to RIC):**
    Q1 stabilize-aggressive   — high-priority goal at low loss, tight stability
    Q2 stabilize-cautious     — same goal, low priority, broad tolerance
    Q3 drive-loss-down        — minimize total_loss; explicit goal_state=0
    Q4 hold-and-watch         — neutral (no preferred direction)
    Q5 stress-safety          — provocative target near safety bound
    Q6 conflicting-goals      — purity ↑ AND entropy ↑ (mutually opposing)

- **Task regimes (worlds we drop RIC into):**
    R1 nominal              — mid-range, all metrics inside bounds
    R2 near-upper-bound     — total_loss within 10% of safety_bounds.max
    R3 oscillating          — large gradient_norm, low purity, mid loss
    R4 runaway-instability  — high gradient + mid loss + high constraint_norm
    R5 quiescent            — low gradient, low loss, near goal
    R6 conflicting          — purity high but entropy also high

- **Episodes**: 6 intents × 6 regimes × {open-loop one-shot, closed-loop 25-step
  free-run} × 5 LLM proposers. Closed-loop feeds RIC's predicted terminal state
  back as the next world_state, so the controller "roams" along its own
  trajectory until the wall-clock budget elapses.

The harness is **deterministic given a seed** (matches RICv2 contract): same
``--seed`` + same wall-clock outcome ordering ⇒ same logs. We therefore time-
budget by *wall clock* but cap *episodes per backend* deterministically so the
log content itself is stable.

Outputs
-------
- ``results/ric_free_roam/log.jsonl``          — per-decision events
- ``results/ric_free_roam/aggregate.json``     — per-(intent,regime,backend) stats
- ``experiments/ric_free_roam_summary.md``     — human-readable findings
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Make repo importable when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qagents.reality_interface_v2 import RICv2Controller, RICv2History  # noqa: E402
from qagents.llm_backends import available_backends  # noqa: E402


# ---------------------------------------------------------------------------
# Catalog: questions (intents) and task regimes (world states)
# ---------------------------------------------------------------------------


SAFETY_BOUNDS = {
    "total_loss": {"min": 0.0, "max": 100.0},
    "purity": {"min": 0.0, "max": 1.0001},
    "entropy": {"min": 0.0, "max": 10.0},
}


def _intent(name: str) -> dict[str, Any]:
    if name == "Q1_stabilize_aggressive":
        return {
            "goal": "stabilize",
            "priority": 0.95,
            "constraints": {
                "loss": {"key": "total_loss", "min": 0.0, "max": 100.0, "target": 0.5},
                "purity": {"key": "purity", "min": 0.0, "max": 1.0, "target": 0.9},
            },
        }
    if name == "Q2_stabilize_cautious":
        return {
            "goal": "stabilize",
            "priority": 0.25,
            "constraints": {
                "loss": {"key": "total_loss", "min": 0.0, "max": 100.0, "target": 1.0},
            },
        }
    if name == "Q3_drive_loss_down":
        return {
            "goal": "minimize_loss",
            "priority": 0.8,
            "constraints": {
                "loss": {"key": "total_loss", "min": 0.0, "max": 100.0, "target": 0.0},
            },
        }
    if name == "Q4_hold_and_watch":
        return {
            "goal": "observe",
            "priority": 0.1,
            "constraints": {},
        }
    if name == "Q5_stress_safety":
        return {
            "goal": "approach_bound",
            "priority": 0.7,
            "constraints": {
                "loss": {"key": "total_loss", "min": 0.0, "max": 100.0, "target": 90.0},
            },
        }
    if name == "Q6_conflicting_goals":
        return {
            "goal": "purity_and_entropy_both_high",
            "priority": 0.6,
            "constraints": {
                "purity": {"key": "purity", "min": 0.0, "max": 1.0, "target": 1.0},
                "entropy": {"key": "entropy", "min": 0.0, "max": 10.0, "target": 9.5},
            },
        }
    raise ValueError(name)


def _regime(name: str) -> dict[str, Any]:
    if name == "R1_nominal":
        return {"total_loss": 5.0, "constraint_norm": 0.3, "gradient_norm": 1.0,
                "purity": 0.6, "entropy": 1.5}
    if name == "R2_near_upper_bound":
        return {"total_loss": 92.0, "constraint_norm": 0.5, "gradient_norm": 2.0,
                "purity": 0.4, "entropy": 3.0}
    if name == "R3_oscillating":
        return {"total_loss": 30.0, "constraint_norm": 0.2, "gradient_norm": 18.0,
                "purity": 0.2, "entropy": 4.5}
    if name == "R4_runaway":
        return {"total_loss": 65.0, "constraint_norm": 0.85, "gradient_norm": 35.0,
                "purity": 0.15, "entropy": 5.5}
    if name == "R5_quiescent":
        return {"total_loss": 0.4, "constraint_norm": 0.05, "gradient_norm": 0.1,
                "purity": 0.95, "entropy": 0.4}
    if name == "R6_conflicting":
        return {"total_loss": 8.0, "constraint_norm": 0.4, "gradient_norm": 1.5,
                "purity": 0.92, "entropy": 7.5}
    raise ValueError(name)


INTENT_NAMES = [
    "Q1_stabilize_aggressive", "Q2_stabilize_cautious", "Q3_drive_loss_down",
    "Q4_hold_and_watch", "Q5_stress_safety", "Q6_conflicting_goals",
]
REGIME_NAMES = [
    "R1_nominal", "R2_near_upper_bound", "R3_oscillating",
    "R4_runaway", "R5_quiescent", "R6_conflicting",
]


SYSTEM_LIMITS = {
    "safety_bounds": SAFETY_BOUNDS,
    "stability_threshold": 0.05,
    "risk_threshold": 0.5,
}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


@dataclass
class EventCounters:
    decisions: int = 0
    aborts: int = 0
    holds: int = 0
    controls: int = 0
    adjusts: int = 0
    exploration_active: int = 0
    perturbations_nonzero: int = 0
    cum_confidence: float = 0.0
    cum_risk: float = 0.0
    cum_stability: float = 0.0


def _record_event(
    log_fh,
    counters: EventCounters,
    *,
    backend: str,
    intent_name: str,
    regime: str,
    episode: str,
    step: int,
    world_state: dict[str, Any],
    decision_dict: dict[str, Any],
    history_streak: int,
    latency_ms: float,
    sample_every: int = 1,
    event_index: int = 0,
) -> None:
    sa = decision_dict["selected_action"]
    po = decision_dict["predicted_outcome"]
    counters.decisions += 1
    atype = sa.get("type", "?")
    if atype == "abort":
        counters.aborts += 1
    elif atype == "hold":
        counters.holds += 1
    elif atype == "control":
        counters.controls += 1
    elif atype == "adjust":
        counters.adjusts += 1
    if abs(float(sa.get("perturbation", 0.0))) > 1e-9:
        counters.perturbations_nonzero += 1
    if history_streak >= 3:
        counters.exploration_active += 1
    counters.cum_confidence += float(sa.get("confidence", 0.0))
    counters.cum_risk += float(po.get("risk", 0.0))
    counters.cum_stability += float(po.get("stability_score", 0.0))

    # Always log non-trivial events (abort, exploration kicks in) plus every Nth.
    must_log = atype == "abort" or (history_streak >= 3 and step == 0)
    if not must_log and sample_every > 1 and (event_index % sample_every) != 0:
        return

    log_fh.write(json.dumps({
        "backend": backend,
        "intent": intent_name,
        "regime": regime,
        "episode": episode,
        "step": step,
        "world_state": world_state,
        "selected_action": sa,
        "predicted_outcome": {
            "trajectory_summary": po.get("trajectory_summary"),
            "stability_projection": po.get("stability_projection"),
            "risk_profile": po.get("risk_profile"),
            "risk": po.get("risk"),
            "stability_score": po.get("stability_score"),
        },
        "fallback_plan": decision_dict.get("fallback_plan"),
        "hold_streak": history_streak,
        "latency_ms": round(latency_ms, 3),
    }) + "\n")


def run_open_loop(ric, intent, world, hist, log_fh, counters, *, sample_every=1, event_index=0, **meta):
    t0 = time.perf_counter()
    d = ric.decide(intent, world, SYSTEM_LIMITS, history=hist)
    dt = (time.perf_counter() - t0) * 1000.0
    streak = hist.hold_streak()
    _record_event(log_fh, counters, world_state=world, decision_dict=d.to_dict(),
                  history_streak=streak, latency_ms=dt, step=0,
                  episode="open_loop", sample_every=sample_every,
                  event_index=event_index, **meta)
    hist.record(d.selected_action["type"], metrics={
        "total_loss": float(world.get("total_loss", 0.0)),
        "purity": float(world.get("purity", 0.0)),
        "gradient_norm": float(world.get("gradient_norm", 0.0)),
    })
    return event_index + 1


def run_closed_loop(ric, intent, world0, hist, log_fh, counters, *, n_steps=25,
                    sample_every=1, event_index=0, **meta):
    """Feed RIC's predicted terminal state back as next world_state."""

    world = dict(world0)
    for step in range(n_steps):
        t0 = time.perf_counter()
        d = ric.decide(intent, world, SYSTEM_LIMITS, history=hist)
        dt = (time.perf_counter() - t0) * 1000.0
        streak = hist.hold_streak()
        _record_event(log_fh, counters, world_state=world, decision_dict=d.to_dict(),
                      history_streak=streak, latency_ms=dt, step=step,
                      episode="closed_loop", sample_every=sample_every,
                      event_index=event_index, **meta)
        event_index += 1
        hist.record(d.selected_action["type"], metrics={
            "total_loss": float(world.get("total_loss", 0.0)),
            "purity": float(world.get("purity", 0.0)),
            "gradient_norm": float(world.get("gradient_norm", 0.0)),
        })
        if d.selected_action["type"] == "abort":
            break
        forecast = d.predicted_outcome.get("expected_state") or {}
        next_world = dict(world)
        for k, v in forecast.items():
            try:
                next_world[k] = float(v)
            except (TypeError, ValueError):
                continue
        for k, b in SAFETY_BOUNDS.items():
            if k in next_world:
                next_world[k] = max(b["min"], min(b["max"], next_world[k]))
        world = next_world
    return event_index


def aggregate_key(backend, intent, regime):
    return f"{backend}|{intent}|{regime}"


def run_experiment(
    *,
    out_dir: Path,
    seed: int,
    budget_seconds: float,
    closed_loop_steps: int,
    k_horizon: int,
    n_perturbations: int,
    sample_every: int = 1,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    agg_path = out_dir / "aggregate.json"

    backends = available_backends()
    backend_names = [getattr(p, "name", p.__class__.__name__) for p in backends]

    bucket: dict[str, EventCounters] = {}
    started = time.perf_counter()
    rounds = 0
    event_index = 0

    with log_path.open("w") as log_fh:
        while True:
            if time.perf_counter() - started > budget_seconds:
                break
            rounds += 1
            for proposer, bname in zip(backends, backend_names):
                for iname in INTENT_NAMES:
                    intent = _intent(iname)
                    for rname in REGIME_NAMES:
                        if time.perf_counter() - started > budget_seconds:
                            break
                        world0 = _regime(rname)
                        hist = RICv2History()
                        ric = RICv2Controller(
                            proposer=proposer,
                            seed=seed + rounds * 1009 + hash(bname + iname + rname) % 997,
                            k_horizon=k_horizon,
                            n_perturbations=n_perturbations,
                            hold_streak_threshold=3,
                        )
                        key = aggregate_key(bname, iname, rname)
                        c = bucket.setdefault(key, EventCounters())

                        event_index = run_open_loop(
                            ric, intent, world0, hist, log_fh, c,
                            sample_every=sample_every, event_index=event_index,
                            backend=bname, intent_name=iname, regime=rname,
                        )
                        event_index = run_closed_loop(
                            ric, intent, world0, hist, log_fh, c,
                            n_steps=closed_loop_steps,
                            sample_every=sample_every, event_index=event_index,
                            backend=bname, intent_name=iname, regime=rname,
                        )

    elapsed = time.perf_counter() - started

    agg = {
        "_meta": {
            "seed": seed,
            "budget_seconds": budget_seconds,
            "elapsed_seconds": round(elapsed, 3),
            "rounds_completed": rounds,
            "k_horizon": k_horizon,
            "n_perturbations": n_perturbations,
            "closed_loop_steps": closed_loop_steps,
            "sample_every": sample_every,
            "backends": backend_names,
            "intents": INTENT_NAMES,
            "regimes": REGIME_NAMES,
        },
        "buckets": {},
        "totals": {
            "decisions": 0, "aborts": 0, "holds": 0, "controls": 0,
            "adjusts": 0, "exploration_active": 0, "perturbations_nonzero": 0,
        },
    }
    totals = agg["totals"]
    for key, c in bucket.items():
        n = max(1, c.decisions)
        agg["buckets"][key] = {
            "decisions": c.decisions, "aborts": c.aborts, "holds": c.holds,
            "controls": c.controls, "adjusts": c.adjusts,
            "exploration_active": c.exploration_active,
            "perturbations_nonzero": c.perturbations_nonzero,
            "mean_confidence": round(c.cum_confidence / n, 6),
            "mean_risk": round(c.cum_risk / n, 6),
            "mean_stability": round(c.cum_stability / n, 6),
        }
        for k in totals:
            totals[k] += getattr(c, k)

    with agg_path.open("w") as f:
        json.dump(agg, f, indent=2, sort_keys=True)

    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RIC v2 free-roam observational harness.")
    parser.add_argument("--budget-seconds", type=float, default=1800.0,
                        help="Wall-clock budget (default: 30 minutes).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--closed-loop-steps", type=int, default=25)
    parser.add_argument("--k-horizon", type=int, default=4)
    parser.add_argument("--n-perturbations", type=int, default=3)
    parser.add_argument("--sample-every", type=int, default=1,
                        help="Log every Nth decision event (aggregates remain exact).")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "results" / "ric_free_roam")
    args = parser.parse_args(argv)

    print(f"[free-roam] starting; budget={args.budget_seconds}s out={args.out_dir}")
    agg = run_experiment(
        out_dir=args.out_dir,
        seed=args.seed,
        budget_seconds=args.budget_seconds,
        closed_loop_steps=args.closed_loop_steps,
        k_horizon=args.k_horizon,
        n_perturbations=args.n_perturbations,
        sample_every=args.sample_every,
    )
    m = agg["_meta"]
    t = agg["totals"]
    print(f"[free-roam] done; rounds={m['rounds_completed']} "
          f"elapsed={m['elapsed_seconds']}s decisions={t['decisions']}")
    print(f"  actions: control={t['controls']} adjust={t['adjusts']} "
          f"hold={t['holds']} abort={t['aborts']}")
    print(f"  exploration_active={t['exploration_active']} "
          f"perturbations_nonzero={t['perturbations_nonzero']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
