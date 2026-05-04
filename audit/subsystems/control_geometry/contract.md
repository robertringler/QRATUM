# Subsystem contract: `control_geometry`

_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:29:11Z

**Path**: `qagents/control_geometry/`

## Files

- sensitivity.py
- embedding.py
- reachability.py
- controllability_rank.py
- control_directions.py
- policy_optimizer.py

## Public types / API surface

- DisplacementCache, action_to_displacement, action_similarity (cosine)
- ReachabilityGraph (constraint-gated by default)
- controllability_rank (SVD)
- classify_action_space → productive/neutral/degenerate/unsafe

## Invariants

- No global RNG — all functions deterministic
- select_action / control_policy invoke validate_action before returning
