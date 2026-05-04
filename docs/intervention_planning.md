# Intervention Planning

The intervention layer (`qintervention`) provides deterministic actions, constraints, planning, rollout, and evaluation. Actions are scheduled by logical tick and never rely on wall-clock time.

- **Actions & Scheduling**: `InterventionAction` and `ScheduledAction` capture intent with ordered keys.
- **Constraints**: `ConstraintSet` enforces safety/whitelist checks before actions join a plan.
- **Planner**: `InterventionPlanner` merges agent proposals into a consistent `InterventionPlan`.
- **Rollout**: `rollout_plan` applies actions against scenario state with reproducible metrics.
- **Evaluation**: `InterventionEvaluation` scores compliance and impact using deterministic weights.

Interventions connect agents, scenarios, QuNimbus governance, and QuASIM dynamics in a reproducible pipeline.
