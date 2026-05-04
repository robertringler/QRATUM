# Agent-Scenario Bridge

The agent bridge connects scenario state, grounded observations, and agent actions.

- **Observations**: `observation_from_state` merges scenario context with grounded worldmodel state into an `AgentObservation`.
- **Actions**: agent decisions become `InterventionAction` instances applied deterministically to the scenario state.
- **Integration**: used by the multi-agent planner to align QScenario ticks, QReal grounding, and QIntervention rollout.

This bridge ensures agents operate on consistent views and that their actions are traceable across replay.
