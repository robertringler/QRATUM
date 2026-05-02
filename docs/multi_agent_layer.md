# Multi-Agent Layer

The multi-agent layer introduces deterministic agents (`qagents`) that consume observations, apply strategies, and emit actions suitable for Q-Stack interventions. Each agent binds a policy to an `AgentState` that records observations, actions, and rewards for replayable trajectories.

## Key components
- **Agent / AgentState**: tracks memory, actions, and rewards without randomness.
- **Strategies**: rule-based or scripted decision rules turned into policies via `PolicyAdapter`.
- **InteractionBus**: deterministic message ordering for agent-to-agent communication.
- **Rewards**: shaped reward helpers aggregate outcomes consistently.

Agents operate over QScenario timelines and can be organized into teams for campaigns.
