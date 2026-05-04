from qagents.base import Agent, AgentObservation, LambdaPolicy


def test_agent_act_records_state():
    policy = LambdaPolicy(lambda obs, state: {"action": "noop", "tick": obs.tick})
    agent = Agent("a1", policy)
    obs = AgentObservation(tick=1, view={"x": 2}, provenance="test")

    action = agent.act(obs)

    assert action["tick"] == 1
    assert agent.state.observations[0].view["x"] == 2
    assert agent.state.actions[0]["action"] == "noop"
