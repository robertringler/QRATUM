from qagents.base import AgentState
from qagents.rewards import aggregate_rewards, shaped_reward


def test_rewards_accumulate_and_aggregate():
    state1 = AgentState(agent_id="a1")
    state1.add_reward(1.0)
    state1.add_reward(2.0)
    state2 = AgentState(agent_id="a2")
    state2.add_reward(3.0)

    totals = aggregate_rewards([state1, state2])
    assert totals["a1"] == 3.0
    assert totals["a2"] == 3.0

    shaped = shaped_reward({"metric": 2.0, "other": 1.0}, {"metric": 0.5, "other": 0.5})
    assert shaped == 1.5
