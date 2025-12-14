from qagents.base import Agent, LambdaPolicy
from qcampaign.campaign import Campaign
from qcampaign.teams import RedTeam, BlueTeam


def test_scoring_rewards_low_incidents():
    policy = LambdaPolicy(lambda obs, state: {"action": "noop"})
    red = RedTeam(name="red", agents=[Agent("r1", policy)])
    blue = BlueTeam(name="blue", agents=[Agent("b1", policy)])
    campaign = Campaign(name="cmp", scenario_config={}, teams=[red, blue], timeline_ticks=1)

    scorecard = campaign.run()

    assert scorecard.team_scores["red"] == scorecard.team_scores["blue"]
    assert all(score >= 9.0 for score in scorecard.team_scores.values())
