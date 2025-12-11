from qagents.base import Agent, LambdaPolicy
from qcampaign.campaign import Campaign
from qcampaign.teams import BlueTeam, RedTeam


def test_campaign_runs_and_logs():
    policy = LambdaPolicy(lambda obs, state: {"action": "noop", "tick": obs.tick})
    red = RedTeam(name="red", agents=[Agent("r1", policy)])
    blue = BlueTeam(name="blue", agents=[Agent("b1", policy)])
    campaign = Campaign(name="cmp", scenario_config={"risk": 1}, teams=[red, blue], timeline_ticks=2)

    scorecard = campaign.run()

    assert len(campaign.logs) == 4  # 2 ticks * 2 teams
    assert set(scorecard.team_scores.keys()) == {"red", "blue"}
