from qagents.base import Agent, LambdaPolicy
from qcampaign.campaign import Campaign
from qcampaign.reporting import CampaignReport
from qcampaign.teams import RedTeam


def test_campaign_report_contains_scores():
    policy = LambdaPolicy(lambda obs, state: {"action": "noop"})
    red = RedTeam(name="red", agents=[Agent("r1", policy)])
    campaign = Campaign(name="cmp", scenario_config={}, teams=[red], timeline_ticks=1)

    scorecard = campaign.run()
    report = campaign.report(scorecard)

    assert isinstance(report, CampaignReport)
    summary = report.summary()
    assert summary["scores"]["red"] >= 0
    assert "highlights" in summary
