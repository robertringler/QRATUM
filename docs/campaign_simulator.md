# Campaign Simulator

Campaigns orchestrate multiple teams of agents across scenario timelines. Red/Blue/Gray teams run their policies against a shared configuration, producing deterministic logs and scorecards.

- **Campaign**: captures scenario config, teams, and constraints.
- **Teams**: Red (adversarial), Blue (defensive), Gray (neutral/systemic) built from deterministic agents.
- **Scoring**: `CampaignScorer` computes team scores from logged actions and rewards.
- **Reporting**: `CampaignReport` summarizes results and highlights outcomes.

Campaign runs are fully replayable and integrate with QScenario outcomes and QuNimbus governance signals.
