# Constitutional Upgrade Process

Upgrades to the Q-Stack constitution are deterministic and ledger-backed. Proposals describe new articles and target versions, are evaluated against quorum thresholds, and once accepted produce a new charter entry recorded in the ledger.

## Steps
1. Draft a `ConstitutionalUpgradeProposal` with added articles and target version id.
2. Evaluate via `UpgradePath.evaluate` using deterministic vote lists or governance scoring.
3. Apply with `UpgradePath.apply` to generate a new `Charter` version.
4. Append ledger records for proposal, acceptance, and resulting version for auditability.
