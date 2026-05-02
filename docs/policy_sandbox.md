# Policy Sandbox

The policy sandbox runs a single Scenario across multiple PolicyVariants to compare outcomes. Core safety constraints are automatically enforced to ensure alignment.

## Workflow
1. Define base policies and construct PolicyVariant instances with overrides.
2. Pass a scenario builder into PolicySandbox.run.
3. Collect a ComparisonReport highlighting classifications and identifying the best outcome.

## Determinism
All variants execute the same Timeline with deterministic ordering. No randomness or wall-clock inputs are used.
