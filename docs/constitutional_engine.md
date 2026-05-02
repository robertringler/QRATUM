# Q-Stack Constitutional Engine

The constitutional engine captures the core safety, alignment, and governance rules that every node, cluster, and scenario must satisfy. A charter version contains an article set with deterministic constraints. Validation is executed through the interpreter to ensure configurations declare allowed syscalls and route violations to the ledger.

## Components
- **Articles**: atomic constraints with scopes and metadata.
- **Charter**: active constitutional version with enactment tick.
- **Interpreter**: evaluates a subject context against scoped articles.
- **Validator**: reusable helpers to enforce the constitution on node configs and policies.
- **Upgrade Path**: deterministic rules to move between constitutional versions with quorum thresholds.

## Usage
1. Load the active charter (`qconstitution.charter.default_charter`).
2. Run `validate_node_config` before node startup to enforce safety envelopes.
3. Record violations and upgrades into `qledger` for replayable audit.
