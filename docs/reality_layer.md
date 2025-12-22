# Reality Layer

The reality layer brings external observations into Q-Stack deterministically. Adapters ingest raw feed data, normalize it, and attach provenance so that every transformation is reproducible. No wall-clock timers or randomness are used; logical ticks drive ingestion order.

## Components

- **Adapters** convert market, telemetry, grid, transport, and scientific data into normalized structures.
- **Sandbox** enforces tick-based rate limiting, caching, and whitelist/blacklist filters.
- **Provenance** hashes normalized payloads alongside the logical tick and source identity.
- **Perception Bridge** turns normalized payloads into AGI percepts for downstream world model updates.
- **Grounding** reconciles simulated state with observed state using deterministic confidence weights.

## Replayability

All adapter inputs and normalized outputs are ordered by tick and can be replayed through the cluster replay mechanisms to reproduce outcomes exactly.
