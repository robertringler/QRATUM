# External Adapters

Adapters live under `qreal` and provide deterministic ingestion for external feeds. Each adapter:
- Validates required fields
- Applies normalization chains (field renames, clamping, key sorting)
- Emits percept-friendly payloads
- Attaches provenance for replay and attestation

## Adapter Types
- **MarketAdapter**: wraps OHLCV-style bars.
- **TelemetryAdapter**: ingests vehicle state vectors.
- **GridAdapter**: captures load, generation, and frequency measurements.
- **TransportAdapter**: handles aviation/rail/road positional and status updates.
- **ScienceAdapter**: records scientific measurements with units.

Adapters can be extended by adding new normalization steps or validators without breaking determinism.
