# Provenance and Grounding

## Provenance
`qreal.provenance` computes SHA-256 digests of normalized payloads combined with the logical tick and source identifier. These records provide deterministic fingerprints for attestation and replay.

## Grounding
`qnx_agi.worldmodel.grounding` merges simulated and observed states using normalized confidence weights. Anchors record the source and tick for every observed field, enabling traceable reconciliation between world models and real-world measurements.
