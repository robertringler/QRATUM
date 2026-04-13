# Discovery Directive Governance (QR-DISC-100-V1)

## Overview

This document defines the governance framework for the QRATUM Discovery Directive, which generates, validates, and archives 100 original discoveries across quantum computing, materials science, AI systems, cryptography, and industrial design.

## Discovery Storage

All validated discoveries are stored in the following directory structure:

```
qratum/discoveries/
├── schema/
│   └── discovery_schema.json          # JSON schema for validation
├── validated/
│   ├── QRD-001.json through QRD-100.json  # Valid discoveries (F >= 0.87)
├── pending/
│   └── (discoveries awaiting validation)
├── rejected/
│   └── (discoveries with F < 0.87)
└── index.json                         # Manifest of all discoveries
```

## Discovery Identification

- All discoveries use the format `QRD-XXX` where XXX is a 3-digit zero-padded number (001-100)
- Discovery IDs are sequential and immutable
- Each discovery has a unique QRADLE provenance hash in format `QRDL-{16 hex chars}`

## Fitness Threshold

The fitness function is defined as:

```
F = α·I_novelty + β·I_feasibility + γ·I_scalability + δ·I_strategic_leverage
```

Where:

- α = 0.30 (novelty weight)
- β = 0.25 (feasibility weight)
- γ = 0.25 (scalability weight)
- δ = 0.20 (strategic leverage weight)

**Minimum Valid Fitness: F >= 0.87**

Only discoveries meeting this threshold are placed in `validated/` directory.

## Provenance Chain

### QRADLE Integration

All discoveries are recorded in the QRADLE ledger for cryptographic provenance:

1. **Generation Timestamp**: ISO 8601 format (e.g., `2025-01-01T12:00:00Z`)
2. **QRADLE Hash**: Deterministic hash based on discovery content and seed
3. **Seed**: Deterministic seed used for generation (ensures reproducibility)
4. **Lattice Node**: Identifier of the lattice node from which discovery emerged

### Chain Verification

The provenance chain must satisfy:

1. All discoveries have valid QRADLE hashes
2. Hashes match recomputed values from discovery content
3. Discovery IDs are sequential (QRD-001, QRD-002, ..., QRD-100)
4. All fitness scores meet threshold (F >= 0.87)
5. All required fields are non-empty and valid

## Discovery Validation Rules

### Required Fields

Every discovery MUST include:

1. **ID**: Valid format `QRD-XXX`
2. **Title**: Minimum 10 characters
3. **Hypothesis**: Formal falsifiable claim
4. **Core Mechanism**: Causal physics/computation explanation
5. **Formulation**: Equations, pseudocode, or formal spec
6. **Validation**: Method, test rig, expected outcome, confidence
7. **Industrial Impact**: Application, market sector, estimated value
8. **Risk Envelope**: Failure modes, safety constraints, mitigation strategies
9. **Fitness Score**: Value in range [0.87, 1.0]
10. **Provenance**: Complete provenance record

### Forbidden Content

Discoveries MUST NOT contain:

- Speculative claims without falsifiable hypotheses
- Unsubstantiated performance claims
- Missing or invalid provenance information
- Fitness scores below 0.87 threshold
- Incomplete risk analysis

## Termination Condition

The Discovery Engine MUST continue recursive generation until:

```
count(discoveries where F >= 0.87) == 100
```

**The engine halts ONLY when exactly 100 valid discoveries are generated.**

If fewer than 100 discoveries meet the threshold, the engine continues exploring the search space with additional mutation strategies.

## Determinism Requirements

### Reproducibility

All discovery generation is deterministic:

1. **Same Seed → Same Discoveries**: Running with identical seed produces identical results
2. **Pod Isolation**: All computation occurs in isolated WASM pods
3. **No External State**: No network I/O, no system time (except for timestamps)
4. **Audit Logging**: Every operation is logged in audit trail

### Rollback Capability

The system supports rollback to any previous state:

1. Rollback points can be created at any stage
2. State can be restored to any rollback point
3. Provenance chain remains valid after rollback
4. All discoveries after rollback point are invalidated

## Access Control

### Generation Phase

- Discovery generation requires no special authorization
- Runs in isolated environment with no external dependencies
- Deterministic and auditable

### Validation Phase

- Validation requires dual-control authorization (if integrated with QRADLE governance)
- All validation results are logged
- Validation cannot modify discoveries, only accept/reject

### Publication Phase

- Moving discoveries to `validated/` requires verification of:
  - Fitness score >= 0.87
  - Valid provenance chain
  - Complete required fields
  - No forbidden content

## Compliance

### Data Privacy

- No personally identifiable information (PII) in discoveries
- No proprietary information without explicit authorization
- All data sources must be documented in provenance

### Intellectual Property

- All discoveries are governed by repository license (Apache-2.0)
- Attribution preserved through provenance chain
- No patent claims without legal review

### Safety

- All discoveries include risk envelope analysis
- Safety constraints explicitly documented
- Mitigation strategies required for high-risk discoveries

## Audit Trail

Every discovery operation is logged:

1. **Generation**: Lattice node, mutation type, fitness computation
2. **Validation**: Schema validation, provenance verification
3. **Storage**: File write operations, directory placement
4. **Retrieval**: Read operations, export operations

Audit trail is immutable and cryptographically secured via QRADLE.

## Discovery Lifecycle

```
┌─────────────┐
│  Lattice    │
│  Exploration│
└──────┬──────┘
       │
       v
┌─────────────┐
│  Mutation   │
│  Generation │
└──────┬──────┘
       │
       v
┌─────────────┐      F < 0.87
│  Fitness    ├──────────────► rejected/
│  Evaluation │
└──────┬──────┘
       │ F >= 0.87
       v
┌─────────────┐
│  Synthesis  │
│  Discovery  │
└──────┬──────┘
       │
       v
┌─────────────┐
│  Provenance │
│  Recording  │
└──────┬──────┘
       │
       v
┌─────────────┐
│  Validation │
│  (Schema)   │
└──────┬──────┘
       │
       v
    validated/
```

## Governance Updates

This governance document is versioned and subject to change:

- **Version**: 1.0.0
- **Effective Date**: 2025-01-01
- **Review Cycle**: Quarterly
- **Change Authority**: QRATUM Platform Maintainers

All changes to governance require:

1. Pull request with rationale
2. Review by at least 2 maintainers
3. No objections from stakeholders
4. Update to version number

## Contact

For questions about discovery governance:

- Repository: <https://github.com/robertringler/QRATUM>
- Issues: <https://github.com/robertringler/QRATUM/issues>
- Email: <info@qratum.ai>

---

**Document Version**: 1.0.0  
**Last Updated**: 2025-01-01  
**Status**: Active
