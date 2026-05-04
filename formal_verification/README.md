# Formal Verification for QRATUM

This directory contains formal specifications and machine-verifiable proofs for QRATUM's core execution semantics and safety invariants.

## Overview

QRATUM's safety depends on maintaining 8 Fatal Invariants across all operations. To ensure these invariants hold under all conditions, we provide formal specifications in four complementary systems:

1. **TLA+** - Temporal logic specification and model checking
2. **Coq** - Interactive theorem proving with dependent types
3. **Lean4** - Modern dependent type theory with excellent tactics
4. **Alloy** - Lightweight formal modeling for structural properties

## 8 Fatal Invariants

1. **Dual-Control Authorization**: All CRITICAL+ operations require two independent authorizations
2. **Merkle Chain Integrity**: Each event correctly chains to previous via cryptographic hash
3. **Determinism**: Same inputs always produce same outputs
4. **Rollback Capability**: System can always rollback to any snapshot
5. **No Data Egress**: No contract can egress data without explicit authorization
6. **Auditability**: All operations recorded in immutable Merkle chain
7. **Authorization Precedence**: Execution only after authorization
8. **State Consistency**: Contract state always consistent with Merkle chain

## Files

### TLA+ Specifications (`tla/`)

- `contract_execution.tla` - Complete TLA+ specification of contract execution semantics
  - Models: Authorization, execution, completion, rollback
  - Invariants: All 8 Fatal Invariants formally specified
  - Properties: Safety (invariants always hold) and liveness (progress guarantees)

- `ledger_state_machine.tla` - Ledger state machine specification
  - Models: Ephemeral lifecycle (materialization → execution → destruction)
  - Invariants: Zero persistence, session bounds, Merkle consistency
  - Properties: Lifecycle ordering, checkpoint bounds, TXO limits

**Usage:**
```bash
# Install TLA+ Tools
# https://lamport.azurewebsites.net/tla/tools.html

# Check contract execution model
tlc contract_execution.tla

# Check ledger state machine model
tlc ledger_state_machine.tla
```

### Coq Proofs (`coq/`)

- `FatalInvariants.v` - Coq formalization and proofs
  - Data types: Operators, contracts, events, system state
  - Definitions: All 8 invariants as Coq propositions
  - Theorems: Initial state safety, operation preservation

- `ReversibleTxo.v` - Reversible TXO mechanism proofs
  - Data types: TXOs, checkpoints, ledger state
  - Invariants: Session-bound rollback, Merkle consistency
  - Theorems: Rollback correctness, determinism, state restoration

**Usage:**
```bash
# Install Coq (version 8.18+)
coqc FatalInvariants.v
coqc ReversibleTxo.v
```

### Alloy Models (`alloy/`)

- `bft_consensus.als` - BFT consensus model
  - Entities: Validators, proposals, votes, decisions
  - Invariants: Byzantine tolerance, agreement, validity, quorum intersection
  - Properties: Safety preservation, no double voting, decision quorum

**Usage:**
```bash
# Install Alloy Analyzer
# https://alloytools.org/download.html

# Analyze BFT consensus model
java -jar alloy.jar bft_consensus.als
```

### Lean4 Proofs (`lean4/`)

- `QRADLE.lean` - Lean4 formalization
  - Modern dependent types
  - Excellent tactic language
  - Mathlib integration

**Usage:**
```bash
# Install Lean4 and elan
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Build Lean project
lake build
```

## Integration

### CI/CD Pipeline

Formal verification is integrated into the CI/CD pipeline:

```yaml
- name: Verify TLA+ Specifications
  run: |
    tlc formal_verification/tla/contract_execution.tla
    tlc formal_verification/tla/ledger_state_machine.tla

- name: Verify Coq Proofs
  run: |
    coqc formal_verification/coq/FatalInvariants.v
    coqc formal_verification/coq/ReversibleTxo.v

- name: Verify Alloy Models
  run: java -jar alloy.jar formal_verification/alloy/bft_consensus.als

- name: Verify Lean4 Proofs
  run: cd formal_verification/lean4 && lake build
```

### Runtime Verification

The formal specifications can be extracted to executable code for runtime monitoring:

```python
# Extract from Coq to OCaml
coqc -extraction-file FatalInvariants.v

# Use in runtime checks
from qratum.verification import check_invariants
assert check_invariants(system_state)
```

## DO-178C Compliance

These formal specifications contribute to DO-178C Level A certification:

- **Requirements Traceability**: Each invariant traced to requirements
- **Formal Methods Credit**: Reduces testing burden (Table A-6)
- **Certification Evidence**: Machine-verifiable proofs accepted by authorities

## Verification Coverage Matrix

| Component | TLA+ | Coq | Alloy | Lean4 |
|-----------|------|-----|-------|-------|
| Contract Execution | ✓ | ✓ | - | - |
| Ledger State Machine | ✓ | - | - | - |
| Reversible TXO | - | ✓ | - | - |
| BFT Consensus | - | - | ✓ | - |
| QRADLE Semantics | - | - | - | ✓ |

## Future Work

- [x] Complete TLA+ spec for ledger state machine
- [x] Add Coq proofs for reversible TXO
- [x] Create Alloy model for BFT consensus
- [ ] Complete Coq proofs for all operations
- [ ] Add refinement proofs (spec → implementation)
- [ ] Integrate with CBMC for C/Rust FFI verification
- [ ] Generate DO-178C artifacts automatically
- [ ] Add temporal properties (liveness, fairness)
- [ ] Verify quantum resistance of cryptographic primitives

## References

- TLA+: https://lamport.azurewebsites.net/tla/tla.html
- Coq: https://coq.inria.fr/
- Alloy: https://alloytools.org/
- Lean4: https://leanprover.github.io/lean4/doc/
- DO-178C: RTCA DO-178C Software Considerations in Airborne Systems
- CBMC: https://www.cprover.org/cbmc/
