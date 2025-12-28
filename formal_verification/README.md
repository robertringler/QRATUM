# Formal Verification for QRATUM

This directory contains formal specifications and machine-verifiable proofs for QRATUM's core execution semantics and safety invariants.

## Overview

QRATUM's safety depends on maintaining 8 Fatal Invariants across all operations. To ensure these invariants hold under all conditions, we provide formal specifications in three complementary systems:

1. **TLA+** - Temporal logic specification and model checking
2. **Coq** - Interactive theorem proving with dependent types
3. **Lean4** - Modern dependent type theory with excellent tactics

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

**Usage:**
```bash
# Install TLA+ Tools
# https://lamport.azurewebsites.net/tla/tools.html

# Check model
tlc contract_execution.tla
```

### Coq Proofs (`coq/`)

- `FatalInvariants.v` - Coq formalization and proofs
  - Data types: Operators, contracts, events, system state
  - Definitions: All 8 invariants as Coq propositions
  - Theorems: Initial state safety, operation preservation

**Usage:**
```bash
# Install Coq (version 8.18+)
coqc FatalInvariants.v
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
  run: tlc formal_verification/tla/contract_execution.tla

- name: Verify Coq Proofs
  run: coqc formal_verification/coq/FatalInvariants.v

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

## Future Work

- [ ] Complete Coq proofs for all operations
- [ ] Add refinement proofs (spec → implementation)
- [ ] Integrate with CBMC for C/Rust FFI verification
- [ ] Generate DO-178C artifacts automatically
- [ ] Add temporal properties (liveness, fairness)
- [ ] Verify quantum resistance of cryptographic primitives

## References

- TLA+: https://lamport.azurewebsites.net/tla/tla.html
- Coq: https://coq.inria.fr/
- Lean4: https://leanprover.github.io/lean4/doc/
- DO-178C: RTCA DO-178C Software Considerations in Airborne Systems
- CBMC: https://www.cprover.org/cbmc/
