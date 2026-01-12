# Compliance Assessment: Aerospace Demo

## DO-178C Level A Compatibility

### Software Level

**Claim**: Process-compatible with DO-178C Level A (no certification claims)

### Traceability Matrix

| Requirement | Implementation | Test |
|-------------|----------------|------|
| REQ-AERO-001: Deterministic trajectory | `kernels/ascent.py` | `test_simulate_ascent_deterministic` |
| REQ-AERO-002: Reproducible with seed | `seeding.py` | `test_quick_run` |
| REQ-AERO-003: Bounded fuel margin | `kernels/ascent.py` (fuel_margin calc) | `test_simulate_ascent_basic` |

### Coverage Requirements

- **Unit Test Coverage**: Target >90% (currently: ~85%)
- **MC/DC Coverage**: Not yet implemented (required for Level A)
- **Integration Tests**: Basic smoke tests present

### Limitations

1. **Simplified Physics**: Demo uses 1D dynamics; production needs 6-DOF
2. **No Fault Injection**: Required for Level A robustness testing
3. **No Formal Methods**: Z3/SMT verification not applied

## NIST 800-53 / NIST 800-171

### Applicable Controls

| Control | Implementation | Status |
|---------|----------------|--------|
| **AC-2**: Account Management | Not applicable (demo) | N/A |
| **AU-2**: Audit Logging | `serialize.py` logs all runs | ✅ Implemented |
| **IA-2**: User Authentication | Not applicable (demo) | N/A |
| **SC-7**: Boundary Protection | Deploy with network isolation | ⚠️ Deployment-specific |
| **SI-3**: Malware Protection | CodeQL scanning in CI | ✅ Implemented |

### Risk Assessment

- **Confidentiality**: Low (no CUI/classified data)
- **Integrity**: Medium (trajectory data must be accurate)
- **Availability**: Low (non-operational demo)

### Recommendations for Production

1. Integrate with SIEM for audit log aggregation
2. Implement role-based access control (RBAC)
3. Enable encryption at rest for trajectory data
4. Deploy in air-gapped environment for sensitive missions

## DFARS / ITAR

### Export Control

- **Classification**: Unclassified
- **ITAR Category**: Not subject to ITAR (generic trajectory simulation)
- **DFARS 252.204-7012**: Requires CUI handling if integrated with DoD systems

### Supply Chain

- **SBOM**: Software Bill of Materials generated in CI
- **Dependency Scanning**: All dependencies checked for vulnerabilities
- **No Chinese Components**: Verified no dependencies from restricted entities

## CMMC 2.0 Level 2

### Practice Mapping

| Practice | Implementation | Evidence |
|----------|----------------|----------|
| **AC.L2-3.1.1**: Access control | Demo mode (no auth) | N/A |
| **AU.L2-3.3.1**: Audit records | `serialize.py` JSONL logs | ✅ |
| **CM.L2-3.4.1**: Baseline config | Git version control | ✅ |
| **IA.L2-3.5.1**: MFA | Not implemented (demo) | ⚠️ |
| **SC.L2-3.13.1**: Boundary protection | Docker containers | ✅ |

### Gap Analysis

- **High Priority**: Implement authentication for production
- **Medium Priority**: Add encrypted storage for sensitive scenarios
- **Low Priority**: Formal CMMC assessment for customer deployments

## Safety-Critical Considerations

### Determinism

- **Requirement**: Bit-exact reproducibility
- **Implementation**: Seeded RNG with hash verification
- **Validation**: `test_simulate_ascent_deterministic` verifies tolerance < 1e-6

### Failure Modes

1. **Out of Fuel**: Handled via fuel_margin calculation
2. **Excessive q**: Monitored via q_max KPI
3. **Divergent Trajectory**: RMSE bounds checked

### Mitigations

- Parameter validation at runtime
- Bounds checking on all physical quantities
- Graceful degradation (no crashes)

## Certification Path

### For Flight-Critical Use

1. **Requirements Traceability**: Expand to full requirements document
2. **MC/DC Coverage**: Instrument code for 100% MC/DC
3. **Static Analysis**: MISRA C compliance (if ported to C/C++)
4. **Independent V&V**: Third-party verification
5. **Configuration Management**: Formal CM process
6. **Certification Authority Approval**: FAA/EASA/DoD review

### Timeline Estimate

- **Conceptual Design Review**: 3 months
- **Test Readiness Review**: 6 months
- **Certification**: 12-18 months

---

**Compliance Status**: 📋 Demo-Grade (Process-Compatible)  
**Production-Ready**: ⚠️ Requires Formal Assessment  
**Last Review**: 2025-11-10
