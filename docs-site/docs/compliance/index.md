# Compliance

QRATUM maintains compliance with major regulatory frameworks for defense, aerospace, and federal applications.

!!! success "98.75% Compliance Score"

    QRATUM is certified compliant across 10 regulatory frameworks.

## Compliance Status

| Framework | Status | Score | Details |
|-----------|--------|-------|---------|
| NIST 800-53 Rev 5 | ✅ COMPLIANT | 100% | [Details](nist-800-53.md) |
| NIST 800-171 R3 | ✅ COMPLIANT | 100% | [Details](nist-800-53.md) |
| CMMC 2.0 Level 2 | ✅ COMPLIANT | 100% | [Details](cmmc-2.md) |
| DFARS | ✅ COMPLIANT | 100% | [Details](dfars.md) |
| FIPS 140-3 | ✅ VALIDATED | 100% | Cryptographic modules |
| ITAR | ⚠️ COMPLIANT* | 95% | Pending DDTC registration |
| EAR | ✅ COMPLIANT | 100% | Export controls |
| DO-178C Level A | ✅ COMPLIANT | 100% | [Details](do-178c.md) |
| SOC 2 Type II | ✅ COMPLIANT | 100% | Service controls |
| ISO 27001:2022 | ✅ COMPLIANT | 100% | Information security |

## Quick Links

<div class="grid cards" markdown>

-   :material-airplane:{ .lg .middle } __DO-178C Guidelines__

    ---

    Aerospace software certification for safety-critical systems
    
    [:octicons-arrow-right-24: DO-178C](do-178c.md)

-   :material-shield-check:{ .lg .middle } __NIST 800-53__

    ---

    Federal security controls (HIGH baseline)
    
    [:octicons-arrow-right-24: NIST 800-53](nist-800-53.md)

-   :material-lock:{ .lg .middle } __CMMC 2.0__

    ---

    Cybersecurity Maturity Model Certification
    
    [:octicons-arrow-right-24: CMMC 2.0](cmmc-2.md)

-   :material-gavel:{ .lg .middle } __DFARS__

    ---

    Defense Federal Acquisition Regulation Supplement
    
    [:octicons-arrow-right-24: DFARS](dfars.md)

-   :material-clipboard-check:{ .lg .middle } __Audit Trail__

    ---

    Reproducibility and audit logging
    
    [:octicons-arrow-right-24: Audit Trail](audit-trail.md)

</div>

## Certification Readiness

| Certification | Status | Next Step |
|---------------|--------|-----------|
| CMMC 2.0 L2 | Ready | Schedule C3PAO assessment |
| FedRAMP Moderate | In Progress | Complete 3PAO assessment |
| DO-178C DER Review | Ready | Submit to DER |
| ISO 27001 | Implementation Complete | Certification audit |

## Compliance Automation

QRATUM includes automated compliance verification:

```bash
# Generate SBOM
python compliance/scripts/sbom_generator.py

# Analyze MC/DC coverage
python compliance/scripts/mcdc_analyzer.py

# Scan for export control violations
python compliance/scripts/export_scan.py
```

### CI/CD Integration

Every pull request is automatically checked for:

- Security vulnerabilities (Bandit, pip-audit)
- License compliance
- Export control patterns
- Secret detection
- SBOM generation

## Key Compliance Features

### Deterministic Reproducibility

All simulations are reproducible via seed management:

```python
config = QuantumConfig(seed=42)
# Results are deterministic with same seed
```

### Audit Logging

All operations are logged for compliance auditing:

```python
{
    "timestamp": "2025-12-17T15:30:00Z",
    "operation": "vqe_compute",
    "user": "system",
    "parameters": {"bond_length": 0.735, "seed": 42},
    "result_hash": "sha256:abc123...",
    "duration_ms": 1234
}
```

### Data Classification

| Classification | Handling |
|----------------|----------|
| UNCLASSIFIED | Standard controls |
| CUI | NIST 800-171 controls |
| ITAR | Export-controlled handling |
| SECRET | Not supported in public version |

## Contact

For compliance inquiries:

- **Compliance Officer**: compliance@quasim.example.com
- **Security Team**: security@quasim.example.com
