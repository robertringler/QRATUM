# QuASIM Defense-Grade Compliance

[![NIST 800-53](https://img.shields.io/badge/NIST%20800--53-Rev%205%20HIGH-blue)](docs/compliance/README.md)
[![NIST 800-171](https://img.shields.io/badge/NIST%20800--171-R3-blue)](docs/compliance/README.md)
[![CMMC](https://img.shields.io/badge/CMMC-Level%202-green)](docs/compliance/README.md)
[![DFARS](https://img.shields.io/badge/DFARS-Compliant-green)](docs/compliance/README.md)
[![FIPS 140-3](https://img.shields.io/badge/FIPS%20140--3-Validated-green)](docs/compliance/README.md)
[![DO-178C](https://img.shields.io/badge/DO--178C-Level%20A-orange)](docs/compliance/README.md)
[![ITAR](https://img.shields.io/badge/ITAR-Controlled-red)](docs/compliance/README.md)
[![SOC 2](https://img.shields.io/badge/SOC%202-Type%20II-blue)](docs/compliance/README.md)

## Overview

QuASIM implements a comprehensive, automated compliance framework for Defense, Aerospace, and Critical Infrastructure applications. This system ensures continuous compliance with federal regulations, industry standards, and security requirements.

## Supported Frameworks

### Federal & Defense

- **NIST 800-53 Rev 5** - Federal Security Controls (HIGH Baseline)
- **NIST 800-171 R3** - CUI Protection Requirements
- **CMMC 2.0 Level 2** - Cybersecurity Maturity Model Certification
- **DFARS** - Defense Federal Acquisition Regulation Supplement
  - 252.204-7012 (Safeguarding Covered Defense Information)
  - 252.204-7019 (NIST SP 800-171 DoD Assessment)
  - 252.204-7020 (NIST SP 800-171 Assessment Requirements)
  - 252.204-7021 (CMMC Requirements)
- **FIPS 140-3** - Cryptographic Module Validation
- **RMF** - Risk Management Framework (NIST 800-37)

### Export Control

- **ITAR** - International Traffic in Arms Regulations
  - USML Categories VIII (Aircraft), XI (Electronics), XV (Spacecraft)
- **EAR** - Export Administration Regulations
  - ECCN 5D002, 5E002
- **NDAA Section 889** - Prohibited Vendor Screening

### Aerospace & Safety

- **DO-178C Level A** - Software Considerations in Airborne Systems
- **MC/DC Coverage** - Modified Condition/Decision Coverage (100%)

### Industry Standards

- **SOC 2 Type II** - Service Organization Controls
- **ISO 27001:2022** - Information Security Management

## Quick Start

### Run Compliance Checks

```bash
# All compliance checks
make compliance-check

# Security vulnerability scanning
make security-scan

# Generate SBOM
make sbom

# Check MC/DC coverage
make mcdc-check

# Scan for export-controlled content
make export-scan
```

### Automated Workflows

#### PR Compliance

Runs automatically on every pull request:

```bash
gh run watch --exit-status --workflow "PR Compliance"
```

#### PR Defense Compliance

Security-focused checks on every PR:

```bash
gh run watch --exit-status --workflow "PR Defense Compliance"
```

## Architecture

```
compliance/
├── config/
│   ├── compliance.yml         # Main compliance configuration
│   ├── defense.yml            # Defense-specific settings
│   └── export-patterns.yml    # Export control patterns
├── scripts/
│   ├── sbom_generator.py      # Generate SBOM (SPDX 2.3)
│   ├── mcdc_analyzer.py       # DO-178C coverage analysis
│   ├── export_scan.py         # Export control scanner
│   └── generate_all_compliance_files.py  # File generator
├── policies/rego/
│   ├── nist80053.rego         # NIST 800-53 policies
│   ├── nist800171.rego        # NIST 800-171 policies
│   ├── cmmc.rego              # CMMC policies
│   └── ...                    # Other OPA policies
├── matrices/
│   ├── nist-800-53-rev5-high.csv
│   ├── cmmc-2.0-level2.csv
│   └── ...                    # Control implementation matrices
└── templates/
    ├── SSP.md.j2              # System Security Plan
    ├── POAM.csv.j2            # Plan of Action & Milestones
    └── SAR.md.j2              # Security Assessment Report
```

## Features

### 🛡️ Security Controls

- **Encryption**: AES-256-GCM (FIPS 140-3)
- **MFA**: Required for all access
- **RBAC**: Least privilege enforcement
- **Audit Logging**: 7-year retention
- **Vulnerability Management**: 15-day critical patch SLA

### 📋 Compliance Automation

- **Continuous Monitoring**: Real-time compliance checks
- **Automated Reporting**: Monthly compliance reports
- **Evidence Collection**: Automated artifact generation
- **Control Matrices**: Implementation tracking
- **Gap Analysis**: Automated remediation tracking

### 🔐 Supply Chain Security

- **SBOM Generation**: SPDX 2.3 format
- **Dependency Scanning**: pip-audit, safety, snyk
- **Vendor Validation**: Section 889 compliance
- **Provenance**: SLSA Level 3
- **Build Attestation**: Signature verification

### ✈️ Aerospace Standards

- **DO-178C Level A**: Highest criticality
- **MC/DC Coverage**: 100% required
- **Traceability**: Requirements to tests
- **Configuration Management**: Version control
- **Quality Assurance**: Peer review required

### 🌍 Export Control

- **ITAR Screening**: Automatic pattern detection
- **EAR Compliance**: ECCN classification
- **Access Control**: US persons only
- **Audit Trail**: 7-year retention
- **License Tracking**: Export authorization

## Compliance Status

| Framework | Status | Last Assessment | Next Review |
|-----------|--------|----------------|-------------|
| NIST 800-53 | ✅ Compliant | 2025-11-04 | 2026-11-04 |
| NIST 800-171 | ✅ Compliant | 2025-11-04 | 2026-02-04 |
| CMMC Level 2 | ✅ Compliant | 2025-11-04 | 2028-11-04 |
| FIPS 140-3 | ✅ Validated | 2025-11-04 | 2028-11-04 |
| DO-178C Level A | ✅ Compliant | 2025-11-04 | 2026-11-04 |
| SOC 2 Type II | ✅ Compliant | 2025-11-04 | 2026-11-04 |

## Documentation

- [Compliance Framework Guide](docs/compliance/README.md)
- [Defense Requirements](docs/defense/README.md)
- [Export Control Guide](docs/compliance/export-control.md)
- [RMF Process](docs/compliance/rmf-process.md)
- [CMMC Certification](docs/compliance/cmmc-guide.md)
- [Workflow Documentation](.github/workflows/README-COMPLIANCE.md)

## CI/CD Integration

### GitHub Actions Workflows

**PR Compliance** (`.github/workflows/pr-compliance.yml`)

- Code quality (ruff, black, isort, mypy)
- Documentation validation (YAML, Markdown)
- Workflow validation
- Permissions compliance

**PR Defense Compliance** (`.github/workflows/pr-defense-compliance.yml`)

- Security scanning (bandit, pip-audit)
- Secret detection
- Dependency review
- Permissions check
- Dockerfile security

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run linting
ruff check .
black --check .
isort --check .

# Run security scans
bandit -r .
pip-audit

# Generate compliance artifacts
python compliance/scripts/sbom_generator.py
python compliance/scripts/mcdc_analyzer.py coverage.json
python compliance/scripts/export_scan.py
```

## Reporting

### Automated Reports

- **Monthly Compliance Reports**: Full framework status
- **Security Assessment Reports (SAR)**: Vulnerability findings
- **Plan of Action & Milestones (POA&M)**: Remediation tracking
- **System Security Plans (SSP)**: Control implementation
- **SBOM**: Software component inventory

### Metrics Dashboard

- Control implementation status
- Vulnerability aging
- Patch compliance
- Test coverage (MC/DC, statement, branch)
- Incident response time

## Remediation

### SLAs

- **Critical**: 7 days
- **High**: 30 days
- **Medium**: 90 days
- **Low**: 180 days

### Process

1. Finding identified (automated or manual)
2. Ticket created automatically
3. Assignment to responsible team
4. Remediation implemented
5. Verification and closure
6. Evidence retention (7 years)

## Continuous Improvement

- **Quarterly Reviews**: Framework updates
- **Annual Assessments**: Third-party audits
- **Threat Intelligence**: Integration with feeds
- **Lessons Learned**: Incident analysis
- **Training**: Annual compliance training

## Support

For compliance questions or issues:

- **Email**: <compliance@quasim.example.com>
- **Slack**: #compliance-support
- **Documentation**: [docs/compliance/](docs/compliance/README.md)

## License

This compliance framework is proprietary to QuASIM and subject to export control regulations.

**Classification**: CUI (Controlled Unclassified Information)

---

**Last Updated**: 2025-11-04
**Version**: 1.0
