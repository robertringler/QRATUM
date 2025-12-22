# QuASIM Compliance Framework

Comprehensive compliance automation for Defense, Aerospace, and Critical Infrastructure.

## Frameworks Supported

- **NIST 800-53 Rev 5** (HIGH Baseline)
- **NIST 800-171 R3** (CUI Protection)
- **CMMC 2.0** (Level 2)
- **DFARS** (252.204-7012, 7019, 7020, 7021)
- **FIPS 140-3** (Cryptographic Module Validation)
- **ITAR** (USML Categories VIII, XI, XV)
- **EAR** (ECCN 5D002, 5E002)
- **DO-178C** (Level A)
- **SOC 2** (Type II)
- **ISO 27001:2022**

## Quick Start

```bash
# Run all compliance checks
make compliance-check

# Generate compliance reports
make compliance-report

# Run security scans
make security-scan

# Generate SBOM
python compliance/scripts/sbom_generator.py

# Analyze MC/DC coverage
python compliance/scripts/mcdc_analyzer.py coverage.json
```

## Architecture

```
compliance/
├── config/           # Configuration files
├── scripts/          # Automation scripts
├── policies/         # OPA Rego policies
├── matrices/         # Control matrices
└── templates/        # Document templates
```

## Workflows

### PR Compliance

Runs on every pull request:

- Code quality checks
- YAML/Markdown validation
- Workflow validation

### PR Defense Compliance

Security-focused checks:

- Vulnerability scanning
- Secret detection
- Dependency review
- Permissions check
- Dockerfile security

## Monitoring

```bash
# Watch PR Compliance
gh run watch --exit-status --workflow "PR Compliance"

# Watch Defense Compliance
gh run watch --exit-status --workflow "PR Defense Compliance"
```

## Documentation

- [Defense Requirements](../defense/README.md)
- [Export Control Guide](export-control.md)
- [RMF Process](rmf-process.md)
- [CMMC Certification](cmmc-guide.md)

## Support

For compliance questions: <compliance@quasim.example.com>
