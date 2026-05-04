# QuASIM Repository Audit System

## Overview

The QuASIM Repository Audit System provides comprehensive automated auditing capabilities for code quality, security, compliance, testing, performance, and documentation.

## Features

### Audit Checks

1. **Code Quality** - Ruff linting, Pylint analysis
2. **Security** - Vulnerability scanning, secret detection
3. **Compliance** - DO-178C, NIST 800-53, CMMC 2.0, ISO 27001
4. **Test Coverage** - Line, branch, MC/DC coverage
5. **Performance** - Benchmark validation, regression detection
6. **Documentation** - Markdown linting, API completeness

## Usage

```bash
# Run full audit
make audit

# Or directly
python3 -m quasim.audit.run --full --export-json audit/audit_summary.json
```

## Scoring

- **9.0-10.0**: PASS ✅
- **7.0-8.9**: WARN ⚠️
- **0.0-6.9**: FAIL ❌

## CI Integration

Runs automatically:
- Nightly at midnight UTC
- On push to main/develop
- On pull requests

See `.github/workflows/audit.yml` for details.
