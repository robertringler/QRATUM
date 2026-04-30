# QRATUM Software Bill of Materials (SBOM) Documentation

This document describes how SBOMs are generated, validated, and verified for the QRATUM project. It is intended for security auditors, government reviewers, and operators who need to reproduce and verify supply chain integrity.

## Table of Contents

1. [Overview](#overview)
2. [Tooling Selection](#tooling-selection)
3. [File Layout](#file-layout)
4. [Generation Commands](#generation-commands)
5. [Validation Commands](#validation-commands)
6. [Verification for Auditors](#verification-for-auditors)
7. [Reproduction for Operators](#reproduction-for-operators)
8. [Policy Gates](#policy-gates)
9. [Security Hardening](#security-hardening)

---

## Overview

QRATUM generates SBOMs in two industry-standard formats:
- **CycloneDX 1.6** - Primary format for supply chain security
- **SPDX 2.3** - Secondary format for license compliance

SBOMs are generated for:
- Python dependencies (from `pyproject.toml` and installed packages)
- Rust dependencies (from `Cargo.toml` in multiple crates)
- Repository-level dependencies (combined view)
- Container images (when built)

### Non-Negotiable Constraints

1. All tool versions are pinned for reproducibility
2. SBOMs are generated DURING the build, not after
3. All SBOMs are validated before artifact upload
4. Cryptographic hashes bind SBOMs to artifacts
5. Build provenance follows SLSA specification

---

## Tooling Selection

| Component | Tool | Version | Justification |
|-----------|------|---------|---------------|
| Python SBOM | cyclonedx-py | 4.6.0 | Official CycloneDX Python tool, direct pip integration |
| Rust SBOM | cargo-cyclonedx | latest | Native Cargo integration, lockfile parsing |
| Repo SBOM | syft | 1.18.1 | Multi-language support, SPDX/CycloneDX dual output |
| Container SBOM | syft | 1.18.1 | OCI image scanning, layer-aware analysis |
| Validation | Python + jq | N/A | No additional dependencies, schema validation |
| Signing | cosign | 2.4.1 | Sigstore integration, keyless signing |

### Why These Tools?

**cyclonedx-py**: Native Python tool that reads from pip's installed packages, ensuring the SBOM reflects the actual runtime environment, not just declared dependencies.

**cargo-cyclonedx**: Reads Cargo.lock for exact pinned versions, including transitive dependencies. Falls back to syft if not available.

**syft**: Industry-standard multi-language scanner from Anchore. Produces both CycloneDX and SPDX from a single scan. Widely adopted in government and enterprise.

---

## File Layout

```
.sbom/
├── cyclonedx/
│   ├── python-cyclonedx-1.6.json     # Python environment SBOM
│   ├── rust-qratum-cyclonedx-1.6.json # Main Rust crate SBOM
│   ├── repo-cyclonedx-1.6.json       # Full repository SBOM
│   └── container-cyclonedx-1.6.json  # Container image SBOM (if built)
├── spdx/
│   ├── python-spdx-2.3.json          # Python SPDX format
│   ├── rust-spdx-2.3.json            # Rust SPDX format
│   ├── repo-spdx-2.3.json            # Repository SPDX format
│   └── container-spdx-2.3.json       # Container SPDX format
├── provenance/
│   ├── build-provenance.json         # SLSA provenance attestation
│   └── artifact-binding.json         # SBOM-to-source binding
├── hashes/
│   ├── sbom-checksums.sha256         # SHA256 of all SBOMs
│   └── sbom-checksums.sha512         # SHA512 of all SBOMs
└── diffs/
    └── *-diff.json                   # Diff from previous build (PRs only)
```

### Naming Convention

Format: `{target}-{format}-{version}.json`

- `target`: python, rust, repo, container
- `format`: cyclonedx, spdx
- `version`: 1.6 (CycloneDX) or 2.3 (SPDX)

---

## Generation Commands

### Prerequisites

```bash
# Install required tools
pip install cyclonedx-py==4.6.0
cargo install cargo-cyclonedx --locked

# Install syft (Linux)
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin v1.18.1
```

### Generate Python SBOM

```bash
# CycloneDX format
cyclonedx-py environment \
  --of json \
  --outfile .sbom/cyclonedx/python-cyclonedx-1.6.json \
  --schema-version 1.6

# SPDX format
syft dir:. \
  --source-name qratum-python \
  --output spdx-json=.sbom/spdx/python-spdx-2.3.json \
  --catalogers python
```

### Generate Rust SBOM

```bash
# From each Rust crate directory
cd qratum-rust
cargo cyclonedx \
  --format json \
  --spec-version 1.6 \
  --output-file ../.sbom/cyclonedx/rust-qratum-cyclonedx-1.6.json

# Or use syft for combined Rust SBOM
syft dir:. \
  --source-name qratum-rust \
  --output cyclonedx-json=.sbom/cyclonedx/rust-cyclonedx-1.6.json \
  --output spdx-json=.sbom/spdx/rust-spdx-2.3.json \
  --catalogers rust
```

### Generate Repository SBOM

```bash
syft dir:. \
  --source-name qratum \
  --source-version $(git rev-parse --short HEAD) \
  --output cyclonedx-json=.sbom/cyclonedx/repo-cyclonedx-1.6.json \
  --output spdx-json=.sbom/spdx/repo-spdx-2.3.json
```

### Generate Container SBOM

```bash
# Build the image first
docker build -t qratum:latest -f Dockerfile.production .

# Generate SBOM
syft qratum:latest \
  --output cyclonedx-json=.sbom/cyclonedx/container-cyclonedx-1.6.json \
  --output spdx-json=.sbom/spdx/container-spdx-2.3.json
```

### Generate Hashes

```bash
# SHA256
sha256sum .sbom/cyclonedx/*.json .sbom/spdx/*.json > .sbom/hashes/sbom-checksums.sha256

# SHA512
sha512sum .sbom/cyclonedx/*.json .sbom/spdx/*.json > .sbom/hashes/sbom-checksums.sha512
```

### Full Generation Script

```bash
./scripts/sbom/generate_sbom.sh
```

---

## Validation Commands

### Validate Structure

```bash
./scripts/sbom/validate_sbom.sh
```

### Manual Validation

```bash
# Verify JSON syntax
python3 -m json.tool .sbom/cyclonedx/python-cyclonedx-1.6.json > /dev/null

# Verify CycloneDX schema
python3 -c "
import json
with open('.sbom/cyclonedx/python-cyclonedx-1.6.json') as f:
    data = json.load(f)
assert data['bomFormat'] == 'CycloneDX'
assert data['specVersion'].startswith('1.')
assert 'metadata' in data
print('Valid CycloneDX SBOM')
"

# Verify SPDX schema
python3 -c "
import json
with open('.sbom/spdx/python-spdx-2.3.json') as f:
    data = json.load(f)
assert data['spdxVersion'].startswith('SPDX-')
assert 'SPDXID' in data
print('Valid SPDX SBOM')
"
```

### Verify Hash Integrity

```bash
cd .sbom && sha256sum -c hashes/sbom-checksums.sha256
```

---

## Verification for Auditors

### Step 1: Download SBOM Artifacts

From GitHub Release:
```bash
# Download the SBOM archive
curl -LO https://github.com/robertringler/QRATUM/releases/download/v1.0.0/sbom-v1.0.0.tar.gz
curl -LO https://github.com/robertringler/QRATUM/releases/download/v1.0.0/sbom-v1.0.0.tar.gz.sha256

# Verify archive integrity
sha256sum -c sbom-v1.0.0.tar.gz.sha256

# Extract
tar -xzf sbom-v1.0.0.tar.gz
```

From GitHub Actions (for specific commit):
```bash
# Use GitHub CLI
gh run download --name sbom-<commit-sha>
```

### Step 2: Verify SBOM Integrity

```bash
cd .sbom

# Verify all SBOM hashes
sha256sum -c hashes/sbom-checksums.sha256

# Verify provenance
python3 -c "
import json
with open('provenance/build-provenance.json') as f:
    p = json.load(f)
print(f\"Build by: {p['predicate']['runDetails']['builder']['id']}\")
print(f\"Git SHA: {p['predicate']['buildDefinition']['externalParameters']['sha']}\")
for s in p['subject']:
    print(f\"Subject: {s['name']} sha256:{s['digest']['sha256']}\")
"
```

### Step 3: Cross-Reference with Source

```bash
# Verify artifact binding
python3 -c "
import json
with open('provenance/artifact-binding.json') as f:
    b = json.load(f)
print(f\"Git SHA: {b['gitSha']}\")
for binding in b['bindings']:
    print(f\"SBOM {binding['sbom']} -> {binding['sources']}\")
"
```

### Step 4: Validate SBOM Contents

```bash
# List all Python dependencies
python3 -c "
import json
with open('cyclonedx/python-cyclonedx-1.6.json') as f:
    data = json.load(f)
for c in sorted(data['components'], key=lambda x: x['name']):
    print(f\"{c['name']}=={c['version']}\")
"
```

---

## Reproduction for Operators

### Prerequisites

1. Same OS version (Ubuntu 22.04 recommended)
2. Same Python version (3.12)
3. Same Rust version (stable)
4. Pinned tool versions as specified above

### Reproduction Steps

```bash
# Clone repository at specific commit
git clone https://github.com/robertringler/QRATUM.git
cd QRATUM
git checkout <commit-sha>

# Install exact dependencies
pip install -e ".[dev]"

# Install SBOM tools (pinned versions)
pip install cyclonedx-py==4.6.0
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin v1.18.1

# Generate SBOMs
./scripts/sbom/generate_sbom.sh

# Compare with original
diff -r .sbom/cyclonedx/ original-sbom/cyclonedx/
```

### Determinism Notes

- Python SBOM may differ slightly due to installed vs declared deps
- Rust SBOM should be identical if using same Cargo.lock
- Timestamps will differ (ignored in comparison)
- Component order may differ (use sorted comparison)

---

## Policy Gates

The CI pipeline enforces these policies:

| Policy | Threshold | Failure Condition |
|--------|-----------|-------------------|
| Missing SBOM | Any | No CycloneDX or SPDX files generated |
| Invalid SBOM | Any | JSON parse error or missing required fields |
| Dependency Explosion | 1000 | Total dependencies exceed threshold |
| Unsigned Artifact | Release only | SBOM not attached to release |
| New Dependencies | 10 per PR | More than 10 new deps added in single PR |

### Policy Configuration

Environment variables in CI:
```yaml
POLICY_MAX_DEPS: 1000
POLICY_REQUIRE_CYCLONEDX: true
POLICY_REQUIRE_SPDX: true
MAX_ADDED_DEPS: 10
```

---

## Security Hardening

### SBOM Generation Cannot Be Bypassed

1. SBOM generation is mandatory in CI (workflow runs on all pushes and PRs)
2. Validation job depends on generation job
3. Release artifact attachment depends on validation
4. No merge to main without passing SBOM checks

### SBOMs Are Immutable Post-Build

1. SBOMs are uploaded as workflow artifacts immediately after generation
2. Artifacts have 90-day retention
3. Release SBOMs are attached to immutable GitHub releases
4. SHA256/SHA512 hashes are generated and stored

### Artifact ↔ SBOM Binding Is Verifiable

1. Build provenance includes all SBOM hashes as subjects
2. Artifact binding maps SBOMs to source files
3. Git SHA embedded in provenance and binding
4. GitHub run ID provides audit trail

### Threat Model Mitigations

| Threat | Mitigation |
|--------|------------|
| Insider modifies SBOM | Hash verification, provenance attestation |
| Supply chain compromise | Pinned tool versions, reproducible builds |
| Missing SBOM bypass | Mandatory CI gate, no manual override |
| Tampered release artifact | SHA256 checksum in separate file |

---

## Quick Reference

### Generate
```bash
./scripts/sbom/generate_sbom.sh
```

### Validate
```bash
./scripts/sbom/validate_sbom.sh
```

### Verify Hashes
```bash
cd .sbom && sha256sum -c hashes/sbom-checksums.sha256
```

### View Dependencies
```bash
python3 -c "import json; print('\\n'.join(c['name']+'@'+c['version'] for c in json.load(open('.sbom/cyclonedx/python-cyclonedx-1.6.json'))['components']))"
```

---

## Contact

For questions about SBOM generation or verification:
- Open an issue on GitHub
- Reference the specific commit SHA and SBOM hash
