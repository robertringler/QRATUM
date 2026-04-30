#!/usr/bin/env bash
# SBOM Validation Script - QRATUM
# Validates SBOM integrity, format, and policy compliance
# Version: 1.0.0

set -euo pipefail

SBOM_DIR="${SBOM_DIR:-.sbom}"
POLICY_MAX_DEPS="${POLICY_MAX_DEPS:-1000}"
POLICY_REQUIRE_CYCLONEDX="${POLICY_REQUIRE_CYCLONEDX:-true}"
POLICY_REQUIRE_SPDX="${POLICY_REQUIRE_SPDX:-true}"

# ============================================================================
# LOGGING
# ============================================================================
log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

error() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ERROR: $*" >&2
}

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

# Check SBOM directory exists and has required files
validate_structure() {
    log "Validating SBOM directory structure..."
    
    local required_dirs=("cyclonedx" "spdx" "provenance" "hashes")
    local missing=()
    
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "${SBOM_DIR}/${dir}" ]]; then
            missing+=("$dir")
        fi
    done
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        error "Missing required directories: ${missing[*]}"
        return 1
    fi
    
    # Check for at least one SBOM
    local sbom_count
    sbom_count=$(find "${SBOM_DIR}" -name "*.json" -type f | wc -l)
    if [[ $sbom_count -eq 0 ]]; then
        error "No SBOM files found in ${SBOM_DIR}"
        return 1
    fi
    
    log "Structure validation passed. Found ${sbom_count} SBOM files."
}

# Validate CycloneDX format
validate_cyclonedx() {
    log "Validating CycloneDX SBOMs..."
    
    local failed=0
    local count=0
    
    for sbom in "${SBOM_DIR}"/cyclonedx/*.json; do
        [[ -f "$sbom" ]] || continue
        count=$((count + 1))
        
        log "  Checking: $(basename "$sbom")"
        
        if ! python3 << EOF
import json
import sys

with open('${sbom}') as f:
    data = json.load(f)

# Required fields per CycloneDX 1.6
errors = []

if data.get('bomFormat') != 'CycloneDX':
    errors.append('bomFormat must be CycloneDX')

spec = data.get('specVersion', '')
if not spec.startswith('1.'):
    errors.append(f'Invalid specVersion: {spec}')

if 'metadata' not in data:
    errors.append('Missing metadata section')
else:
    if 'timestamp' not in data['metadata']:
        errors.append('Missing metadata.timestamp')

if 'components' in data:
    for i, comp in enumerate(data['components']):
        if 'name' not in comp:
            errors.append(f'Component {i}: missing name')
        if 'version' not in comp:
            errors.append(f'Component {i}: missing version')

if errors:
    for e in errors:
        print(f'  ERROR: {e}', file=sys.stderr)
    sys.exit(1)

print(f'  Valid CycloneDX {spec} with {len(data.get("components", []))} components')
EOF
        then
            failed=1
        fi
    done
    
    if [[ "$POLICY_REQUIRE_CYCLONEDX" == "true" && $count -eq 0 ]]; then
        error "No CycloneDX SBOMs found (policy requires at least one)"
        return 1
    fi
    
    if [[ $failed -eq 1 ]]; then
        error "CycloneDX validation failed"
        return 1
    fi
    
    log "CycloneDX validation passed (${count} files)"
}

# Validate SPDX format
validate_spdx() {
    log "Validating SPDX SBOMs..."
    
    local failed=0
    local count=0
    
    for sbom in "${SBOM_DIR}"/spdx/*.json; do
        [[ -f "$sbom" ]] || continue
        count=$((count + 1))
        
        log "  Checking: $(basename "$sbom")"
        
        if ! python3 << EOF
import json
import sys

with open('${sbom}') as f:
    data = json.load(f)

errors = []

# Required SPDX 2.3 fields
version = data.get('spdxVersion', '')
if not version.startswith('SPDX-'):
    errors.append(f'Invalid spdxVersion: {version}')

if 'SPDXID' not in data:
    errors.append('Missing SPDXID')

if 'name' not in data:
    errors.append('Missing name')

if 'creationInfo' not in data:
    errors.append('Missing creationInfo')

if errors:
    for e in errors:
        print(f'  ERROR: {e}', file=sys.stderr)
    sys.exit(1)

pkg_count = len(data.get('packages', []))
print(f'  Valid {version} with {pkg_count} packages')
EOF
        then
            failed=1
        fi
    done
    
    if [[ "$POLICY_REQUIRE_SPDX" == "true" && $count -eq 0 ]]; then
        error "No SPDX SBOMs found (policy requires at least one)"
        return 1
    fi
    
    if [[ $failed -eq 1 ]]; then
        error "SPDX validation failed"
        return 1
    fi
    
    log "SPDX validation passed (${count} files)"
}

# Validate hash integrity
validate_hashes() {
    log "Validating SBOM hashes..."
    
    local checksum_file="${SBOM_DIR}/hashes/sbom-checksums.sha256"
    
    if [[ ! -f "$checksum_file" ]]; then
        error "Hash file not found: ${checksum_file}"
        return 1
    fi
    
    local failed=0
    
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        
        local expected_hash file_name
        expected_hash=$(echo "$line" | awk '{print $1}')
        file_name=$(echo "$line" | awk '{print $2}')
        
        # Find the file
        local file_path=""
        for dir in cyclonedx spdx; do
            if [[ -f "${SBOM_DIR}/${dir}/${file_name}" ]]; then
                file_path="${SBOM_DIR}/${dir}/${file_name}"
                break
            fi
        done
        
        if [[ -z "$file_path" ]]; then
            error "File not found: ${file_name}"
            failed=1
            continue
        fi
        
        local actual_hash
        actual_hash=$(sha256sum "$file_path" | awk '{print $1}')
        
        if [[ "$expected_hash" != "$actual_hash" ]]; then
            error "Hash mismatch for ${file_name}"
            error "  Expected: ${expected_hash}"
            error "  Actual:   ${actual_hash}"
            failed=1
        else
            log "  ${file_name}: OK"
        fi
    done < "$checksum_file"
    
    if [[ $failed -eq 1 ]]; then
        error "Hash validation failed"
        return 1
    fi
    
    log "Hash validation passed"
}

# Validate provenance
validate_provenance() {
    log "Validating build provenance..."
    
    local provenance_file="${SBOM_DIR}/provenance/build-provenance.json"
    
    if [[ ! -f "$provenance_file" ]]; then
        error "Provenance file not found: ${provenance_file}"
        return 1
    fi
    
    if ! python3 << EOF
import json
import sys

with open('${provenance_file}') as f:
    data = json.load(f)

errors = []

# Check SLSA provenance structure
if data.get('_type') != 'https://in-toto.io/Statement/v1':
    errors.append('Invalid _type (expected in-toto Statement v1)')

if 'subject' not in data or not data['subject']:
    errors.append('Missing or empty subject list')

if 'predicate' not in data:
    errors.append('Missing predicate')
else:
    pred = data['predicate']
    if 'buildDefinition' not in pred:
        errors.append('Missing predicate.buildDefinition')
    if 'runDetails' not in pred:
        errors.append('Missing predicate.runDetails')

if errors:
    for e in errors:
        print(f'  ERROR: {e}', file=sys.stderr)
    sys.exit(1)

print(f'  Valid provenance with {len(data.get("subject", []))} subjects')
EOF
    then
        error "Provenance validation failed"
        return 1
    fi
    
    log "Provenance validation passed"
}

# Policy: Check for dependency explosion
validate_policy_deps() {
    log "Checking dependency count policy (max: ${POLICY_MAX_DEPS})..."
    
    local total_deps=0
    
    for sbom in "${SBOM_DIR}"/cyclonedx/*.json; do
        [[ -f "$sbom" ]] || continue
        
        local count
        count=$(python3 << EOF
import json
with open('${sbom}') as f:
    data = json.load(f)
print(len(data.get('components', [])))
EOF
        )
        total_deps=$((total_deps + count))
    done
    
    if [[ $total_deps -gt $POLICY_MAX_DEPS ]]; then
        error "Dependency explosion detected: ${total_deps} dependencies (max: ${POLICY_MAX_DEPS})"
        return 1
    fi
    
    log "Dependency count check passed (${total_deps} total)"
}

# ============================================================================
# MAIN
# ============================================================================
main() {
    log "QRATUM SBOM Validation Starting"
    log "SBOM Directory: ${SBOM_DIR}"
    
    local failed=0
    
    validate_structure || failed=1
    validate_cyclonedx || failed=1
    validate_spdx || failed=1
    validate_hashes || failed=1
    validate_provenance || failed=1
    validate_policy_deps || failed=1
    
    if [[ $failed -eq 1 ]]; then
        error "SBOM validation FAILED"
        exit 1
    fi
    
    log "All SBOM validations PASSED"
}

main "$@"
