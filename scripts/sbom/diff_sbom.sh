#!/usr/bin/env bash
# SBOM Diff Script - QRATUM
# Compares SBOMs between builds to detect dependency changes
# Version: 1.0.0

set -euo pipefail

SBOM_DIR="${SBOM_DIR:-.sbom}"
BASELINE_DIR="${BASELINE_DIR:-}"
MAX_ADDED_DEPS="${MAX_ADDED_DEPS:-10}"
FAIL_ON_REMOVALS="${FAIL_ON_REMOVALS:-false}"

# ============================================================================
# LOGGING
# ============================================================================
log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

error() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ERROR: $*" >&2
}

warn() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] WARN: $*" >&2
}

# ============================================================================
# DIFF FUNCTIONS
# ============================================================================

extract_components() {
    local sbom_file="$1"
    python3 << EOF
import json
with open('${sbom_file}') as f:
    data = json.load(f)

components = data.get('components', [])
for comp in sorted(components, key=lambda x: x.get('name', '')):
    name = comp.get('name', 'unknown')
    version = comp.get('version', 'unknown')
    print(f'{name}@{version}')
EOF
}

diff_sboms() {
    local current="$1"
    local baseline="$2"
    local output_file="$3"
    
    log "Comparing: $(basename "$current") vs $(basename "$baseline")"
    
    local current_deps baseline_deps
    current_deps=$(mktemp)
    baseline_deps=$(mktemp)
    
    extract_components "$current" > "$current_deps"
    extract_components "$baseline" > "$baseline_deps"
    
    local added removed
    added=$(comm -13 "$baseline_deps" "$current_deps" || true)
    removed=$(comm -23 "$baseline_deps" "$current_deps" || true)
    
    local added_count removed_count
    added_count=$(echo "$added" | grep -c . || echo 0)
    removed_count=$(echo "$removed" | grep -c . || echo 0)
    
    # Generate diff report
    python3 << EOF
import json
from datetime import datetime, timezone

diff_report = {
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'current': '$(basename "$current")',
    'baseline': '$(basename "$baseline")',
    'added': [],
    'removed': [],
    'unchanged_count': 0
}

added_text = '''${added}'''
removed_text = '''${removed}'''

for line in added_text.strip().split('\n'):
    if line:
        diff_report['added'].append(line)

for line in removed_text.strip().split('\n'):
    if line:
        diff_report['removed'].append(line)

# Count unchanged
with open('${current_deps}') as f:
    current_set = set(line.strip() for line in f if line.strip())
with open('${baseline_deps}') as f:
    baseline_set = set(line.strip() for line in f if line.strip())

diff_report['unchanged_count'] = len(current_set & baseline_set)

with open('${output_file}', 'w') as f:
    json.dump(diff_report, f, indent=2)

print(f'Added: {len(diff_report["added"])}, Removed: {len(diff_report["removed"])}, Unchanged: {diff_report["unchanged_count"]}')
EOF
    
    rm -f "$current_deps" "$baseline_deps"
    
    # Log details
    if [[ $added_count -gt 0 ]]; then
        log "  Added dependencies:"
        echo "$added" | while read -r dep; do
            [[ -n "$dep" ]] && log "    + ${dep}"
        done
    fi
    
    if [[ $removed_count -gt 0 ]]; then
        warn "  Removed dependencies:"
        echo "$removed" | while read -r dep; do
            [[ -n "$dep" ]] && warn "    - ${dep}"
        done
    fi
    
    echo "$added_count:$removed_count"
}

# ============================================================================
# MAIN
# ============================================================================
main() {
    log "QRATUM SBOM Diff Starting"
    log "Current SBOM Dir: ${SBOM_DIR}"
    log "Baseline SBOM Dir: ${BASELINE_DIR}"
    
    if [[ -z "$BASELINE_DIR" ]]; then
        log "No baseline provided. Skipping diff."
        exit 0
    fi
    
    if [[ ! -d "$BASELINE_DIR" ]]; then
        log "Baseline directory not found. First build - skipping diff."
        exit 0
    fi
    
    local failed=0
    local total_added=0
    local total_removed=0
    
    mkdir -p "${SBOM_DIR}/diffs"
    
    # Compare each SBOM type
    for current in "${SBOM_DIR}"/cyclonedx/*.json; do
        [[ -f "$current" ]] || continue
        
        local basename
        basename=$(basename "$current")
        local baseline="${BASELINE_DIR}/cyclonedx/${basename}"
        
        if [[ ! -f "$baseline" ]]; then
            log "No baseline for ${basename} (new SBOM)"
            continue
        fi
        
        local diff_output="${SBOM_DIR}/diffs/${basename%.json}-diff.json"
        local result
        result=$(diff_sboms "$current" "$baseline" "$diff_output")
        
        local added removed
        added=$(echo "$result" | cut -d: -f1)
        removed=$(echo "$result" | cut -d: -f2)
        
        total_added=$((total_added + added))
        total_removed=$((total_removed + removed))
    done
    
    log "Total changes: +${total_added} / -${total_removed}"
    
    # Policy checks
    if [[ $total_added -gt $MAX_ADDED_DEPS ]]; then
        error "Too many new dependencies added: ${total_added} (max: ${MAX_ADDED_DEPS})"
        failed=1
    fi
    
    if [[ "$FAIL_ON_REMOVALS" == "true" && $total_removed -gt 0 ]]; then
        error "Dependencies removed: ${total_removed} (policy requires review)"
        failed=1
    fi
    
    if [[ $failed -eq 1 ]]; then
        error "SBOM diff check FAILED"
        exit 1
    fi
    
    log "SBOM diff check PASSED"
}

main "$@"
