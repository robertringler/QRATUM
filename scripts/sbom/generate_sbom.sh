#!/usr/bin/env bash
# SBOM Generation Script - QRATUM
# Generates CycloneDX and SPDX SBOMs for all project components
# Version: 1.0.0
# Pinned tool versions for reproducibility

set -euo pipefail

# ============================================================================
# CONFIGURATION - PINNED VERSIONS
# ============================================================================
SYFT_VERSION="1.18.1"
CDXGEN_VERSION="10.10.7"
CYCLONEDX_PY_VERSION="4.6.0"
COSIGN_VERSION="2.4.1"

# Rust crate directories to scan for dependencies
RUST_CRATE_DIRS=(
    "qratum-rust"
    "crypto/kdf"
    "crypto/rng"
    "crypto/pqc"
    "q-substrate"
    "Aethernet"
    "soi/rust_core/soi_telemetry_core"
    "qrVITRA/merkler-static"
    "qratum_desktop/src-tauri"
)

# Output directories
SBOM_DIR="${SBOM_DIR:-.sbom}"
# Convert to absolute path
SBOM_DIR="$(cd "$(dirname "$SBOM_DIR")" 2>/dev/null && pwd)/$(basename "$SBOM_DIR")" || SBOM_DIR="$(pwd)/.sbom"
export SBOM_DIR

BUILD_ID="${BUILD_ID:-$(date +%Y%m%d%H%M%S)}"
GIT_SHA="${GIT_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}"
export GIT_SHA
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')}"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

error() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ERROR: $*" >&2
}

check_tool() {
    if ! command -v "$1" &> /dev/null; then
        error "Required tool not found: $1"
        return 1
    fi
}

compute_sha256() {
    local file="$1"
    sha256sum "$file" | awk '{print $1}'
}

compute_sha512() {
    local file="$1"
    sha512sum "$file" | awk '{print $1}'
}

# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================
# Layout:
#   .sbom/
#   ├── cyclonedx/
#   │   ├── python-cyclonedx-1.6.json
#   │   ├── rust-cyclonedx-1.6.json
#   │   ├── repo-cyclonedx-1.6.json
#   │   └── container-cyclonedx-1.6.json
#   ├── spdx/
#   │   ├── python-spdx-2.3.json
#   │   ├── rust-spdx-2.3.json
#   │   ├── repo-spdx-2.3.json
#   │   └── container-spdx-2.3.json
#   ├── provenance/
#   │   ├── build-provenance.json
#   │   └── artifact-binding.json
#   └── hashes/
#       └── sbom-checksums.sha256
# ============================================================================

setup_directories() {
    log "Setting up SBOM directory structure..."
    mkdir -p "${SBOM_DIR}/cyclonedx"
    mkdir -p "${SBOM_DIR}/spdx"
    mkdir -p "${SBOM_DIR}/provenance"
    mkdir -p "${SBOM_DIR}/hashes"
}

# ============================================================================
# PYTHON SBOM GENERATION
# ============================================================================
generate_python_sbom() {
    log "Generating Python SBOM..."
    
    # CycloneDX format
    if command -v cyclonedx-py &> /dev/null; then
        log "Using cyclonedx-py for Python CycloneDX SBOM..."
        cyclonedx-py environment \
            --of json \
            --outfile "${SBOM_DIR}/cyclonedx/python-cyclonedx-1.6.json" \
            --schema-version 1.6 || {
            error "cyclonedx-py failed, falling back to pip-based generation"
            generate_python_sbom_fallback
        }
    else
        generate_python_sbom_fallback
    fi
    
    # SPDX format using syft or fallback
    if command -v syft &> /dev/null; then
        log "Using syft for Python SPDX SBOM..."
        syft dir:. \
            --source-name qratum-python \
            --output spdx-json="${SBOM_DIR}/spdx/python-spdx-2.3.json" \
            --catalogers python
    else
        generate_python_spdx_fallback
    fi
    
    log "Python SBOM generation complete"
}

generate_python_spdx_fallback() {
    log "Generating Python SPDX SBOM via pip list..."
    python3 << 'EOF'
import json
import subprocess
import sys
from datetime import datetime, timezone
import os

result = subprocess.run(
    [sys.executable, '-m', 'pip', 'list', '--format=json'],
    capture_output=True, text=True, check=True
)
packages = json.loads(result.stdout)

spdx = {
    'spdxVersion': 'SPDX-2.3',
    'dataLicense': 'CC0-1.0',
    'SPDXID': 'SPDXRef-DOCUMENT',
    'name': 'qratum-python',
    'documentNamespace': f'https://github.com/robertringler/QRATUM/sbom/python',
    'creationInfo': {
        'created': datetime.now(timezone.utc).isoformat(),
        'creators': ['Tool: qratum-sbom-generator']
    },
    'packages': []
}

for pkg in packages:
    spdx['packages'].append({
        'SPDXID': f"SPDXRef-Package-{pkg['name']}",
        'name': pkg['name'],
        'versionInfo': pkg['version'],
        'downloadLocation': f"https://pypi.org/project/{pkg['name']}/{pkg['version']}/"
    })

sbom_dir = os.getenv('SBOM_DIR', '.sbom')
with open(f'{sbom_dir}/spdx/python-spdx-2.3.json', 'w') as f:
    json.dump(spdx, f, indent=2)

print(f'Generated Python SPDX SBOM with {len(packages)} packages')
EOF
}

generate_python_sbom_fallback() {
    log "Generating Python SBOM via pip freeze..."
    local uuid_val
    uuid_val=$(python3 -c "import uuid; print(uuid.uuid4())")
    
    python3 << EOF
import json
import subprocess
import sys
from datetime import datetime, timezone

# Get pip packages
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'list', '--format=json'],
    capture_output=True, text=True, check=True
)
packages = json.loads(result.stdout)

# CycloneDX 1.6 format
sbom = {
    '\$schema': 'http://cyclonedx.org/schema/bom-1.6.schema.json',
    'bomFormat': 'CycloneDX',
    'specVersion': '1.6',
    'serialNumber': 'urn:uuid:${uuid_val}',
    'version': 1,
    'metadata': {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'tools': {
            'components': [{
                'type': 'application',
                'name': 'qratum-sbom-generator',
                'version': '1.0.0'
            }]
        },
        'component': {
            'type': 'application',
            'name': 'qratum',
            'version': '${GIT_SHA}'
        }
    },
    'components': []
}

for pkg in packages:
    sbom['components'].append({
        'type': 'library',
        'name': pkg['name'],
        'version': pkg['version'],
        'purl': f"pkg:pypi/{pkg['name']}@{pkg['version']}"
    })

with open('${SBOM_DIR}/cyclonedx/python-cyclonedx-1.6.json', 'w') as f:
    json.dump(sbom, f, indent=2)
EOF
}

# ============================================================================
# RUST SBOM GENERATION
# ============================================================================
generate_rust_sbom() {
    log "Generating Rust SBOM..."
    
    for crate_dir in "${RUST_CRATE_DIRS[@]}"; do
        if [[ -f "${crate_dir}/Cargo.toml" ]]; then
            log "Processing Rust crate: ${crate_dir}"
            
            if command -v cargo-cyclonedx &> /dev/null; then
                pushd "${crate_dir}" > /dev/null
                cargo cyclonedx \
                    --format json \
                    --spec-version 1.6 \
                    --output-file "../${SBOM_DIR}/cyclonedx/rust-${crate_dir//\//-}-cyclonedx-1.6.json" 2>/dev/null || {
                    log "cargo-cyclonedx failed for ${crate_dir}"
                }
                popd > /dev/null
            fi
        fi
    done
    
    # Generate combined Rust SBOM using syft if available, otherwise fallback
    if command -v syft &> /dev/null; then
        log "Using syft for combined Rust SBOM..."
        syft dir:. \
            --source-name qratum-rust \
            --output cyclonedx-json="${SBOM_DIR}/cyclonedx/rust-cyclonedx-1.6.json" \
            --output spdx-json="${SBOM_DIR}/spdx/rust-spdx-2.3.json" \
            --catalogers rust
    else
        generate_rust_sbom_fallback
    fi
    
    log "Rust SBOM generation complete"
}

generate_rust_sbom_fallback() {
    log "Generating Rust SBOM via cargo metadata..."
    
    if ! command -v cargo &> /dev/null; then
        log "cargo not available, creating minimal Rust SBOM from Cargo.toml files..."
        
        python3 << 'EOF'
import json
import os
import glob
import re
from datetime import datetime, timezone

sbom = {
    '$schema': 'http://cyclonedx.org/schema/bom-1.6.schema.json',
    'bomFormat': 'CycloneDX',
    'specVersion': '1.6',
    'version': 1,
    'metadata': {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'component': {
            'type': 'application',
            'name': 'qratum-rust',
            'version': os.getenv('GIT_SHA', 'unknown')
        }
    },
    'components': []
}

# Parse Cargo.toml files to extract dependencies
for toml_path in glob.glob('**/Cargo.toml', recursive=True):
    try:
        with open(toml_path) as f:
            content = f.read()
        
        # Simple regex to extract [dependencies] section
        in_deps = False
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('[dependencies]') or line.startswith('[dev-dependencies]'):
                in_deps = True
                continue
            elif line.startswith('['):
                in_deps = False
                continue
            
            if in_deps and '=' in line and not line.startswith('#'):
                # Extract dep name and version
                parts = line.split('=', 1)
                dep_name = parts[0].strip()
                version_part = parts[1].strip()
                
                # Handle version = "x.y.z" or dep = { version = "..." }
                version_match = re.search(r'"([^"]+)"', version_part)
                version = version_match.group(1) if version_match else 'unknown'
                
                # Avoid duplicates
                existing = [c for c in sbom['components'] if c['name'] == dep_name]
                if not existing:
                    sbom['components'].append({
                        'type': 'library',
                        'name': dep_name,
                        'version': version,
                        'purl': f'pkg:cargo/{dep_name}@{version}'
                    })
    except Exception as e:
        pass

sbom_dir = os.getenv('SBOM_DIR', '.sbom')
with open(f'{sbom_dir}/cyclonedx/rust-cyclonedx-1.6.json', 'w') as f:
    json.dump(sbom, f, indent=2)

# Also create SPDX format
spdx = {
    'spdxVersion': 'SPDX-2.3',
    'dataLicense': 'CC0-1.0',
    'SPDXID': 'SPDXRef-DOCUMENT',
    'name': 'qratum-rust',
    'documentNamespace': f'https://github.com/robertringler/QRATUM/sbom/rust',
    'creationInfo': {
        'created': datetime.now(timezone.utc).isoformat(),
        'creators': ['Tool: qratum-sbom-generator']
    },
    'packages': []
}

for comp in sbom['components']:
    spdx['packages'].append({
        'SPDXID': f"SPDXRef-Package-{comp['name']}",
        'name': comp['name'],
        'versionInfo': comp['version'],
        'downloadLocation': 'NOASSERTION'
    })

with open(f'{sbom_dir}/spdx/rust-spdx-2.3.json', 'w') as f:
    json.dump(spdx, f, indent=2)

print(f'Generated Rust SBOM with {len(sbom["components"])} components')
EOF
        return
    fi
    
    # If cargo IS available, try cargo metadata for each crate
    for crate_dir in "${RUST_CRATE_DIRS[@]}"; do
        if [[ -f "${crate_dir}/Cargo.toml" ]]; then
            log "Generating SBOM for crate: ${crate_dir}"
            local crate_name="${crate_dir//\//-}"
            local temp_meta="/tmp/cargo_meta_${crate_name}.json"
            
            pushd "${crate_dir}" > /dev/null
            cargo metadata --format-version 1 --no-deps > "$temp_meta" 2>/dev/null || {
                log "cargo metadata failed for ${crate_dir}"
                popd > /dev/null
                continue
            }
            popd > /dev/null
            
            python3 << EOF
import json
import os
from datetime import datetime, timezone

with open('${temp_meta}') as f:
    metadata = json.load(f)

packages = metadata.get('packages', [])

sbom = {
    '\$schema': 'http://cyclonedx.org/schema/bom-1.6.schema.json',
    'bomFormat': 'CycloneDX',
    'specVersion': '1.6',
    'version': 1,
    'metadata': {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'component': {
            'type': 'application',
            'name': '${crate_name}',
            'version': os.getenv('GIT_SHA', 'unknown')
        }
    },
    'components': []
}

for pkg in packages:
    for dep in pkg.get('dependencies', []):
        version = dep.get('req', 'unknown').lstrip('^>=<~')
        sbom['components'].append({
            'type': 'library',
            'name': dep['name'],
            'version': version,
            'purl': f"pkg:cargo/{dep['name']}@{version}"
        })

sbom_dir = os.getenv('SBOM_DIR', '.sbom')
outfile = f'{sbom_dir}/cyclonedx/rust-${crate_name}-cyclonedx-1.6.json'
with open(outfile, 'w') as f:
    json.dump(sbom, f, indent=2)

print(f'Generated SBOM for ${crate_name} with {len(sbom["components"])} components')
EOF
            rm -f "$temp_meta"
        fi
    done
    
    # Create combined Rust SBOM
    log "Creating combined Rust SBOM..."
    python3 << 'EOF'
import json
import os
import glob
from datetime import datetime, timezone

sbom = {
    '$schema': 'http://cyclonedx.org/schema/bom-1.6.schema.json',
    'bomFormat': 'CycloneDX',
    'specVersion': '1.6',
    'version': 1,
    'metadata': {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'component': {
            'type': 'application',
            'name': 'qratum-rust',
            'version': os.getenv('GIT_SHA', 'unknown')
        }
    },
    'components': []
}

sbom_dir = os.getenv('SBOM_DIR', '.sbom')
seen = set()

# Aggregate from individual rust SBOMs
for path in glob.glob(f'{sbom_dir}/cyclonedx/rust-*-cyclonedx-1.6.json'):
    if 'rust-cyclonedx-1.6.json' == os.path.basename(path):
        continue
    try:
        with open(path) as f:
            data = json.load(f)
        for comp in data.get('components', []):
            key = f"{comp['name']}@{comp.get('version', 'unknown')}"
            if key not in seen:
                seen.add(key)
                sbom['components'].append(comp)
    except:
        pass

with open(f'{sbom_dir}/cyclonedx/rust-cyclonedx-1.6.json', 'w') as f:
    json.dump(sbom, f, indent=2)

# SPDX format
spdx = {
    'spdxVersion': 'SPDX-2.3',
    'dataLicense': 'CC0-1.0',
    'SPDXID': 'SPDXRef-DOCUMENT',
    'name': 'qratum-rust',
    'documentNamespace': 'https://github.com/robertringler/QRATUM/sbom/rust',
    'creationInfo': {
        'created': datetime.now(timezone.utc).isoformat(),
        'creators': ['Tool: qratum-sbom-generator']
    },
    'packages': []
}

for comp in sbom['components']:
    spdx['packages'].append({
        'SPDXID': f"SPDXRef-Package-{comp['name']}",
        'name': comp['name'],
        'versionInfo': comp.get('version', 'unknown'),
        'downloadLocation': 'NOASSERTION'
    })

with open(f'{sbom_dir}/spdx/rust-spdx-2.3.json', 'w') as f:
    json.dump(spdx, f, indent=2)

print(f'Generated combined Rust SBOM with {len(sbom["components"])} components')
EOF
}

# ============================================================================
# REPO-LEVEL SBOM GENERATION
# ============================================================================
generate_repo_sbom() {
    log "Generating repo-level SBOM..."
    
    if command -v syft &> /dev/null; then
        log "Using syft for repo-level SBOM..."
        syft dir:. \
            --source-name qratum \
            --source-version "${GIT_SHA}" \
            --output cyclonedx-json="${SBOM_DIR}/cyclonedx/repo-cyclonedx-1.6.json" \
            --output spdx-json="${SBOM_DIR}/spdx/repo-spdx-2.3.json"
    elif command -v cdxgen &> /dev/null; then
        log "Using cdxgen for repo-level SBOM..."
        cdxgen -o "${SBOM_DIR}/cyclonedx/repo-cyclonedx-1.6.json" \
            --spec-version 1.6 \
            --format json
    else
        log "syft/cdxgen not available, generating minimal repo SBOM..."
        generate_repo_sbom_fallback
    fi
    
    log "Repo-level SBOM generation complete"
}

generate_repo_sbom_fallback() {
    python3 << 'EOF'
import json
import os
import glob
from datetime import datetime, timezone

sbom = {
    '$schema': 'http://cyclonedx.org/schema/bom-1.6.schema.json',
    'bomFormat': 'CycloneDX',
    'specVersion': '1.6',
    'version': 1,
    'metadata': {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'component': {
            'type': 'application',
            'name': 'qratum',
            'version': os.getenv('GIT_SHA', 'unknown'),
            'description': 'QRATUM - Sovereign AI Platform'
        },
        'tools': {
            'components': [{
                'type': 'application',
                'name': 'qratum-sbom-generator',
                'version': '1.0.0'
            }]
        }
    },
    'components': []
}

# Aggregate components from Python and Rust SBOMs
sbom_dir = os.getenv('SBOM_DIR', '.sbom')
seen = set()

for sbom_path in glob.glob(f'{sbom_dir}/cyclonedx/*.json'):
    if 'repo-' in sbom_path:
        continue
    try:
        with open(sbom_path) as f:
            data = json.load(f)
        for comp in data.get('components', []):
            key = f"{comp['name']}@{comp.get('version', 'unknown')}"
            if key not in seen:
                seen.add(key)
                sbom['components'].append(comp)
    except:
        pass

with open(f'{sbom_dir}/cyclonedx/repo-cyclonedx-1.6.json', 'w') as f:
    json.dump(sbom, f, indent=2)

# SPDX format
spdx = {
    'spdxVersion': 'SPDX-2.3',
    'dataLicense': 'CC0-1.0',
    'SPDXID': 'SPDXRef-DOCUMENT',
    'name': 'qratum',
    'documentNamespace': f'https://github.com/robertringler/QRATUM/sbom/repo',
    'creationInfo': {
        'created': datetime.now(timezone.utc).isoformat(),
        'creators': ['Tool: qratum-sbom-generator']
    },
    'packages': []
}

for comp in sbom['components']:
    spdx['packages'].append({
        'SPDXID': f"SPDXRef-Package-{comp['name']}",
        'name': comp['name'],
        'versionInfo': comp.get('version', 'unknown'),
        'downloadLocation': 'NOASSERTION'
    })

with open(f'{sbom_dir}/spdx/repo-spdx-2.3.json', 'w') as f:
    json.dump(spdx, f, indent=2)

print(f'Generated repo SBOM with {len(sbom["components"])} components')
EOF
}

# ============================================================================
# CONTAINER IMAGE SBOM GENERATION
# ============================================================================
generate_container_sbom() {
    local image_ref="${1:-qratum:latest}"
    log "Generating container SBOM for: ${image_ref}"
    
    if command -v syft &> /dev/null; then
        syft "${image_ref}" \
            --output cyclonedx-json="${SBOM_DIR}/cyclonedx/container-cyclonedx-1.6.json" \
            --output spdx-json="${SBOM_DIR}/spdx/container-spdx-2.3.json"
    else
        log "syft not available, skipping container SBOM"
    fi
    
    log "Container SBOM generation complete"
}

# ============================================================================
# SBOM VALIDATION
# ============================================================================
validate_sboms() {
    log "Validating generated SBOMs..."
    local validation_failed=0
    
    # Validate CycloneDX SBOMs
    for sbom in "${SBOM_DIR}"/cyclonedx/*.json; do
        if [[ -f "$sbom" ]]; then
            log "Validating: ${sbom}"
            
            # Basic JSON validation
            if ! python3 -m json.tool "$sbom" > /dev/null 2>&1; then
                error "Invalid JSON in ${sbom}"
                validation_failed=1
                continue
            fi
            
            # Check required CycloneDX fields
            if ! python3 << EOF
import json
import sys

with open('${sbom}') as f:
    sbom = json.load(f)

required_fields = ['bomFormat', 'specVersion', 'metadata']
missing = [f for f in required_fields if f not in sbom]

if missing:
    print(f'Missing required fields: {missing}', file=sys.stderr)
    sys.exit(1)

if sbom.get('bomFormat') != 'CycloneDX':
    print('bomFormat must be CycloneDX', file=sys.stderr)
    sys.exit(1)

print('Valid CycloneDX SBOM')
EOF
            then
                error "CycloneDX validation failed for ${sbom}"
                validation_failed=1
            fi
        fi
    done
    
    # Validate SPDX SBOMs
    for sbom in "${SBOM_DIR}"/spdx/*.json; do
        if [[ -f "$sbom" ]]; then
            log "Validating: ${sbom}"
            
            if ! python3 -m json.tool "$sbom" > /dev/null 2>&1; then
                error "Invalid JSON in ${sbom}"
                validation_failed=1
                continue
            fi
            
            # Check required SPDX fields
            if ! python3 << EOF
import json
import sys

with open('${sbom}') as f:
    sbom = json.load(f)

required_fields = ['spdxVersion', 'SPDXID', 'name']
missing = [f for f in required_fields if f not in sbom]

if missing:
    print(f'Missing required fields: {missing}', file=sys.stderr)
    sys.exit(1)

if not sbom.get('spdxVersion', '').startswith('SPDX-'):
    print('Invalid spdxVersion format', file=sys.stderr)
    sys.exit(1)

print('Valid SPDX SBOM')
EOF
            then
                error "SPDX validation failed for ${sbom}"
                validation_failed=1
            fi
        fi
    done
    
    if [[ $validation_failed -eq 1 ]]; then
        error "SBOM validation failed"
        return 1
    fi
    
    log "All SBOM validations passed"
}

# ============================================================================
# HASH GENERATION
# ============================================================================
generate_hashes() {
    log "Generating SBOM hashes..."
    
    local checksum_file="${SBOM_DIR}/hashes/sbom-checksums.sha256"
    local sha512_file="${SBOM_DIR}/hashes/sbom-checksums.sha512"
    
    : > "$checksum_file"
    : > "$sha512_file"
    
    for sbom in "${SBOM_DIR}"/cyclonedx/*.json "${SBOM_DIR}"/spdx/*.json; do
        if [[ -f "$sbom" ]]; then
            local basename
            basename=$(basename "$sbom")
            local sha256
            sha256=$(compute_sha256 "$sbom")
            local sha512
            sha512=$(compute_sha512 "$sbom")
            
            echo "${sha256}  ${basename}" >> "$checksum_file"
            echo "${sha512}  ${basename}" >> "$sha512_file"
        fi
    done
    
    log "Hash files generated: ${checksum_file}, ${sha512_file}"
}

# ============================================================================
# PROVENANCE GENERATION
# ============================================================================
generate_provenance() {
    log "Generating build provenance..."
    
    local provenance_file="${SBOM_DIR}/provenance/build-provenance.json"
    
    python3 << EOF
import json
import os
import subprocess
import glob
import hashlib
import shutil
from datetime import datetime, timezone

# Get environment info
def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except:
        return 'unknown'

provenance = {
    '_type': 'https://in-toto.io/Statement/v1',
    'subject': [],
    'predicateType': 'https://slsa.dev/provenance/v1',
    'predicate': {
        'buildDefinition': {
            'buildType': 'https://github.com/qratum/build/v1',
            'externalParameters': {
                'repository': os.getenv('GITHUB_REPOSITORY', 'local'),
                'ref': os.getenv('GITHUB_REF', '${GIT_BRANCH}'),
                'sha': os.getenv('GITHUB_SHA', '${GIT_SHA}')
            },
            'internalParameters': {
                'buildId': '${BUILD_ID}',
                'os': run_cmd(['uname', '-a']),
                'pythonVersion': run_cmd(['python3', '--version']),
                'rustVersion': run_cmd(['rustc', '--version']) if shutil.which('rustc') else 'not installed'
            }
        },
        'runDetails': {
            'builder': {
                'id': os.getenv('GITHUB_ACTIONS', 'local-build'),
                'version': {
                    'syft': '${SYFT_VERSION}',
                    'cyclonedx-py': '${CYCLONEDX_PY_VERSION}'
                }
            },
            'metadata': {
                'invocationId': os.getenv('GITHUB_RUN_ID', '${BUILD_ID}'),
                'startedOn': datetime.now(timezone.utc).isoformat()
            }
        }
    }
}

# Add SBOM subjects
for sbom_path in glob.glob('${SBOM_DIR}/cyclonedx/*.json') + glob.glob('${SBOM_DIR}/spdx/*.json'):
    with open(sbom_path, 'rb') as f:
        content = f.read()
        sha256 = hashlib.sha256(content).hexdigest()
    provenance['subject'].append({
        'name': os.path.basename(sbom_path),
        'digest': {
            'sha256': sha256
        }
    })

with open('${provenance_file}', 'w') as f:
    json.dump(provenance, f, indent=2)

print('Build provenance generated')
EOF
    
    # Generate artifact binding
    log "Generating artifact binding..."
    
    local binding_file="${SBOM_DIR}/provenance/artifact-binding.json"
    
    python3 << EOF
import json
import hashlib
import glob
import os
from datetime import datetime, timezone

binding = {
    'version': '1.0.0',
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'buildId': '${BUILD_ID}',
    'gitSha': '${GIT_SHA}',
    'gitBranch': '${GIT_BRANCH}',
    'bindings': []
}

# Map SBOMs to their targets
targets = {
    'python': ['pyproject.toml', 'requirements.txt'],
    'rust': ['qratum-rust/Cargo.toml', 'crypto/kdf/Cargo.toml'],
    'repo': ['.'],
    'container': ['Dockerfile', 'Dockerfile.production']
}

for sbom_path in glob.glob('${SBOM_DIR}/cyclonedx/*.json'):
    with open(sbom_path, 'rb') as f:
        content = f.read()
        sha256 = hashlib.sha256(content).hexdigest()
    
    basename = os.path.basename(sbom_path)
    for target_type, sources in targets.items():
        if target_type in basename:
            binding['bindings'].append({
                'sbom': basename,
                'sbomSha256': sha256,
                'targetType': target_type,
                'sources': sources
            })
            break

with open('${binding_file}', 'w') as f:
    json.dump(binding, f, indent=2)

print('Artifact binding generated')
EOF
    
    log "Provenance generation complete"
}

# ============================================================================
# MAIN
# ============================================================================
main() {
    log "QRATUM SBOM Generation Starting"
    log "Build ID: ${BUILD_ID}"
    log "Git SHA: ${GIT_SHA}"
    log "Git Branch: ${GIT_BRANCH}"
    
    setup_directories
    generate_python_sbom
    generate_rust_sbom
    generate_repo_sbom
    
    # Container SBOM only if image exists
    if [[ -n "${CONTAINER_IMAGE:-}" ]]; then
        generate_container_sbom "${CONTAINER_IMAGE}"
    fi
    
    validate_sboms
    generate_hashes
    generate_provenance
    
    log "SBOM generation complete. Artifacts in: ${SBOM_DIR}"
    log "Files generated:"
    find "${SBOM_DIR}" -type f -name "*.json" -o -name "*.sha*" | sort
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
