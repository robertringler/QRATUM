#!/usr/bin/env bash
# SBOM (Software Bill of Materials) generation script

set -e

OUTPUT_FILE="${1:-sbom.json}"

echo "Generating SBOM..."

# Generate Python package SBOM
if command -v pip &> /dev/null; then
    echo "Collecting Python dependencies..."
    pip list --format=json > /tmp/pip_packages.json
fi

# Generate container image SBOM if syft is available
if command -v syft &> /dev/null; then
    echo "Scanning container images..."
    syft packages . -o json > /tmp/syft_packages.json 2>/dev/null || echo "[]" > /tmp/syft_packages.json
fi

# Combine into unified SBOM
cat > "$OUTPUT_FILE" <<EOF
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.4",
  "version": 1,
  "metadata": {
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "tools": ["pip", "syft"],
    "component": {
      "type": "application",
      "name": "qratum-ai-platform",
      "version": "1.0.0"
    }
  },
  "components": $(cat /tmp/pip_packages.json 2>/dev/null || echo "[]")
}
EOF

echo "SBOM generated: $OUTPUT_FILE"
echo "Total components: $(jq '.components | length' "$OUTPUT_FILE" 2>/dev/null || echo 'N/A')"
