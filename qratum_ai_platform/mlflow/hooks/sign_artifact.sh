#!/usr/bin/env bash
# Artifact signing hook for MLflow model registry

set -e

ARTIFACT_PATH="$1"

if [ -z "$ARTIFACT_PATH" ]; then
    echo "Usage: sign_artifact.sh <artifact_path>"
    exit 1
fi

if [ ! -f "$ARTIFACT_PATH" ]; then
    echo "Error: Artifact not found: $ARTIFACT_PATH"
    exit 1
fi

SIGNATURE_PATH="${ARTIFACT_PATH}.sig"

# Simulated signing: create SHA256 checksum
# In production: use KMS/HSM or cosign for cryptographic signing
echo "Signing artifact: $ARTIFACT_PATH"
sha256sum "$ARTIFACT_PATH" | awk '{print $1}' > "$SIGNATURE_PATH"

# Create metadata
METADATA_PATH="${ARTIFACT_PATH}.metadata.json"
cat > "$METADATA_PATH" <<EOF
{
  "artifact": "$(basename $ARTIFACT_PATH)",
  "signature": "$(cat $SIGNATURE_PATH)",
  "algorithm": "SHA256",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "signer": "mlflow-hook"
}
EOF

echo "Signed $ARTIFACT_PATH -> $SIGNATURE_PATH"
echo "Metadata: $METADATA_PATH"

# Verify signature immediately
echo "Verifying signature..."
COMPUTED_HASH=$(sha256sum "$ARTIFACT_PATH" | awk '{print $1}')
STORED_HASH=$(cat "$SIGNATURE_PATH")

if [ "$COMPUTED_HASH" = "$STORED_HASH" ]; then
    echo "✓ Signature verification successful"
    exit 0
else
    echo "✗ Signature verification failed"
    exit 1
fi
