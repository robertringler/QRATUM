#!/bin/bash
# Build reproducible Guix pack for VITRA-E0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "  VITRA-E0 Guix Pack Builder"
echo "=========================================="
echo

# Check Guix installation
if ! command -v guix &> /dev/null; then
    echo "ERROR: Guix not found. Install from: https://guix.gnu.org"
    exit 1
fi

echo "Guix version:"
guix --version | head -1
echo

# Pull latest channels
echo "Updating Guix channels..."
guix pull -C "$SCRIPT_DIR/../configs/guix_channels.scm"
echo

# Build merkler-static
echo "Building merkler-static..."
guix build -f "$SCRIPT_DIR/merkler-static.scm"
echo

# Create relocatable pack
echo "Creating relocatable pack..."
PACK_OUTPUT="$SCRIPT_DIR/../vitra-e0-pack.tar.gz"

guix pack -f tarball \
  -C gzip \
  -S /bin=bin \
  -S /share=share \
  -m "$SCRIPT_DIR/merkler-static.scm" \
  > "$PACK_OUTPUT"

echo
echo "=========================================="
echo "  Build Complete"
echo "=========================================="
echo
echo "Relocatable pack: $PACK_OUTPUT"
echo "Size: $(du -h "$PACK_OUTPUT" | cut -f1)"
echo
echo "To extract:"
echo "  tar xzf $PACK_OUTPUT -C /opt/vitra-e0"
echo
echo "Reproducible build complete!"
