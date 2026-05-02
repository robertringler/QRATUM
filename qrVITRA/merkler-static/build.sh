#!/bin/bash
# Build script for merkler-static with biokey support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== VITRA-E0 Merkler-Static Build ==="
echo "Building with biokey feature enabled..."
echo

# Build in release mode with optimizations
cargo build --release --features biokey

# Verify binary exists
if [ ! -f "target/release/merkler-static" ]; then
    echo "ERROR: Build failed - binary not found"
    exit 1
fi

# Show binary info
echo
echo "Build successful!"
echo "Binary location: $(pwd)/target/release/merkler-static"
echo "Binary size: $(du -h target/release/merkler-static | cut -f1)"
echo

# Run tests
echo "Running unit tests..."
cargo test --release --features biokey

echo
echo "=== Build Complete ==="
echo "To install: cp target/release/merkler-static /usr/local/bin/"
echo "To verify: merkler-static version"
