#!/bin/bash
# Demo script for SOI Rust telemetry core
# This simulates the telemetry stream without requiring Unreal Engine

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  QRATUM SOI - Rust Telemetry Core Demo                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check if Rust library is built
if [ ! -f "$SCRIPT_DIR/target/release/libsoi_telemetry_core.so" ] && \
   [ ! -f "$SCRIPT_DIR/target/release/soi_telemetry_core.dll" ] && \
   [ ! -f "$SCRIPT_DIR/target/release/libsoi_telemetry_core.dylib" ]; then
    echo "❌ Rust library not found. Building..."
    cargo build --release
fi

echo "✓ Rust telemetry core ready"
echo ""
echo "This demo shows the Rust core functionality:"
echo "  1. Telemetry state management"
echo "  2. FFI function exports"
echo "  3. Thread-safe state access"
echo ""
echo "Note: Visual implementation requires Unreal Engine 5 Editor"
echo "See: ../unreal_bridge/BLUEPRINT_IMPLEMENTATION_GUIDE.md"
echo ""

# Run tests to demonstrate functionality
echo "Running telemetry core tests..."
cargo test -- --nocapture

echo ""
echo "✅ Demo complete!"
echo ""
echo "To use with Unreal Engine 5:"
echo "  1. Open: ../unreal_bridge/SoiGame.uproject"
echo "  2. Follow: ../unreal_bridge/BLUEPRINT_IMPLEMENTATION_GUIDE.md"
echo "  3. Connect to telemetry endpoint in Blueprint"
