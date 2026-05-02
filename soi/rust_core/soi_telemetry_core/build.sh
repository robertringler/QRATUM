#!/bin/bash
# Build script for SOI Rust Telemetry Core

set -e  # Exit on error

echo "================================"
echo "SOI Telemetry Core Build Script"
echo "================================"

# Navigate to Rust crate directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Error: Rust is not installed"
    echo "Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

echo "✓ Rust toolchain found: $(rustc --version)"

# Clean previous builds (optional)
if [ "$1" == "--clean" ]; then
    echo "🧹 Cleaning previous builds..."
    cargo clean
fi

# Build in release mode
echo "🔨 Building Rust telemetry core..."
cargo build --release

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    # Show output location
    if [ -f "target/release/libsoi_telemetry_core.so" ]; then
        echo "📦 Output: target/release/libsoi_telemetry_core.so"
        ls -lh target/release/libsoi_telemetry_core.so
    elif [ -f "target/release/soi_telemetry_core.dll" ]; then
        echo "📦 Output: target/release/soi_telemetry_core.dll"
        ls -lh target/release/soi_telemetry_core.dll
    elif [ -f "target/release/libsoi_telemetry_core.dylib" ]; then
        echo "📦 Output: target/release/libsoi_telemetry_core.dylib"
        ls -lh target/release/libsoi_telemetry_core.dylib
    fi
    
    # Show exported symbols (Linux/macOS only)
    if command -v nm &> /dev/null; then
        echo ""
        echo "📋 Exported FFI functions:"
        if [ -f "target/release/libsoi_telemetry_core.so" ]; then
            nm -gD target/release/libsoi_telemetry_core.so | grep soi_ || true
        elif [ -f "target/release/libsoi_telemetry_core.dylib" ]; then
            nm -gU target/release/libsoi_telemetry_core.dylib | grep soi_ || true
        fi
    fi
    
    echo ""
    echo "✨ Ready for Unreal Engine integration!"
    echo "Next step: Open SoiGame.uproject in Unreal Editor"
else
    echo "❌ Build failed"
    exit 1
fi
