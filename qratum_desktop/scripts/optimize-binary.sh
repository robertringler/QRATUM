#!/bin/bash
# Ultra-lightweight build optimization

set -e

echo "🦀 Building QRATUM Desktop (Ultra-Lightweight Mode)"

# Step 1: Build with size optimization
cd src-tauri
cargo build --release

# Step 2: Strip symbols (already done by Cargo profile)
echo "✅ Symbols stripped via Cargo profile"

# Step 3: Compress with UPX (if available)
if command -v upx &> /dev/null; then
    echo "🗜️ Compressing with UPX..."
    upx --best --lzma target/release/qratum-desktop 2>/dev/null || upx --best target/release/qratum-desktop
    echo "✅ UPX compression complete"
else
    echo "⚠️ UPX not found, skipping compression (install from: https://upx.github.io/)"
fi

# Step 4: Report final size
if [ -f "target/release/qratum-desktop" ]; then
    BINARY_SIZE=$(du -h target/release/qratum-desktop | cut -f1)
    echo "🎉 Final binary size: $BINARY_SIZE"
else
    echo "⚠️ Binary not found at target/release/qratum-desktop"
fi

# Step 5: Build installer
cd ..
cargo tauri build

echo "✅ Build complete!"
echo "📦 Installer location: src-tauri/target/release/bundle/"
