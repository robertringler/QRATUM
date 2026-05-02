#!/bin/bash
# Validation script for SOI UE5 migration setup

# Don't exit on error - we want to collect all issues
set +e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  QRATUM SOI - UE5 Migration Setup Validator             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ERRORS=0
WARNINGS=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

error() {
    echo -e "${RED}✗${NC} $1"
    ((ERRORS++))
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

info() {
    echo -e "  $1"
}

echo "=== Prerequisites Check ==="
echo ""

# Check Rust
if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version | cut -d' ' -f2)
    success "Rust installed: $RUST_VERSION"
    
    if command -v cargo &> /dev/null; then
        success "Cargo installed"
    else
        error "Cargo not found (should come with Rust)"
    fi
else
    error "Rust not installed"
    info "Install from: https://rustup.rs/"
fi

echo ""

# Check Unreal Engine (optional - may not be in PATH)
if command -v UnrealEditor &> /dev/null; then
    success "Unreal Editor found in PATH"
elif [ -d "/opt/UnrealEngine" ]; then
    success "Unreal Engine found at /opt/UnrealEngine"
elif [ -d "/Applications/Epic Games/UE_5.3" ]; then
    success "Unreal Engine found at /Applications/Epic Games/UE_5.3"
elif [ -d "C:\\Program Files\\Epic Games\\UE_5.3" ]; then
    success "Unreal Engine found at C:\\Program Files\\Epic Games\\UE_5.3"
else
    warning "Unreal Engine 5.3 not detected (install from Epic Games Launcher)"
fi

echo ""

# Check C++ compiler
if command -v clang++ &> /dev/null; then
    success "C++ compiler found: clang++"
elif command -v g++ &> /dev/null; then
    success "C++ compiler found: g++"
elif command -v cl &> /dev/null; then
    success "C++ compiler found: MSVC"
else
    error "No C++ compiler found"
    info "Install build tools for your platform"
fi

echo ""
echo "=== File Structure Check ==="
echo ""

# Check Rust files
if [ -f "$SCRIPT_DIR/../rust_core/soi_telemetry_core/Cargo.toml" ]; then
    success "Rust Cargo.toml found"
else
    error "Rust Cargo.toml missing"
fi

if [ -f "$SCRIPT_DIR/../rust_core/soi_telemetry_core/src/lib.rs" ]; then
    success "Rust lib.rs found"
else
    error "Rust lib.rs missing"
fi

if [ -x "$SCRIPT_DIR/../rust_core/soi_telemetry_core/build.sh" ]; then
    success "Rust build.sh found and executable"
else
    if [ -f "$SCRIPT_DIR/../rust_core/soi_telemetry_core/build.sh" ]; then
        warning "build.sh found but not executable (run: chmod +x build.sh)"
    else
        error "Rust build.sh missing"
    fi
fi

echo ""

# Check Unreal files
if [ -f "$SCRIPT_DIR/SoiGame.uproject" ]; then
    success "UE5 project file found"
else
    error "UE5 project file missing"
fi

if [ -f "$SCRIPT_DIR/Source/SoiGame/Public/SoiTelemetrySubsystem.h" ]; then
    success "C++ header found"
else
    error "C++ header missing"
fi

if [ -f "$SCRIPT_DIR/Source/SoiGame/Private/SoiTelemetrySubsystem.cpp" ]; then
    success "C++ implementation found"
else
    error "C++ implementation missing"
fi

if [ -f "$SCRIPT_DIR/Source/SoiGame/SoiGame.Build.cs" ]; then
    success "UE5 Build.cs found"
else
    error "UE5 Build.cs missing"
fi

echo ""

# Check documentation
if [ -f "$SCRIPT_DIR/README_UE5_MIGRATION.md" ]; then
    success "Migration README found"
else
    warning "Migration README missing"
fi

if [ -f "$SCRIPT_DIR/BLUEPRINT_IMPLEMENTATION_GUIDE.md" ]; then
    success "Blueprint guide found"
else
    warning "Blueprint guide missing"
fi

if [ -f "$SCRIPT_DIR/ARCHITECTURE.md" ]; then
    success "Architecture documentation found"
else
    warning "Architecture documentation missing"
fi

echo ""
echo "=== Rust Build Test ==="
echo ""

cd "$SCRIPT_DIR/../rust_core/soi_telemetry_core"

# Try to build
if cargo check --quiet 2>&1 | grep -q "error"; then
    error "Rust code has compilation errors"
    info "Run: cd rust_core/soi_telemetry_core && cargo check"
else
    success "Rust code compiles without errors"
fi

# Try to run tests
if cargo test --quiet 2>&1 | grep -q "test result: ok"; then
    success "Rust tests pass"
else
    warning "Rust tests did not pass or could not run"
fi

echo ""
echo "=== Summary ==="
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Build Rust core: cd rust_core/soi_telemetry_core && ./build.sh"
    echo "  2. Open SoiGame.uproject in Unreal Editor"
    echo "  3. Follow BLUEPRINT_IMPLEMENTATION_GUIDE.md"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Setup complete with ${WARNINGS} warning(s)${NC}"
    echo ""
    echo "You can proceed, but review the warnings above."
    exit 0
else
    echo -e "${RED}✗ Setup incomplete: ${ERRORS} error(s), ${WARNINGS} warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before proceeding."
    exit 1
fi
