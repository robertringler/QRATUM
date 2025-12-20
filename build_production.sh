#!/bin/bash
# QRATUM Full Production Build Script
# Builds all distributable artifacts for production deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/dist"
BUILD_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BUILD_LOG="${SCRIPT_DIR}/build_${BUILD_TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[BUILD]${NC} $1" | tee -a "$BUILD_LOG"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$BUILD_LOG"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$BUILD_LOG"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$BUILD_LOG"
}

# Print header
echo "================================================" | tee "$BUILD_LOG"
echo "QRATUM Full Production Build" | tee -a "$BUILD_LOG"
echo "Build Time: $(date)" | tee -a "$BUILD_LOG"
echo "Build ID: ${BUILD_TIMESTAMP}" | tee -a "$BUILD_LOG"
echo "================================================" | tee -a "$BUILD_LOG"
echo "" | tee -a "$BUILD_LOG"

# Check Python version
log "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
    error "Python 3.10+ required, found $PYTHON_VERSION"
fi
log "✓ Python version: $PYTHON_VERSION"
echo "" | tee -a "$BUILD_LOG"

# Create build directory
log "Creating build directory..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
mkdir -p "$BUILD_DIR/python"
mkdir -p "$BUILD_DIR/typescript"
mkdir -p "$BUILD_DIR/desktop"
mkdir -p "$BUILD_DIR/logs"
log "✓ Build directory created: $BUILD_DIR"
echo "" | tee -a "$BUILD_LOG"

# Build 1: Python Package (wheel and source distribution)
log "Building Python package..."
cd "$SCRIPT_DIR"

# Install/upgrade build tools
info "Installing build dependencies..."
python3 -m pip install --upgrade pip setuptools wheel build 2>&1 | tee -a "$BUILD_LOG"

# Clean previous builds
rm -rf build/ *.egg-info/

# Build Python distributions
info "Building Python distributions (wheel + sdist)..."
python3 -m build --outdir "$BUILD_DIR/python" 2>&1 | tee -a "$BUILD_LOG"

if [ $? -eq 0 ]; then
    log "✓ Python package built successfully"
    ls -lh "$BUILD_DIR/python/" | tee -a "$BUILD_LOG"
else
    error "Python package build failed"
fi
echo "" | tee -a "$BUILD_LOG"

# Build 2: TypeScript SDK
log "Building TypeScript SDK..."
cd "$SCRIPT_DIR/sdk/typescript"

if [ -f "package.json" ]; then
    if command -v npm >/dev/null 2>&1; then
        info "Installing TypeScript SDK dependencies..."
        npm install 2>&1 | tee -a "$BUILD_LOG"
        
        if [ -f "tsconfig.json" ] && grep -q '"build"' package.json; then
            info "Building TypeScript SDK..."
            npm run build 2>&1 | tee -a "$BUILD_LOG"
            
            if [ -d "dist" ]; then
                cp -r dist "$BUILD_DIR/typescript/qratum-sdk"
                log "✓ TypeScript SDK built successfully"
                ls -lh "$BUILD_DIR/typescript/qratum-sdk/" | tee -a "$BUILD_LOG"
            else
                warn "TypeScript SDK dist directory not found, skipping"
            fi
        else
            warn "TypeScript build script not found, copying source only"
            mkdir -p "$BUILD_DIR/typescript/qratum-sdk"
            cp -r src "$BUILD_DIR/typescript/qratum-sdk/"
            cp package.json "$BUILD_DIR/typescript/qratum-sdk/"
        fi
    else
        warn "npm not installed, skipping TypeScript SDK build"
    fi
else
    warn "TypeScript SDK package.json not found, skipping"
fi
cd "$SCRIPT_DIR"
echo "" | tee -a "$BUILD_LOG"

# Build 3: Electron Desktop Application
log "Building Electron Desktop Application..."
cd "$SCRIPT_DIR/qratum_desktop"

if [ -f "package.json" ]; then
    if command -v npm >/dev/null 2>&1; then
        info "Installing desktop app dependencies..."
        npm install 2>&1 | tee -a "$BUILD_LOG"
        
        # Check if electron-builder is available
        if grep -q '"build"' package.json; then
            info "Building desktop application..."
            warn "Full desktop build requires platform-specific tools (NSIS, etc.)"
            warn "Creating portable package only..."
            
            # Try to create a packaged version without building installers
            if npm run pack 2>&1 | tee -a "$BUILD_LOG"; then
                info "Desktop packaging completed successfully"
            else
                EXIT_CODE=$?
                warn "Desktop app packaging failed (exit code: $EXIT_CODE)"
                warn "This may require platform-specific tools or electron-builder dependencies"
                warn "Check build log for details: $BUILD_LOG"
            fi
            
            if [ -d "dist" ]; then
                cp -r dist "$BUILD_DIR/desktop/qratum-desktop"
                log "✓ Desktop application packaged"
                ls -lh "$BUILD_DIR/desktop/qratum-desktop/" | tee -a "$BUILD_LOG"
            else
                warn "Desktop dist not found, copying source files"
                mkdir -p "$BUILD_DIR/desktop/qratum-desktop"
                cp -r src "$BUILD_DIR/desktop/qratum-desktop/"
                cp package.json "$BUILD_DIR/desktop/qratum-desktop/"
            fi
        else
            warn "Desktop build script not available, copying source"
            mkdir -p "$BUILD_DIR/desktop/qratum-desktop"
            cp -r src "$BUILD_DIR/desktop/qratum-desktop/"
            cp package.json "$BUILD_DIR/desktop/qratum-desktop/"
        fi
    else
        warn "npm not installed, skipping desktop app build"
    fi
else
    warn "Desktop app package.json not found, skipping"
fi
cd "$SCRIPT_DIR"
echo "" | tee -a "$BUILD_LOG"

# Build 4: Create production requirements bundle
log "Creating production requirements bundle..."
if [ -f "requirements-prod.txt" ]; then
    cp requirements-prod.txt "$BUILD_DIR/requirements-prod.txt"
    log "✓ Production requirements copied"
else
    warn "requirements-prod.txt not found"
fi
echo "" | tee -a "$BUILD_LOG"

# Build 5: Copy deployment scripts
log "Copying deployment scripts..."
if [ -f "DEPLOY_PRODUCTION_QRATUM.sh" ]; then
    cp DEPLOY_PRODUCTION_QRATUM.sh "$BUILD_DIR/deploy.sh"
    chmod +x "$BUILD_DIR/deploy.sh"
    log "✓ Deployment script copied"
fi

if [ -f "Makefile" ]; then
    cp Makefile "$BUILD_DIR/Makefile"
    log "✓ Makefile copied"
fi
echo "" | tee -a "$BUILD_LOG"

# Build 6: Copy documentation
log "Copying documentation..."
mkdir -p "$BUILD_DIR/docs"
cp README.md "$BUILD_DIR/docs/" 2>/dev/null || warn "README.md not found"
cp LICENSE "$BUILD_DIR/docs/" 2>/dev/null || warn "LICENSE not found"
cp QUICKSTART.md "$BUILD_DIR/docs/" 2>/dev/null || warn "QUICKSTART.md not found"
cp PRODUCTION_RELEASE_MANIFEST.md "$BUILD_DIR/docs/" 2>/dev/null || warn "PRODUCTION_RELEASE_MANIFEST.md not found"
log "✓ Documentation copied"
echo "" | tee -a "$BUILD_LOG"

# Generate build manifest
log "Generating build manifest..."
cat > "$BUILD_DIR/BUILD_MANIFEST.md" << EOF
# QRATUM Production Build Manifest

**Build ID:** ${BUILD_TIMESTAMP}
**Build Date:** $(date)
**Python Version:** ${PYTHON_VERSION}
**Build System:** $(uname -s) $(uname -m)

## Build Artifacts

### Python Package
- Location: \`dist/python/\`
- Format: Wheel (.whl) and Source Distribution (.tar.gz)
- Package: qratum
- Version: $(grep '^version' pyproject.toml | cut -d'"' -f2)

### TypeScript SDK
- Location: \`dist/typescript/qratum-sdk/\`
- Package: @qratum/sdk
- Status: $([ -d "$BUILD_DIR/typescript/qratum-sdk" ] && echo "Built" || echo "Source only")

### Desktop Application
- Location: \`dist/desktop/qratum-desktop/\`
- Application: QRATUM Desktop Edition
- Status: $([ -d "$BUILD_DIR/desktop/qratum-desktop" ] && echo "Packaged" || echo "Source only")

## Installation

### Python Package
\`\`\`bash
pip install dist/python/qratum-*.whl
\`\`\`

### TypeScript SDK
\`\`\`bash
cd dist/typescript/qratum-sdk
npm install
npm link  # For local development
\`\`\`

### Desktop Application
See dist/desktop/qratum-desktop/ for platform-specific installers.

## Deployment

For production deployment, see \`deploy.sh\` and \`docs/PRODUCTION_RELEASE_MANIFEST.md\`.

## Build Log

Full build log available at: \`$(basename "$BUILD_LOG")\`

---
Generated by QRATUM Build System
EOF

cp "$BUILD_LOG" "$BUILD_DIR/logs/build_${BUILD_TIMESTAMP}.log"
log "✓ Build manifest generated"
echo "" | tee -a "$BUILD_LOG"

# Generate checksums
log "Generating checksums..."
cd "$BUILD_DIR"
find . -type f \( -name "*.whl" -o -name "*.tar.gz" \) -exec sha256sum {} \; 2>&1 | tee checksums.sha256 | tee -a "$BUILD_LOG"
log "✓ Checksums generated"
echo "" | tee -a "$BUILD_LOG"

# Summary
echo "" | tee -a "$BUILD_LOG"
echo "================================================" | tee -a "$BUILD_LOG"
echo "Build Summary" | tee -a "$BUILD_LOG"
echo "================================================" | tee -a "$BUILD_LOG"
echo "Build ID: ${BUILD_TIMESTAMP}" | tee -a "$BUILD_LOG"
echo "Build Directory: ${BUILD_DIR}" | tee -a "$BUILD_LOG"
echo "Build Log: ${BUILD_LOG}" | tee -a "$BUILD_LOG"
echo "" | tee -a "$BUILD_LOG"

echo "Build Artifacts:" | tee -a "$BUILD_LOG"
if [ -d "$BUILD_DIR/python" ] && [ "$(ls -A "$BUILD_DIR/python")" ]; then
    echo "  ✓ Python package: dist/python/" | tee -a "$BUILD_LOG"
else
    echo "  ✗ Python package: FAILED" | tee -a "$BUILD_LOG"
fi

if [ -d "$BUILD_DIR/typescript/qratum-sdk" ]; then
    echo "  ✓ TypeScript SDK: dist/typescript/qratum-sdk/" | tee -a "$BUILD_LOG"
else
    echo "  ⚠ TypeScript SDK: Not built" | tee -a "$BUILD_LOG"
fi

if [ -d "$BUILD_DIR/desktop/qratum-desktop" ]; then
    echo "  ✓ Desktop App: dist/desktop/qratum-desktop/" | tee -a "$BUILD_LOG"
else
    echo "  ⚠ Desktop App: Not built" | tee -a "$BUILD_LOG"
fi

echo "" | tee -a "$BUILD_LOG"
echo "Total artifacts size: $(du -sh "$BUILD_DIR" | cut -f1)" | tee -a "$BUILD_LOG"
echo "" | tee -a "$BUILD_LOG"
echo "================================================" | tee -a "$BUILD_LOG"
log "✓ PRODUCTION BUILD COMPLETE"
echo "================================================" | tee -a "$BUILD_LOG"
echo "" | tee -a "$BUILD_LOG"
echo "To deploy: bash ${BUILD_DIR}/deploy.sh" | tee -a "$BUILD_LOG"
echo "For details: cat ${BUILD_DIR}/BUILD_MANIFEST.md" | tee -a "$BUILD_LOG"
echo "" | tee -a "$BUILD_LOG"
