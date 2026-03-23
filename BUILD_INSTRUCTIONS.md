# QRATUM Production Build Instructions

This document describes how to generate the full production build for QRATUM.

## Overview

The QRATUM production build system generates distributable artifacts for:

- Python Package (wheel and source distribution)
- TypeScript SDK
- Electron Desktop Application
- Documentation bundle
- Deployment scripts

## Prerequisites

### System Requirements

- Python 3.10 or higher
- Node.js and npm (for TypeScript SDK and Desktop app)
- Git

### Python Dependencies

- setuptools
- wheel
- build

### Node.js Dependencies

- TypeScript (for SDK)
- Electron and electron-builder (for Desktop app)

## Quick Start

Run the production build script:

```bash
bash build_production.sh
```

This will:

1. Validate Python version (requires 3.10+)
2. Create `dist/` directory structure
3. Build Python package (wheel + sdist)
4. Build TypeScript SDK
5. Build Electron Desktop application
6. Copy deployment scripts and documentation
7. Generate checksums and build manifest

## Build Output

After a successful build, the `dist/` directory will contain:

```
dist/
├── BUILD_MANIFEST.md          # Build details and metadata
├── Makefile                   # Make targets for deployment
├── checksums.sha256           # SHA256 checksums of artifacts
├── deploy.sh                  # Production deployment script
├── python/
│   ├── qratum-2.0.0-py3-none-any.whl
│   └── qratum-2.0.0.tar.gz
├── typescript/
│   └── qratum-sdk/           # Compiled TypeScript SDK
├── desktop/
│   └── qratum-desktop/       # Electron app (platform-specific)
├── docs/                      # Documentation bundle
└── logs/                      # Build logs
```

## Installation

### Python Package

```bash
pip install dist/python/qratum-*.whl
```

### TypeScript SDK

```bash
cd dist/typescript/qratum-sdk
npm install
npm link  # For local development
```

### Desktop Application

The desktop application is packaged in `dist/desktop/qratum-desktop/`.

- For Linux: See `linux-unpacked/` directory
- For Windows/Mac: Additional platform-specific builds required

## Platform-Specific Builds

### Windows

To build Windows installers, run on a Windows machine:

```bash
cd qratum_desktop
npm install
npm run build:win
```

### macOS

To build macOS DMG, run on a Mac:

```bash
cd qratum_desktop
npm install
npm run build:mac
```

### Linux

Linux AppImage/deb/rpm are built automatically with:

```bash
cd qratum_desktop
npm install
npm run build:linux
```

## CI/CD Integration

The build script is designed to work in CI/CD environments:

```yaml
- name: Build Production Artifacts
  run: bash build_production.sh
  
- name: Upload Artifacts
  uses: actions/upload-artifact@v4
  with:
    name: qratum-production-build
    path: dist/
```

## Verification

After building, verify the artifacts:

```bash
# Check Python package
pip install dist/python/qratum-*.whl
python -c "import qratum; print(qratum.__version__)"

# Check TypeScript SDK
cd dist/typescript/qratum-sdk
node -e "console.log(require('./index.js'))"

# Verify checksums
cd dist
sha256sum -c checksums.sha256
```

## Deployment

For production deployment, see:

- `dist/deploy.sh` - Automated deployment script
- `dist/docs/PRODUCTION_RELEASE_MANIFEST.md` - Release documentation
- `dist/BUILD_MANIFEST.md` - Build-specific details

## Troubleshooting

### Build Fails on Python Package

- Ensure Python 3.10+ is installed: `python3 --version`
- Install build dependencies: `pip install --upgrade setuptools wheel build`

### TypeScript Build Errors

- Install dependencies: `cd sdk/typescript && npm install`
- Check TypeScript version: `npm list typescript`
- Some TypeScript errors are warnings and don't prevent artifact generation

### Electron Build Issues

- Electron builds require platform-specific tools
- On Linux, some Windows/Mac features may not be available
- Desktop app source is still copied even if full build fails

### Large Build Size

- The desktop application includes the full Electron runtime (~250MB)
- For distribution, consider using electron-builder's compression
- Python wheels are typically 1-2 MB

## Build Logs

All build output is saved to:

- `build_YYYYMMDD_HHMMSS.log` - Main build log
- `dist/logs/` - Copy of build logs for reference

## Clean Build

To perform a clean build:

```bash
# Remove previous artifacts
rm -rf dist/ build/ *.egg-info/

# Run production build
bash build_production.sh
```

## Support

For issues with the build system:

- Check `dist/logs/build_*.log` for detailed error messages
- Review `dist/BUILD_MANIFEST.md` for build configuration
- See repository documentation in `docs/`

## Version Information

- Build Script Version: 1.0.0
- QRATUM Version: 2.0.0
- Python Target: 3.10+
- Node.js Target: 16+
- Electron Version: 28+

---
Generated: December 2025
