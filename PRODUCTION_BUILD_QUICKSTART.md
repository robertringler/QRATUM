# QRATUM Production Build - Quick Start

🚀 **Generate the full production build in one command:**

```bash
bash build_production.sh
```

## What Gets Built

The build script generates a complete production-ready distribution:

| Component | Size | Description |
|-----------|------|-------------|
| Python Package | 1.7 MB | Wheel + Source distribution |
| TypeScript SDK | 56 KB | Compiled JS + Type definitions |
| Desktop App | 255 MB | Electron application (Linux) |
| **Total** | **257 MB** | Complete distribution |

## Output Structure

```
dist/
├── python/qratum-2.0.0-py3-none-any.whl     ← Install this
├── typescript/qratum-sdk/                    ← Compiled SDK
├── desktop/qratum-desktop/                   ← Desktop app
├── BUILD_MANIFEST.md                         ← Build details
└── checksums.sha256                          ← Verify integrity
```

## Install Python Package

```bash
pip install dist/python/qratum-*.whl
```

## Run Build in CI/CD

```yaml
- run: bash build_production.sh
- uses: actions/upload-artifact@v4
  with:
    name: qratum-build
    path: dist/
```

## Requirements

- Python 3.10+
- Node.js 16+ (for SDK and Desktop)
- npm

## Documentation

- 📖 **[BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)** - Complete guide
- 📋 **[PRODUCTION_BUILD_SUMMARY.md](PRODUCTION_BUILD_SUMMARY.md)** - Implementation details
- 🔧 **[build_production.sh](build_production.sh)** - Build script source

## Troubleshooting

**Build fails?**

```bash
# Check Python version
python3 --version  # Should be 3.10+

# Clean and retry
rm -rf dist/ build/
bash build_production.sh
```

**Need help?** See BUILD_INSTRUCTIONS.md for detailed troubleshooting.

---
✅ Production Build System v1.0.0
