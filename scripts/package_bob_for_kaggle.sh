#!/bin/bash
set -e

echo "📦 Packaging BOB for Kaggle submission..."
echo ""

# Check we're in the right directory
if [ ! -d "kaggle_models/bob" ]; then
    echo "❌ Error: Must run from QRATUM repository root"
    exit 1
fi

cd kaggle_models/bob

# Run validation tests
echo "  ✓ Running validation tests..."
python tests/test_prediction.py
if [ $? -ne 0 ]; then
    echo "❌ Tests failed! Fix issues before packaging."
    exit 1
fi
echo ""

# Check package size
echo "  ✓ Checking package size..."
SIZE=$(du -sb . | cut -f1)
SIZE_MB=$((SIZE / 1024 / 1024))
echo "    Package size: ${SIZE_MB} MB"

if [ $SIZE_MB -gt 500 ]; then
    echo "❌ Package too large! Must be < 500MB (Kaggle limit)"
    exit 1
fi
echo ""

# Verify required files exist
echo "  ✓ Verifying required files..."
REQUIRED_FILES=(
    "model-metadata.json"
    "predict.py"
    "requirements.txt"
    "README.md"
    "engine/__init__.py"
    "engine/bob_engine.py"
    "tests/test_prediction.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing required file: $file"
        exit 1
    fi
    echo "    ✓ $file"
done
echo ""

# Generate updated metadata
echo "  ✓ Generating metadata..."
python ../../scripts/generate_bob_metadata.py > model-metadata.json
echo ""

# Clean up cache files before packaging
echo "  ✓ Cleaning cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo ""

# Package for upload
echo "  ✓ Creating archive..."
cd ..
tar -czf bob-chess-engine.tar.gz --exclude="__pycache__" --exclude="*.pyc" bob/
ARCHIVE_SIZE=$(du -h bob-chess-engine.tar.gz | cut -f1)
echo "    Archive size: ${ARCHIVE_SIZE}"
echo ""

echo "✅ BOB packaged successfully!"
echo "📁 Package: kaggle_models/bob-chess-engine.tar.gz"
echo ""
echo "Next steps:"
echo "  1. Review package contents:"
echo "     tar -tzf kaggle_models/bob-chess-engine.tar.gz | head -20"
echo ""
echo "  2. Submit to Kaggle (requires Kaggle API setup):"
echo "     kaggle models create kaggle_models/bob/"
echo ""
echo "  3. Submit to benchmark:"
echo "     kaggle benchmarks submit --benchmark kaggle/chess --model robertringler/bob"
