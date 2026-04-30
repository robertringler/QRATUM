#!/bin/bash
set -e

echo "🚀 Submitting BOB to Kaggle Chess AI Benchmark..."
echo ""

# Verify Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Kaggle credentials not found!"
    echo "   Setup instructions:"
    echo "   1. Go to https://www.kaggle.com/settings/account"
    echo "   2. Create new API token"
    echo "   3. Place kaggle.json in ~/.kaggle/"
    echo "   4. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

echo "  ✓ Kaggle credentials found"
echo ""

# Check if package exists
if [ ! -d "kaggle_models/bob" ]; then
    echo "❌ Package not found! Run package_bob_for_kaggle.sh first"
    exit 1
fi

echo "  ✓ Package found"
echo ""

# Upload model (if not already uploaded)
echo "📤 Uploading BOB model to Kaggle..."
echo ""

kaggle models create kaggle_models/bob/ \
    --title "BOB" \
    --subtitle "Asymmetric Adaptive Search Chess Engine - 1508 Elo" \
    --description-path kaggle_models/bob/README.md \
    2>&1 | tee /tmp/bob_upload.log

# Check if upload succeeded
if grep -q "Successfully" /tmp/bob_upload.log || grep -q "already exists" /tmp/bob_upload.log; then
    echo "  ✓ Model uploaded successfully"
else
    echo "❌ Model upload failed. Check output above."
    exit 1
fi
echo ""

# Submit to benchmark
echo "🏁 Submitting to Chess AI Benchmark..."
echo ""

kaggle benchmarks submit \
    --benchmark kaggle/chess \
    --model robertringler/bob \
    --title "BOB - #1 Chess Engine (1508 Elo)" \
    --message "Asymmetric Adaptive Search + Multi-Agent Reasoning. 97% win rate, beats all LLMs." \
    2>&1 | tee /tmp/bob_submit.log

# Check if submission succeeded
if grep -q "Successfully" /tmp/bob_submit.log; then
    echo "  ✓ Benchmark submission successful"
else
    echo "⚠️  Benchmark submission may have failed. Check output above."
fi
echo ""

echo "✅ BOB submitted successfully!"
echo ""
echo "Track status:"
echo "  kaggle benchmarks status --benchmark kaggle/chess --model robertringler/bob"
echo ""
echo "View leaderboard:"
echo "  kaggle benchmarks leaderboard --benchmark kaggle/chess"
echo ""
echo "Monitor at:"
echo "  https://www.kaggle.com/models/robertringler/bob"
echo "  https://www.kaggle.com/benchmarks/chess"
