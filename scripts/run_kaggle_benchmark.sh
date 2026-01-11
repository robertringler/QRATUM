#!/bin/bash
# QRATUM Chess - Kaggle Leaderboard Benchmark Wrapper
# 
# This script downloads Kaggle chess leaderboard data and runs QRATUM engine benchmarks.
# Optionally submits results back to Kaggle leaderboard.
#
# Usage:
#   ./scripts/run_kaggle_benchmark.sh                    # Run benchmark only
#   ./scripts/run_kaggle_benchmark.sh --submit           # Run and submit to leaderboard
#   ./scripts/run_kaggle_benchmark.sh --submit --message "QRATUM v1.0"
# run_kaggle_benchmark.sh
#
# QRATUM-Chess Kaggle Competition Pipeline
#
# Fully automated pipeline for Kaggle chess competition domination:
# 1. Downloads real Kaggle competition data
# 2. Runs actual QRATUM engine (no mocks)
# 3. Produces compliant Kaggle submissions
# 4. Submits automatically
# 5. Tracks ranking, ELO, novelty motifs, and performance drift
# 6. Iterates hyperparameters based on leaderboard feedback
#
# Usage:
#   ./scripts/run_kaggle_benchmark.sh --submit --optimize --track --no-mock
#   ./scripts/run_kaggle_benchmark.sh --competition chess-competition
#   ./scripts/run_kaggle_benchmark.sh --quick

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPETITION_ID="chess-engine-leaderboard"
DATA_FILE="kaggle_leaderboard_data.json"
RESULTS_FILE="kaggle_benchmark_results.json"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse arguments
SUBMIT_FLAG=""
MESSAGE_FLAG=""
DRY_RUN_FLAG=""
USE_SAMPLE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --submit|-s)
            SUBMIT_FLAG="--submit"
            shift
            ;;
        --message|-m)
            MESSAGE_FLAG="--message \"$2\""
            shift 2
            ;;
        --dry-run)
            DRY_RUN_FLAG="--dry-run"
            shift
            ;;
        --sample)
            USE_SAMPLE=true
# Default configuration
COMPETITION=""
DEPTH=15
TIME_LIMIT=""
MAX_POSITIONS=""
DATA_DIR="data/kaggle/current"
OUTPUT_DIR="submissions"
QUICK_MODE=false
SUBMIT=false
OPTIMIZE=false
TRACK=false
NO_MOCK=false
ENABLE_TRIMODAL=false
ENABLE_NOVELTY=false
NOVELTY_PRESSURE=0.5

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --competition|-c)
            COMPETITION="$2"
            shift 2
            ;;
        --depth)
            DEPTH="$2"
            shift 2
            ;;
        --time-limit)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --max-positions)
            MAX_POSITIONS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --submit)
            SUBMIT=true
            shift
            ;;
        --optimize)
            OPTIMIZE=true
            shift
            ;;
        --track)
            TRACK=true
            shift
            ;;
        --no-mock)
            NO_MOCK=true
            shift
            ;;
        --enable-trimodal)
            ENABLE_TRIMODAL=true
            shift
            ;;
        --enable-novelty-pressure)
            ENABLE_NOVELTY=true
            shift
            ;;
        --novelty-pressure)
            NOVELTY_PRESSURE="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            DEPTH=10
            MAX_POSITIONS=20
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --submit, -s              Submit results to Kaggle leaderboard"
            echo "  --message, -m MESSAGE     Submission message/description"
            echo "  --dry-run                 Validate submission without posting"
            echo "  --sample                  Use sample positions (no download)"
            echo "  --help, -h                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # Run benchmark only"
            echo "  $0 --submit                     # Run and submit"
            echo "  $0 --submit --message \"v1.0\"    # Run and submit with message"
            echo "  $0 --sample                     # Use sample positions"
            echo "  --competition, -c ID   Kaggle competition ID"
            echo "  --depth N              Search depth (default: 15)"
            echo "  --time-limit MS        Time limit per position in ms"
            echo "  --max-positions N      Maximum positions to test"
            echo "  --output-dir DIR       Output directory (default: submissions)"
            echo "  --submit               Submit to Kaggle after generation"
            echo "  --optimize             Enable hyperparameter optimization"
            echo "  --track                Track leaderboard ranking"
            echo "  --no-mock              Enforce no mock data (real competition only)"
            echo "  --enable-trimodal      Enable tri-modal cortex fusion"
            echo "  --enable-novelty-pressure  Enable novelty pressure"
            echo "  --novelty-pressure N   Novelty pressure value (0.0-1.0, default: 0.5)"
            echo "  --quick                Quick mode (depth=10, max=20)"
            echo "  --help, -h             Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                              ║"
echo "║   🤖 QRATUM Chess - Kaggle Benchmark Wrapper                                 ║"
# Print banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                              ║"
echo "║   🏆 QRATUM-Chess Kaggle Competition Pipeline                                ║"
echo "║                                                                              ║"
echo "║   Objective: Sustained Leaderboard Supremacy                                 ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

cd "$REPO_ROOT"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3 found${NC}"

# Check for Kaggle credentials if submitting
if [ -n "$SUBMIT_FLAG" ]; then
    if [ -f "$HOME/.kaggle/kaggle.json" ]; then
        echo -e "${GREEN}✓ Kaggle credentials found at ~/.kaggle/kaggle.json${NC}"
    elif [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
        echo -e "${GREEN}✓ Kaggle credentials found in environment variables${NC}"
    else
        echo -e "${RED}✗ Kaggle credentials not found${NC}"
        echo ""
        echo "To submit results, you need Kaggle API credentials:"
        echo "1. Go to https://www.kaggle.com/settings"
        echo "2. Click 'Create New API Token'"
        echo "3. Save kaggle.json to ~/.kaggle/kaggle.json"
        echo ""
        echo "Or set environment variables:"
        echo "  export KAGGLE_USERNAME=your_username"
        echo "  export KAGGLE_KEY=your_api_key"
        exit 1
    fi
fi

# Download Kaggle leaderboard data (if not using sample)
if [ "$USE_SAMPLE" = false ]; then
    echo ""
    echo -e "${YELLOW}⬇ Attempting to download Kaggle leaderboard data...${NC}"
    
    # Try using kaggle CLI if available
    if command -v kaggle &> /dev/null; then
        echo "Using kaggle CLI..."
        if kaggle competitions leaderboard "$COMPETITION_ID" --show > "$DATA_FILE" 2>&1; then
            echo -e "${GREEN}✓ Downloaded leaderboard data via kaggle CLI${NC}"
        else
            echo -e "${YELLOW}⚠ Could not download via kaggle CLI, will use Python API${NC}"
            rm -f "$DATA_FILE"
        fi
    else
        echo "kaggle CLI not found, will use Python API"
    fi
    
    # If download failed or no CLI, use Python API download
    if [ ! -f "$DATA_FILE" ]; then
        echo "Will download via Python API during benchmark execution"
    fi
fi

# Run QRATUM Chess benchmark
echo ""
echo -e "${YELLOW}🚀 Running QRATUM Chess benchmark...${NC}"
echo ""

PYTHON_CMD="python3 qratum_chess/benchmarks/benchmark_kaggle.py"

if [ "$USE_SAMPLE" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --use-sample"
elif [ -f "$DATA_FILE" ]; then
    PYTHON_CMD="$PYTHON_CMD --input $DATA_FILE"
else
    PYTHON_CMD="$PYTHON_CMD --download"
fi

if [ -n "$SUBMIT_FLAG" ]; then
    PYTHON_CMD="$PYTHON_CMD $SUBMIT_FLAG"
fi

if [ -n "$MESSAGE_FLAG" ]; then
    PYTHON_CMD="$PYTHON_CMD $MESSAGE_FLAG"
fi

if [ -n "$DRY_RUN_FLAG" ]; then
    PYTHON_CMD="$PYTHON_CMD $DRY_RUN_FLAG"
fi

PYTHON_CMD="$PYTHON_CMD --output $RESULTS_FILE"

# Execute benchmark
if eval $PYTHON_CMD; then
    echo ""
    echo -e "${GREEN}✓ Benchmark completed successfully${NC}"
    
    # Show results summary
    if [ -f "$RESULTS_FILE" ]; then
        echo ""
        echo -e "${BLUE}Results saved to: $RESULTS_FILE${NC}"
        
        # Extract summary from JSON
        if command -v jq &> /dev/null; then
            echo ""
            echo "Quick Summary:"
            jq -r '.metadata | "  Engine: \(.engine)\n  Positions: \(.total_positions)\n  Depth: \(.depth)\n  Time limit: \(.time_limit_ms) ms"' "$RESULTS_FILE"
        fi
    fi
    
    exit 0
else
    echo ""
    echo -e "${RED}✗ Benchmark failed${NC}"
# Verify Kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo -e "${RED}✗ Error: Kaggle CLI not found. Install with: pip install kaggle${NC}"
    exit 1
fi

# Check for Kaggle credentials
if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ -z "$KAGGLE_USERNAME" ]; then
    echo -e "${RED}✗ Error: Kaggle credentials not found.${NC}"
    echo "Please either:"
    echo "  1. Create ~/.kaggle/kaggle.json with your API credentials"
    echo "  2. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables"
    echo "Get your API key from: https://www.kaggle.com/settings/account"
    exit 1
fi

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"

# Build Python command
PYTHON_CMD="python3 run_qratum_chess_kaggle.py"

# Add competition if specified
if [ -n "$COMPETITION" ]; then
    PYTHON_CMD="$PYTHON_CMD --competition $COMPETITION"
fi

# Add depth
PYTHON_CMD="$PYTHON_CMD --depth $DEPTH"

# Add time limit if specified
if [ -n "$TIME_LIMIT" ]; then
    PYTHON_CMD="$PYTHON_CMD --time-limit $TIME_LIMIT"
fi

# Add max positions if specified
if [ -n "$MAX_POSITIONS" ]; then
    PYTHON_CMD="$PYTHON_CMD --max-positions $MAX_POSITIONS"
fi

# Add output directory
PYTHON_CMD="$PYTHON_CMD --output-dir $OUTPUT_DIR"

# Add flags
if [ "$SUBMIT" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --submit"
fi

if [ "$OPTIMIZE" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --optimize"
fi

if [ "$TRACK" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --track"
fi

if [ "$NO_MOCK" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --no-mock"
fi

if [ "$ENABLE_TRIMODAL" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --enable-trimodal"
fi

if [ "$ENABLE_NOVELTY" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --enable-novelty-pressure --novelty-pressure $NOVELTY_PRESSURE"
fi

# Add disable randomness for reproducibility
PYTHON_CMD="$PYTHON_CMD --disable-randomness"

# Print configuration
echo -e "${YELLOW}Configuration:${NC}"
if [ -n "$COMPETITION" ]; then
    echo "  Competition: $COMPETITION"
fi
echo "  Search depth: $DEPTH"
if [ -n "$TIME_LIMIT" ]; then
    echo "  Time limit: ${TIME_LIMIT}ms"
fi
if [ -n "$MAX_POSITIONS" ]; then
    echo "  Max positions: $MAX_POSITIONS"
fi
echo "  Submit: $SUBMIT"
echo "  Optimize: $OPTIMIZE"
echo "  Track: $TRACK"
if [ "$ENABLE_NOVELTY" = true ]; then
    echo "  Novelty pressure: $NOVELTY_PRESSURE"
fi
echo ""

# Execute pipeline
echo -e "${YELLOW}Running QRATUM-Chess Kaggle Pipeline...${NC}"
echo ""

if eval "$PYTHON_CMD"; then
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${GREEN}✓ Pipeline completed successfully${NC}                                           ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    exit 0
else
    echo ""
    echo -e "${RED}✗ Pipeline failed${NC}"
    exit 1
fi
