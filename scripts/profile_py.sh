#!/bin/bash
# Python profiling script for QuASIM kernels
# Supports cProfile, line_profiler, and memory_profiler

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PROFILE_DIR="$REPO_ROOT/profiles/python"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    cat <<EOF
Usage: $0 [OPTIONS] KERNEL_ID

Profile Python compute kernels in the QuASIM repository.

OPTIONS:
    -m, --mode MODE        Profiling mode: cprofile, line, memory (default: cprofile)
    -o, --output FILE      Output file (default: profiles/python/KERNEL_ID_MODE.txt)
    -h, --help            Show this help message

KERNEL_ID:
    k001                   Tensor Contraction (runtime/python/quasim/runtime.py)
    k002                   Tensor Generation (benchmarks/quasim_bench.py)
    k003                   Columnar Sum (sdk/rapids/dataframe.py)

EXAMPLES:
    # Profile tensor contraction with cProfile
    $0 k001

    # Profile with line-level profiling
    $0 --mode line k001

    # Profile memory usage
    $0 --mode memory k003

EOF
    exit 1
}

# Default values
MODE="cprofile"
OUTPUT=""
KERNEL_ID=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            KERNEL_ID="$1"
            shift
            ;;
    esac
done

if [ -z "$KERNEL_ID" ]; then
    echo -e "${RED}Error: KERNEL_ID required${NC}"
    usage
fi

# Set default output if not specified
if [ -z "$OUTPUT" ]; then
    OUTPUT="$PROFILE_DIR/${KERNEL_ID}_${MODE}.txt"
fi

mkdir -p "$(dirname "$OUTPUT")"

echo -e "${GREEN}QuASIM Python Profiler${NC}"
echo "Kernel: $KERNEL_ID"
echo "Mode: $MODE"
echo "Output: $OUTPUT"
echo ""

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/runtime/python:$REPO_ROOT/quantum:$REPO_ROOT/sdk"

case $KERNEL_ID in
    k001)
        BENCHMARK_SCRIPT="benchmarks/harnesses/bench_k001_tensor_contraction.py"
        ;;
    k002)
        BENCHMARK_SCRIPT="benchmarks/harnesses/bench_k002_tensor_generation.py"
        ;;
    k003)
        BENCHMARK_SCRIPT="benchmarks/harnesses/bench_k003_columnar_sum.py"
        ;;
    *)
        echo -e "${RED}Error: Unknown kernel ID: $KERNEL_ID${NC}"
        exit 1
        ;;
esac

case $MODE in
    cprofile)
        echo -e "${YELLOW}Running cProfile...${NC}"
        python3 -m cProfile -s cumtime "$BENCHMARK_SCRIPT" 2>&1 | tee "$OUTPUT"
        ;;
    line)
        echo -e "${YELLOW}Line-level profiling requires kernprof${NC}"
        echo "Install with: pip install line_profiler"
        echo "Then add @profile decorator to target functions"
        ;;
    memory)
        echo -e "${YELLOW}Memory profiling requires memory_profiler${NC}"
        echo "Install with: pip install memory_profiler"
        echo "Then run: python3 -m memory_profiler $BENCHMARK_SCRIPT"
        ;;
    *)
        echo -e "${RED}Error: Unknown mode: $MODE${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}Profile saved to: $OUTPUT${NC}"
echo ""
echo "Summary:"
python3 - <<EOF
import sys
if "$MODE" == "cprofile":
    print("Top time-consuming functions:")
    print("  (see $OUTPUT for full details)")
else:
    print("  Mode '$MODE' requires additional setup - see instructions above")
EOF
