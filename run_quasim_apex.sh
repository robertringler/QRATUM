#!/bin/bash
# QuASIM APEX Mode Runner
# Wrapper script to execute QuNimbus IP and Supercomputer synthesis in APEX mode

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_FILE=".github/copilot-tasks/qunimbus_ip_and_supercomputer.yaml"

echo "Running QuASIM in APEX Mode..."
echo "Task: QuNimbus IP Miner + Supercomputer Synthesis"
echo ""

python3 "${SCRIPT_DIR}/scripts/run_apex_task.py" \
    "${TASK_FILE}" \
    --enhance \
    --level=apex

echo ""
echo "✅ APEX run complete. Check docs/ip/ and docs/supercomputer/ for artifacts."
