#!/usr/bin/env bash
set -euo pipefail

export PYTHON_VERSION=3.12
export ORD_INTERVAL_SEC=${ORD_INTERVAL_SEC:-30}
export ORD_CHECKPOINT_HOURS=${ORD_CHECKPOINT_HOURS:-6}
export QMP_BASE_PRICE_USD=${QMP_BASE_PRICE_USD:-0.0004}

pip install numpy scipy scikit-learn zstandard
python tools/phi_qevf/phi_qevf.py
python tools/phi_qevf/anomaly_stack.py
QMP_BASE_PRICE_USD=${QMP_BASE_PRICE_USD} python tools/qmp_sandbox/qmp.py > out/qmp_stream.json
ORD_INTERVAL_SEC=${ORD_INTERVAL_SEC} ORD_CHECKPOINT_HOURS=${ORD_CHECKPOINT_HOURS} python infra/ord_pipeline/writer.py
# Quick validation
python -c "
import json
s = json.load(open('out/phi_qevf_summary.json'))
print(f'KS: {s[\"ks_confidence_pct\"]:.1f}%, Pass: {s[\"pass\"]}')
"
echo "Local run complete. Check out/ for outputs."
