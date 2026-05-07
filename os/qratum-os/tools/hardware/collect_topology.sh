#!/usr/bin/env bash
# P0.6 — collect_topology.sh — emit canonical topology JSON.
set -euo pipefail
if command -v lscpu >/dev/null 2>&1; then
  n=$(nproc)
  echo "{\"status\":\"measured\",\"n_cores\":$n,\"cores\":[]}"
else
  echo '{"status":"deferred","n_cores":0,"cores":[]}'
fi
