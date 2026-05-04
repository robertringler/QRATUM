#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $0 [--preflight] [--metrics-url <url>]

Checks ORD SLOs using Prometheus instant queries.
USAGE
}

METRICS_URL="http://localhost:9090"
PREFLIGHT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preflight)
      PREFLIGHT=true
      shift
      ;;
    --metrics-url)
      METRICS_URL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

function query_prom() {
  local query="$1"
  curl -fsSL "${METRICS_URL}/api/v1/query" --get --data-urlencode "query=${query}"
}

if [[ "${PREFLIGHT}" == true ]]; then
  command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
  command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }
  echo "Preflight checks passed."
  exit 0
fi

QEVF_QUERY='avg_over_time(ord_qevf_ops_per_kwh[4h])'
FIDELITY_QUERY='quantile_over_time(0.95, ord_fidelity[1h])'
ENERGY_QUERY='stddev_over_time(ord_energy_kwh[1h])'

QEVF=$(query_prom "${QEVF_QUERY}" | jq -r '.data.result[0].value[1]' || echo 0)
FIDELITY=$(query_prom "${FIDELITY_QUERY}" | jq -r '.data.result[0].value[1]' || echo 0)
ENERGY=$(query_prom "${ENERGY_QUERY}" | jq -r '.data.result[0].value[1]' || echo 1)

SLO_FAIL=0

awk "BEGIN {exit !(${QEVF} >= 1.0e17)}" || { echo "Φ_QEVF SLO breached: ${QEVF}"; SLO_FAIL=1; }
awk "BEGIN {exit !(${FIDELITY} >= 0.997)}" || { echo "Fidelity SLO breached: ${FIDELITY}"; SLO_FAIL=1; }
awk "BEGIN {exit !(${ENERGY} <= 0.05)}" || { echo "Energy variance SLO breached: ${ENERGY}"; SLO_FAIL=1; }

if [[ ${SLO_FAIL} -ne 0 ]]; then
  echo "ORD SLO verification failed."
  exit 1
fi

echo "All ORD SLOs satisfied."
