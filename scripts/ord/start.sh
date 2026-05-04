#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $0 [--duration <duration>] [--stop] [--port <port>] [--interval <seconds>]

Options:
  --duration <duration>   Duration for telemetry run (default: 72h)
  --interval <seconds>    Collection interval in seconds (default: 15)
  --port <port>           Metrics port (default: 9000)
  --stop                  Signal an existing agent to stop via PID file
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
PID_FILE="${ROOT_DIR}/runs/ord-agent.pid"
LOG_DIR="${ROOT_DIR}/runs"
AGENT="${ROOT_DIR}/services/telemetry/agent.py"
DURATION="72h"
INTERVAL="15"
PORT="9000"

mkdir -p "${LOG_DIR}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --stop)
      if [[ -f "${PID_FILE}" ]]; then
        kill "$(cat "${PID_FILE}")" && rm -f "${PID_FILE}"
        echo "Stopped existing telemetry agent."
      else
        echo "No telemetry agent PID file found." >&2
        exit 1
      fi
      exit 0
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

if [[ ! -f "${AGENT}" ]]; then
  echo "Telemetry agent not found at ${AGENT}" >&2
  exit 1
fi

python3 "${AGENT}" \
  --duration "${DURATION}" \
  --collection-interval "${INTERVAL}" \
  --port "${PORT}" \
  > "${LOG_DIR}/ord-agent.log" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"
echo "Telemetry agent started with PID ${PID} for duration ${DURATION} at port ${PORT}."
