#!/usr/bin/env bash
# qratum-boot.sh — boot QRATUM as an OS-style stack on Linux/macOS.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAEMON="$REPO_ROOT/qratum_kernel/qratum_kernel_daemon.py"
case "$OSTYPE" in
    darwin*) STATE_DIR="$HOME/Library/Application Support/QRATUM/bridge" ;;
    *)       STATE_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/QRATUM/bridge" ;;
esac
PIDFILE="$STATE_DIR/kernel.pid"

post() {
    echo "[QRATUM] POST  ..."
    command -v python3 >/dev/null 2>&1 || { echo "[QRATUM] POST FAIL: python3 missing"; exit 1; }
    [[ -f "$DAEMON" ]] || { echo "[QRATUM] POST FAIL: $DAEMON missing"; exit 1; }
    mkdir -p "$STATE_DIR"
    echo "[QRATUM] POST  OK"
}

start_kernel() {
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "[QRATUM] kernel already running pid=$(cat "$PIDFILE")"
        return
    fi
    echo "[QRATUM] INIT  spawning kernel daemon ..."
    nohup python3 "$DAEMON" --hz 60 --quiet >/dev/null 2>&1 &
    echo $! > "$PIDFILE"
    for _ in $(seq 1 50); do
        [[ -f "$STATE_DIR/state.json" ]] && break
        sleep 0.1
    done
    echo "[QRATUM] INIT  kernel pid=$(cat "$PIDFILE") publishing to $STATE_DIR/state.json"
}

halt_kernel() {
    if [[ -f "$PIDFILE" ]]; then
        PID=$(cat "$PIDFILE")
        kill "$PID" 2>/dev/null || true
        rm -f "$PIDFILE"
        echo "[QRATUM] halted pid=$PID"
    else
        echo "[QRATUM] no kernel running"
    fi
}

case "${1:-boot}" in
    boot)  post; start_kernel ;;
    halt)  halt_kernel ;;
    status)
        if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "running pid=$(cat "$PIDFILE")"
        else
            echo "offline"
        fi
        ;;
    *) echo "usage: $0 {boot|halt|status}"; exit 2 ;;
esac
