#!/usr/bin/env bash
# P0.4 — Reproducibility bootstrap (Linux / macOS).
set -euo pipefail
cd "$(dirname "$0")/../.."
rustup toolchain install nightly-2025-04-01 --component rust-src
rustup component add rust-src --toolchain nightly-2025-04-01
rustup show active-toolchain
cargo --version
rustc --version
echo "P0.4 bootstrap: OK"
