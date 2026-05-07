#!/usr/bin/env bash
# P0.6 — validate_receipts.sh — runs receipt-related host tests.
set -euo pipefail
cd "$(dirname "$0")/../.."
cargo test -p qratum-arbiter \
  --test receipt_merkle_integrity \
  --test receipt_anchor_integrity \
  --test cryptographic_anchor_validation \
  --test deterministic_receipt_chain
