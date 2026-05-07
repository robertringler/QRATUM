# P0.6 — validate_receipts.ps1 — runs receipt-related host tests.
$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '..\..\')
try {
  cargo test -p qratum-arbiter --target x86_64-pc-windows-msvc `
    --test receipt_merkle_integrity `
    --test receipt_anchor_integrity `
    --test cryptographic_anchor_validation `
    --test deterministic_receipt_chain
} finally { Pop-Location }
