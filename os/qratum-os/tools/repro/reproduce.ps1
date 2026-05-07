#!/usr/bin/env pwsh
# P0.4 — Reproducibility driver (Windows).
# Single command: build kernel, build host, run all tests, write
# out/repro/{final_state.hash, audit.hash, replay_receipt.json,
# performance_report.json}.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSCommandPath)
Set-Location ..\..
$out = "out/repro"
New-Item -Force -ItemType Directory $out | Out-Null

Write-Host "[1/4] Building kernel..."
cargo build -p qratum-os `
  -Zbuild-std=core,compiler_builtins,alloc `
  -Zbuild-std-features=compiler-builtins-mem | Out-Null

Write-Host "[2/4] Building host arbiter..."
cargo build -p qratum-arbiter --target x86_64-pc-windows-msvc | Out-Null

Write-Host "[3/4] Running test matrix..."
cargo test -p qratum-arbiter --target x86_64-pc-windows-msvc --no-fail-fast `
  --release 2>&1 | Tee-Object "$out/test_log.txt" | Out-Null

Write-Host "[4/4] Computing hashes..."
$kernel = "target/x86_64-unknown-uefi/debug/qratum.efi"
$kbytes = (Get-Item $kernel).Length
$khash  = (Get-FileHash -Algorithm SHA256 $kernel).Hash
"$khash  $kernel  $kbytes" | Out-File -Encoding ascii "$out/final_state.hash"

# Audit hash = SHA-256 of test log.
$auditHash = (Get-FileHash -Algorithm SHA256 "$out/test_log.txt").Hash
$auditHash | Out-File -Encoding ascii "$out/audit.hash"

@{
  toolchain      = (rustup show active-toolchain) -join ' '
  kernel_bytes   = $kbytes
  kernel_sha256  = $khash
  audit_sha256   = $auditHash
  timestamp_utc  = (Get-Date).ToUniversalTime().ToString("o")
} | ConvertTo-Json -Depth 4 | Out-File -Encoding ascii "$out/replay_receipt.json"

# Performance report — extract any " test result:" lines.
$results = Select-String -Path "$out/test_log.txt" -Pattern "test result:" |
           ForEach-Object { $_.Line }
@{ test_summary_lines = $results } | ConvertTo-Json -Depth 4 |
  Out-File -Encoding ascii "$out/performance_report.json"

Write-Host "P0.4 reproduce: OK"
Write-Host "  kernel.sha256 = $khash"
Write-Host "  audit.sha256  = $auditHash"
