#!/usr/bin/env pwsh
# P0.4 — Reproducibility bootstrap (Windows / PowerShell).
# Pins toolchain, installs build-std components, verifies versions.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSCommandPath)
Set-Location ..\..
rustup toolchain install nightly-2025-04-01 --component rust-src
rustup component add rust-src --toolchain nightly-2025-04-01
rustup show active-toolchain
cargo --version
rustc --version
Write-Host "P0.4 bootstrap: OK"
