# Boot qratum.efi in QEMU + OVMF (UEFI x86_64).
# Requirements:
#   - qemu-system-x86_64 on PATH
#   - OVMF firmware: set $env:QRATUM_OVMF_CODE and $env:QRATUM_OVMF_VARS, OR
#     drop OVMF_CODE.fd / OVMF_VARS.fd into ./os/qratum-os/firmware/
#
# Usage: pwsh ./scripts/run-qemu.ps1 [debug|release]

param([string]$Profile = "debug")

$ErrorActionPreference = 'Stop'

Push-Location $PSScriptRoot/..
try {
    & "$PSScriptRoot/build.ps1" $Profile

    $code = $env:QRATUM_OVMF_CODE
    $vars = $env:QRATUM_OVMF_VARS
    if (-not $code -or -not (Test-Path $code)) { $code = Join-Path $PWD "firmware/OVMF_CODE.fd" }
    if (-not $vars -or -not (Test-Path $vars)) { $vars = Join-Path $PWD "firmware/OVMF_VARS.fd" }
    if (-not (Test-Path $code)) { throw "OVMF_CODE.fd not found. Set `$env:QRATUM_OVMF_CODE or place it in firmware/" }
    if (-not (Test-Path $vars)) { throw "OVMF_VARS.fd not found. Set `$env:QRATUM_OVMF_VARS or place it in firmware/" }

    # Make a per-run writable copy of NVRAM so the master file isn't mutated.
    $varsRun = Join-Path $PWD "out/OVMF_VARS_run.fd"
    Copy-Item $vars $varsRun -Force

    $esp = Join-Path $PWD "out/esp"

    Write-Host "[qemu] booting QRATUM (Ctrl+A then X to quit)" -ForegroundColor Cyan
    qemu-system-x86_64 `
        -machine q35,accel=tcg `
        -cpu qemu64 `
        -m 512M `
        -nodefaults `
        -no-reboot `
        -drive if=pflash,format=raw,readonly=on,file="$code" `
        -drive if=pflash,format=raw,file="$varsRun" `
        -drive format=raw,file=fat:rw:"$esp" `
        -serial stdio `
        -display none
}
finally { Pop-Location }
