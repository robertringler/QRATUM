# Build qratum.efi
# Usage: pwsh ./scripts/build.ps1 [debug|release]
param([string]$Profile = "debug")

$ErrorActionPreference = 'Stop'

Push-Location $PSScriptRoot/..
try {
    Write-Host "[build] cargo build ($Profile)" -ForegroundColor Cyan
    $bstd = '-Zbuild-std=core,compiler_builtins,alloc'
    $bstdf = '-Zbuild-std-features=compiler-builtins-mem'
    if ($Profile -eq 'release') {
        cargo build -p qratum-os --release $bstd $bstdf
        $efi = "target/x86_64-unknown-uefi/release/qratum.efi"
    } else {
        cargo build -p qratum-os $bstd $bstdf
        $efi = "target/x86_64-unknown-uefi/debug/qratum.efi"
    }
    if (-not (Test-Path $efi)) { throw "build did not produce $efi" }
    Write-Host "[build] OK  ->  $efi" -ForegroundColor Green

    # Stage an ESP image at out/esp/EFI/BOOT/BOOTX64.EFI
    $esp = Join-Path $PWD "out/esp"
    New-Item -ItemType Directory -Force -Path "$esp/EFI/BOOT" | Out-Null
    Copy-Item $efi "$esp/EFI/BOOT/BOOTX64.EFI" -Force
    Write-Host "[build] ESP staged at $esp" -ForegroundColor Green
}
finally { Pop-Location }
