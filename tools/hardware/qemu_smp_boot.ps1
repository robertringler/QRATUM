# Boots qratum.efi under QEMU + OVMF with -smp 4 and captures serial output.
# Prereqs:
#   - QEMU for Windows installed and `qemu-system-x86_64.exe` on PATH
#   - OVMF firmware available at $env:OVMF_FD or C:\Tools\OVMF\OVMF.fd
#
# Usage:
#   pwsh tools\hardware\qemu_smp_boot.ps1
#   pwsh tools\hardware\qemu_smp_boot.ps1 -Cores 8 -Timeout 30

param(
    [int]$Cores   = 4,
    [int]$Timeout = 20,
    [string]$Ovmf = $env:OVMF_FD
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$efi  = Join-Path $repo 'os\qratum-os\target\x86_64-unknown-uefi\debug\qratum.efi'
if (-not (Test-Path $efi)) {
    Write-Error "qratum.efi not found at $efi. Build first: cargo +nightly-2025-04-01 build -p qratum-os ..."
}

if (-not $Ovmf -or -not (Test-Path $Ovmf)) {
    foreach ($p in @('C:\Tools\OVMF\OVMF.fd', 'C:\Program Files\qemu\share\OVMF.fd')) {
        if (Test-Path $p) { $Ovmf = $p; break }
    }
}
if (-not $Ovmf -or -not (Test-Path $Ovmf)) {
    Write-Error "OVMF.fd not found. Set `$env:OVMF_FD or pass -Ovmf <path>."
}

# Stage an ESP image: copy qratum.efi → EFI/BOOT/BOOTX64.EFI inside a temp FAT image.
$stage = Join-Path $env:TEMP "qratum_esp_$([guid]::NewGuid().ToString('N'))"
$null = New-Item -ItemType Directory -Path "$stage\EFI\BOOT" -Force
Copy-Item $efi "$stage\EFI\BOOT\BOOTX64.EFI" -Force

$log = Join-Path $repo 'reports\qemu_smp_boot.log'
$null = New-Item -ItemType Directory -Path (Split-Path $log) -Force -ErrorAction SilentlyContinue

Write-Host "[qemu] -smp $Cores  efi=$efi  ovmf=$Ovmf"
Write-Host "[qemu] capturing serial → $log (timeout ${Timeout}s)"

$qemu = 'qemu-system-x86_64.exe'
$args = @(
    '-machine', 'q35,accel=tcg',
    '-cpu',     'qemu64,+x2apic',
    '-smp',     "$Cores",
    '-m',       '512',
    '-bios',    $Ovmf,
    '-drive',   "format=raw,file=fat:rw:$stage",
    '-serial',  "file:$log",
    '-display', 'none',
    '-no-reboot',
    '-no-shutdown'
)

$proc = Start-Process -FilePath $qemu -ArgumentList $args -PassThru -WindowStyle Hidden
try {
    if (-not $proc.WaitForExit($Timeout * 1000)) {
        $proc.Kill()
    }
} finally {
    if (Test-Path $stage) { Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue }
}

if (-not (Test-Path $log)) { Write-Error "No serial log produced." }
$content = Get-Content $log -Raw

$pass = @()
$fail = @()
if ($content -match 'BootReady')          { $pass += 'BootReady' }      else { $fail += 'BootReady' }
if ($content -match 'audit')              { $pass += 'audit'  }
# Once hw::init is wired, the line below becomes mandatory:
# if ($content -match "aps_started=$($Cores - 1)") { $pass += 'aps_started' } else { $fail += 'aps_started' }

Write-Host ""
Write-Host "=== qemu serial capture ==="
Write-Host $content
Write-Host "==========================="
Write-Host "PASS: $($pass -join ', ')"
if ($fail.Count) {
    Write-Host "FAIL: $($fail -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "OK" -ForegroundColor Green
