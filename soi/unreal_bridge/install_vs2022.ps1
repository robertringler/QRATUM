# Re-run after a reboot to install Visual Studio 2022 Community
# with the workloads UE 5.7 needs (Game C++ + Native Desktop).
#
# Why this script: the previous winget attempt failed with exit 5008
# because Windows had PendingFileRenameOperations queued (left over
# from the .NET 8 Desktop Runtime install). The VS bootstrapper aborts
# in that state. Reboot first, then run this from an elevated PowerShell.

$ErrorActionPreference = 'Stop'

# Sanity: confirm no pending renames remain.
$pfr = Get-ItemProperty `
    -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager' `
    -Name PendingFileRenameOperations -ErrorAction SilentlyContinue
if ($pfr) {
    Write-Warning "PendingFileRenameOperations still present ($($pfr.PendingFileRenameOperations.Count) entries). Reboot again before continuing."
    exit 1
}

# Pull the latest official bootstrapper directly (bypasses winget's
# cached older bootstrapper that hit "Unexpected Installer Version").
$boot = Join-Path $env:TEMP 'vs_community.exe'
Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vs_community.exe' -OutFile $boot

$args = @(
    '--add', 'Microsoft.VisualStudio.Workload.NativeGame',
    '--add', 'Microsoft.VisualStudio.Workload.NativeDesktop',
    '--add', 'Microsoft.VisualStudio.Workload.ManagedDesktop',
    '--add', 'Microsoft.Net.Component.4.8.SDK',
    '--includeRecommended',
    '--passive',          # show progress, no prompts
    '--norestart',
    '--wait'
)

Write-Host "Launching VS 2022 Community installer..."
$p = Start-Process -FilePath $boot -ArgumentList $args -Wait -PassThru
Write-Host "Installer exited with code $($p.ExitCode)"
exit $p.ExitCode
