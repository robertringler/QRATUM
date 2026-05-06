<#
.SYNOPSIS
    Install / uninstall QRATUM as a logon-time boot task.

.DESCRIPTION
    Registers two scheduled tasks under "QRATUM\":
        QRATUM-Kernel    — headless Φ=2 kernel daemon (always)
        QRATUM-Desktop   — desktop shell (optional, -WithDesktop)
    Both run at user logon, no admin elevation required.

    State path:  %LOCALAPPDATA%\QRATUM\bridge\state.json

.EXAMPLES
    pwsh .\qratum-boot-install.ps1                    # kernel only
    pwsh .\qratum-boot-install.ps1 -WithDesktop       # kernel + GUI
    pwsh .\qratum-boot-install.ps1 -WithUE5           # + UE5 sim
    pwsh .\qratum-boot-install.ps1 -Uninstall
    pwsh .\qratum-boot-install.ps1 -Status
#>
[CmdletBinding()]
param(
    [switch]$WithDesktop,
    [switch]$WithUE5,
    [switch]$Uninstall,
    [switch]$Status,
    [int]$Hz = 60
)

$ErrorActionPreference = 'Stop'

$RepoRoot   = Split-Path -Parent $MyInvocation.MyCommand.Path
$Daemon     = Join-Path $RepoRoot 'qratum_kernel\qratum_kernel_daemon.py'
$DesktopExe = Join-Path $RepoRoot 'dist\desktop\qratum-desktop\QRATUM Desktop.exe'
$BootCmd    = Join-Path $RepoRoot 'qratum-boot.cmd'
$Folder     = '\QRATUM'
$KernelTask = 'QRATUM-Kernel'
$ShellTask  = 'QRATUM-Desktop'

function Get-PythonPath {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $cmd) { throw "python not found on PATH (required for QRATUM kernel)" }
    return $cmd.Source
}

function Show-Status {
    Get-ScheduledTask -TaskPath "$Folder\" -ErrorAction SilentlyContinue |
        Select-Object TaskName, State, @{N='LastRun';E={(Get-ScheduledTaskInfo $_).LastRunTime}},
                      @{N='LastResult';E={'0x{0:X}' -f (Get-ScheduledTaskInfo $_).LastTaskResult}} |
        Format-Table -AutoSize
}

function Remove-QRATUMTasks {
    foreach ($name in @($KernelTask, $ShellTask)) {
        $existing = Get-ScheduledTask -TaskPath "$Folder\" -TaskName $name -ErrorAction SilentlyContinue
        if ($existing) {
            Unregister-ScheduledTask -TaskPath "$Folder\" -TaskName $name -Confirm:$false
            Write-Host "[QRATUM] removed task $Folder\$name"
        }
    }
}

function Register-KernelTask {
    if (-not (Test-Path $Daemon)) { throw "kernel daemon not found: $Daemon" }
    $python = Get-PythonPath
    $argLine = "`"$Daemon`" --hz $Hz --quiet"

    $action  = New-ScheduledTaskAction -Execute $python -Argument $argLine -WorkingDirectory $RepoRoot
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    # Restart on failure; keep alive indefinitely.
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RestartCount 999 `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit (New-TimeSpan -Hours 0)   # unlimited
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

    Register-ScheduledTask -TaskPath $Folder -TaskName $KernelTask `
        -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
        -Description 'QRATUM headless kernel daemon (CIIR/CRS/RIC/QuaSim/QuBIC/RMHD/MVRI/RWB/CGL).' `
        -Force | Out-Null
    Write-Host "[QRATUM] registered $Folder\$KernelTask  (python $argLine)"
}

function Register-DesktopTask {
    if (-not (Test-Path $DesktopExe)) {
        Write-Warning "desktop binary missing: $DesktopExe - skipping QRATUM-Desktop task"
        return
    }
    $action  = New-ScheduledTaskAction -Execute $DesktopExe -WorkingDirectory (Split-Path $DesktopExe)
    # Slight delay so the kernel has published state.json before the GUI paints.
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $trigger.Delay = 'PT5S'
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

    Register-ScheduledTask -TaskPath $Folder -TaskName $ShellTask `
        -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
        -Description 'QRATUM Desktop shell (Tauri).' -Force | Out-Null
    Write-Host "[QRATUM] registered $Folder\$ShellTask  ($DesktopExe)"
}

# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------
if ($Status)    { Show-Status; return }
if ($Uninstall) { Remove-QRATUMTasks; Show-Status; return }

# Always remove first so re-installs are idempotent.
Remove-QRATUMTasks

Register-KernelTask
if ($WithDesktop) { Register-DesktopTask }

if ($WithUE5) {
    Write-Warning ('UE5 autostart not added; launch manually: ' + $BootCmd + ' --with-ue5')
}

$statePath = Join-Path $env:LOCALAPPDATA 'QRATUM\bridge\state.json'
Write-Host ''
Write-Host ('[QRATUM] installed. State on next logon: ' + $statePath)
Write-Host ('[QRATUM] manage:    Get-ScheduledTask -TaskPath ' + $Folder + '\')
Write-Host ('[QRATUM] start now: Start-ScheduledTask -TaskPath ' + $Folder + '\ -TaskName ' + $KernelTask)
Show-Status
