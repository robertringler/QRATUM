# install_win.ps1
# Install IntentOS auto-start for Windows (user-level)
# Creates a shortcut in the Startup folder to launch IntentOS at boot

param(
    [string]$InstallPath = "$PSScriptRoot\..\Binaries\Win64\IntentOS.exe"
)

$StartupFolder = [Environment]::GetFolderPath('Startup')
$ShortcutPath = Join-Path $StartupFolder "IntentOS.lnk"

# Check if the executable exists
if (-not (Test-Path $InstallPath)) {
    Write-Error "IntentOS executable not found at: $InstallPath"
    Write-Error "Please ensure the game is packaged and built first."
    exit 1
}

# Create WScript.Shell COM object
$WshShell = New-Object -ComObject WScript.Shell

# Create shortcut
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $InstallPath
$Shortcut.WorkingDirectory = Split-Path $InstallPath
$Shortcut.Description = "IntentOS - QRATUM Sovereign Operations Interface"
$Shortcut.Save()

Write-Host "IntentOS auto-start installed successfully."
Write-Host "Shortcut created at: $ShortcutPath"
Write-Host "The game will now launch automatically at user login."