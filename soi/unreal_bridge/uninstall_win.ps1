# uninstall_win.ps1
# Uninstall IntentOS auto-start for Windows
# Removes the shortcut from the Startup folder

$StartupFolder = [Environment]::GetFolderPath('Startup')
$ShortcutPath = Join-Path $StartupFolder "IntentOS.lnk"

if (Test-Path $ShortcutPath) {
    Remove-Item $ShortcutPath -Force
    Write-Host "IntentOS auto-start uninstalled successfully."
    Write-Host "Shortcut removed from: $ShortcutPath"
} else {
    Write-Host "IntentOS auto-start shortcut not found. Nothing to uninstall."
}