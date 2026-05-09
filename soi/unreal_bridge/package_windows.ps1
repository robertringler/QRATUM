# package_windows.ps1
# Package IntentOS for Windows distribution
# Creates a zip archive with installer

param(
    [string]$OutputDir = ".",
    [switch]$IncludeInstaller
)

$PackageName = "IntentOS_Windows"
$PackageDir = "PackagedGame\Windows"

if (-not (Test-Path $PackageDir)) {
    Write-Error "Packaged game not found at: $PackageDir"
    Write-Error "Please build and package the game first."
    exit 1
}

# Create package directory
$PackageRoot = Join-Path $OutputDir $PackageName
New-Item -ItemType Directory -Force -Path $PackageRoot | Out-Null

# Copy game files
Copy-Item "$PackageDir\*" $PackageRoot -Recurse -Force

# Copy install scripts
Copy-Item "install_win.ps1" $PackageRoot
Copy-Item "uninstall_win.ps1" $PackageRoot

# Create zip archive
$ArchiveName = "$PackageName.zip"
$ArchivePath = Join-Path $OutputDir $ArchiveName
Compress-Archive -Path $PackageRoot -DestinationPath $ArchivePath -Force

Write-Host "Package created: $ArchivePath"

if ($IncludeInstaller) {
    # Compile Inno Setup installer (requires ISCC.exe in PATH)
    $IssPath = "IntentOS.iss"
    if (Test-Path $IssPath) {
        & ISCC.exe $IssPath /O"$OutputDir" /F"$PackageName`_Installer"
        Write-Host "Installer created: $OutputDir\$PackageName`_Installer.exe"
    } else {
        Write-Warning "Inno Setup script not found: $IssPath"
    }
}

Write-Host "To install:"
Write-Host "  Extract $ArchiveName"
Write-Host "  Run .\install_win.ps1 (as admin if system-wide)"

# Cleanup
Remove-Item $PackageRoot -Recurse -Force