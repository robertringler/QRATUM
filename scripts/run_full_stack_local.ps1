# Run full stack locally without Docker on Windows
# This script starts both backend and frontend services

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$BackendDir = Join-Path $RootDir "autonomous_systems_platform\services\backend"
$FrontendDir = Join-Path $RootDir "autonomous_systems_platform\services\frontend"

Write-Host "=== QuASIM Full Stack Local Runner ===" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Python is required but not installed." -ForegroundColor Red
    exit 1
}

Write-Host "Starting backend service..." -ForegroundColor Green
Set-Location $BackendDir

# Create virtual environment if it doesn't exist
if (-not (Test-Path ".venv")) {
    Write-Host "Creating Python virtual environment..."
    python -m venv .venv
}

# Activate virtual environment
& .venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing backend dependencies..."
pip install -q -r requirements.txt

# Set environment variables
$env:JAX_PLATFORM_NAME = "cpu"
$env:PORT = "8000"

# Start backend in background
Write-Host "Starting backend on http://localhost:8000"
$BackendJob = Start-Job -ScriptBlock {
    param($Dir)
    Set-Location $Dir
    & .venv\Scripts\Activate.ps1
    $env:JAX_PLATFORM_NAME = "cpu"
    $env:PORT = "8000"
    python app.py
} -ArgumentList $BackendDir

# Give backend time to start
Start-Sleep -Seconds 3

# Start frontend
Write-Host "Starting frontend on http://localhost:8080"
Set-Location $FrontendDir
$FrontendJob = Start-Job -ScriptBlock {
    param($Dir)
    Set-Location $Dir
    python -m http.server 8080
} -ArgumentList $FrontendDir

Write-Host ""
Write-Host "=== Services Started ===" -ForegroundColor Cyan
Write-Host "Backend:  http://localhost:8000"
Write-Host "Frontend: http://localhost:8080"
Write-Host ""
Write-Host "Press Ctrl+C to stop all services"
Write-Host ""

# Handle cleanup
try {
    # Wait for user interrupt
    while ($true) {
        Start-Sleep -Seconds 1
        
        # Check if jobs are still running
        if ($BackendJob.State -ne "Running" -or $FrontendJob.State -ne "Running") {
            Write-Host "One or more services stopped unexpectedly." -ForegroundColor Yellow
            break
        }
    }
}
finally {
    Write-Host ""
    Write-Host "Stopping services..." -ForegroundColor Yellow
    Stop-Job -Job $BackendJob, $FrontendJob -ErrorAction SilentlyContinue
    Remove-Job -Job $BackendJob, $FrontendJob -Force -ErrorAction SilentlyContinue
    Write-Host "Done." -ForegroundColor Green
}
