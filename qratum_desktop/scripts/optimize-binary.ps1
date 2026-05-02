# Ultra-lightweight build optimization (Windows)

Write-Host "🦀 Building QRATUM Desktop (Ultra-Lightweight Mode)" -ForegroundColor Green

# Step 1: Build with size optimization
Set-Location src-tauri
cargo build --release

# Step 2: Check if UPX is available
$upxPath = Get-Command upx -ErrorAction SilentlyContinue
if ($upxPath) {
    Write-Host "🗜️ Compressing with UPX..." -ForegroundColor Yellow
    upx --best --lzma target/release/qratum-desktop.exe
    Write-Host "✅ UPX compression complete" -ForegroundColor Green
} else {
    Write-Host "⚠️ UPX not found, skipping compression" -ForegroundColor Yellow
    Write-Host "   Install from: https://upx.github.io/" -ForegroundColor Gray
}

# Step 3: Report final size
if (Test-Path "target/release/qratum-desktop.exe") {
    $binarySize = (Get-Item target/release/qratum-desktop.exe).Length / 1MB
    Write-Host "🎉 Final binary size: $([math]::Round($binarySize, 2)) MB" -ForegroundColor Cyan
} else {
    Write-Host "⚠️ Binary not found at target/release/qratum-desktop.exe" -ForegroundColor Yellow
}

# Step 4: Build installer
Set-Location ..
cargo tauri build

Write-Host "✅ Build complete!" -ForegroundColor Green
Write-Host "📦 Installer: src-tauri/target/release/bundle/msi/" -ForegroundColor Cyan
