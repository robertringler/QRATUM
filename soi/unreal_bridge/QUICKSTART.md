# Quick Start: IntentOS Unreal Engine 5.7.4 Setup

This guide will get you up and running with IntentOS in 10 minutes.

## TL;DR

```bash
# 1. Validate your setup
cd soi/unreal_bridge
./validate_setup.sh

# 2. Build Rust core
cd ../rust_core/soi_telemetry_core
./build.sh

# 3. Open in Unreal Editor (requires UE5 5.7.4)
# Open: soi/unreal_bridge/IntentOS.uproject

# 4. Build and package
# Windows: .\package_windows.ps1
# Linux: ./package_linux.sh

# 5. Install auto-boot
# Windows: .\install_win.ps1
# Linux: ./install_linux.sh
```

## Prerequisites (5 minutes)

### 1. Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 2. Install Unreal Engine 5.7.4
- Download Epic Games Launcher: https://www.unrealengine.com/
- Install Unreal Engine 5.7.4 (exact version required)

### 3. Install C++ Build Tools

**Windows:**
- Visual Studio 2022 with "Desktop development with C++"

**macOS:**
```bash
xcode-select --install
```

**Linux:**
```bash
sudo apt-get install build-essential clang lld
```

## Step-by-Step Setup (5 minutes)

### Step 1: Validate Setup

```bash
cd soi/unreal_bridge
./validate_setup.sh
```

Expected output:
```
✓ Rust installed
✓ Cargo installed
✓ C++ compiler found
✓ All files present
✓ Rust code compiles
✓ Tests pass
```

### Step 2: Build Rust Core

```bash
cd ../rust_core/soi_telemetry_core
./build.sh
```

Expected output:
```
🔨 Building Rust telemetry core...
✅ Build successful!
📦 Output: target/release/libsoi_telemetry_core.so
```

### Step 3: Open UE5 Project

1. Launch Unreal Engine 5.7.4
2. Click "Browse"
3. Navigate to `soi/unreal_bridge/IntentOS.uproject`
4. Click "Open"

When prompted to rebuild modules:
- Click "Yes"
- Wait for compilation (~2-5 minutes)

### Step 4: Build and Package

**Windows:**
```powershell
# Build in UE5 Editor or via command line
# Then package:
.\package_windows.ps1
# Creates: IntentOS_Windows.zip and optional installer
```

**Linux:**
```bash
# Build in UE5 Editor or via command line
# Then package:
./package_linux.sh
# Creates: IntentOS_Linux.tar.gz
```

### Step 5: Install Auto-Boot

**Windows:**
```powershell
# Install auto-start
.\install_win.ps1
# Creates shortcut in Startup folder

# Uninstall
.\uninstall_win.ps1
```

**Linux (Desktop):**
```bash
# Install auto-start
./install_linux.sh
# Creates ~/.config/autostart/IntentOS.desktop

# Uninstall
./uninstall_linux.sh
```

**Linux (System/Kiosk):**
```bash
# Package with systemd support
./package_linux.sh . true
# Then install system-wide:
sudo cp IntentOS_Linux/intentos.service /etc/systemd/system/
sudo systemctl enable intentos@$USER.service
sudo systemctl start intentos@$USER.service
```

## CI/CD

GitHub Actions workflows are configured for automated builds:
- `build-package-windows.yml`: Windows builds and packaging
- `build-package-linux.yml`: Linux builds and packaging

Requires Epic Games GitHub token for UE 5.7.4 access.

Or in Blueprint:
1. Create new Blueprint Actor
2. Add node: "Get Game Instance"
3. Add node: "Get Subsystem" → Select "Soi Telemetry Subsystem"
4. Success! The subsystem is available

## Next Steps

### For Testing (Demo Mode)

The system includes demo mode for testing without a telemetry server:

1. Create `BP_TestController` Blueprint
2. Add this logic:
   ```
   Event BeginPlay
   └─> Get Subsystem (USoiTelemetrySubsystem)
       └─> Print String: "SOI Ready"
   ```
3. Place in level and press Play

### For Visual Implementation

Follow the comprehensive guide:

**Read:** [`BLUEPRINT_IMPLEMENTATION_GUIDE.md`](BLUEPRINT_IMPLEMENTATION_GUIDE.md)

This guide includes step-by-step instructions for:
- Holographic HUD with glass effects
- Planetary Map Niagara particles
- Execution Theater PCG lattice
- War Room controller logic
- Red Alert sequences
- Shield ripple effects

Time estimate: 4-6 days for complete visual implementation

## Troubleshooting

### "Rust library not found"

**Solution:** Build Rust core first:
```bash
cd soi/rust_core/soi_telemetry_core
./build.sh
```

### "Module 'SoiGame' could not be loaded"

**Solution:** Rebuild C++ modules:
1. Close Unreal Editor
2. Delete `Binaries/`, `Intermediate/`, `Saved/` directories
3. Right-click `SoiGame.uproject` → Generate Visual Studio project files
4. Reopen project

### "Validation script fails"

**Solution:** Check prerequisites:
```bash
rustc --version  # Should be 1.70+
cargo --version
clang++ --version  # Or g++/MSVC
```

## Architecture Overview

```
┌─────────────────────────────────────────┐
│  Unreal Engine 5 (Visual Cortex)       │
│  • Niagara, Lumen, Nanite              │
│  • CommonUI, Materials, PCG            │
└──────────────┬──────────────────────────┘
               │ Blueprint API
               ↓
┌─────────────────────────────────────────┐
│  C++ Bridge (USoiTelemetrySubsystem)    │
│  • 60Hz polling                         │
│  • Event broadcasting                   │
└──────────────┬──────────────────────────┘
               │ FFI (C ABI)
               ↓
┌─────────────────────────────────────────┐
│  Rust Core (soi_telemetry_core)         │
│  • Async WebSocket                      │
│  • Thread-safe state                    │
└──────────────┬──────────────────────────┘
               │ WebSocket
               ↓
        [Aethernet Telemetry]
```

## Resources

- **Full Migration Guide:** [`README_UE5_MIGRATION.md`](README_UE5_MIGRATION.md)
- **Blueprint Implementation:** [`BLUEPRINT_IMPLEMENTATION_GUIDE.md`](BLUEPRINT_IMPLEMENTATION_GUIDE.md)
- **Architecture Details:** [`ARCHITECTURE.md`](ARCHITECTURE.md)
- **Implementation Summary:** [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md)

## Getting Help

1. **Check Documentation:** 48KB of guides included
2. **Run Validation:** `./validate_setup.sh`
3. **Review Logs:** Check UE5 Output Log for errors
4. **Test Rust:** `cargo test` in `rust_core/soi_telemetry_core/`

## Performance Targets

- Frame time: < 0.1ms per frame (< 1% of 60 FPS budget)
- Memory: ~5 MB total
- Latency: < 100ms end-to-end
- Throughput: 10,000+ events/sec

---

**Ready to build the future of sovereign operations interfaces! 🎬**
