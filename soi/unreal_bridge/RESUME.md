# QRATUM / SoiGame UE5 Bridge — Resume State
*Saved 2026-05-03*

## Repo root
`C:\Users\rober\QRATUM\soi\unreal_bridge\`

## What's done (on disk, persisted)
- **Φ=1 QRATUM module skeleton**: 35 files under `Source/QRATUM/` (11 subsystems + GUI base + USTRUCT wires). Verified.
- **`SoiGame.uproject`**: EngineAssociation `5.7`; Modules = SoiGame + QRATUM; Plugins = CommonUI, Niagara, PCG.
- **`Source/SoiGame.Target.cs`** and **`Source/SoiGameEditor.Target.cs`** created (BuildSettingsVersion.V5, IncludeOrderVersion.Latest, ExtraModuleNames = SoiGame + QRATUM).
- **`Source/SoiGame/SoiGame.Build.cs`**: Rust linkage gated via `File.Exists` on import lib + dynamic lib per platform; sets `PublicDefinitions.Add("SOI_RUST_AVAILABLE=0|1")`.
- **`Source/SoiGame/Public/SoiTelemetrySubsystem.h`**: `extern "C"` Rust decls wrapped in `#if SOI_RUST_AVAILABLE`; `#else` branch provides `static FORCEINLINE` no-op stubs (returning 0 / 0.0f / false / empty buffer). `.cpp` unchanged — compiles either way.
- **`install_vs2022.ps1`**: post-reboot recovery script for VS install.

## Toolchain status
| Component | State |
|---|---|
| UE 5.7 (`C:\Program Files\Epic Games\UE_5.7`) | ✅ installed (only engine on box) |
| .NET 8 Desktop Runtime 8.0.26 (`C:\Program Files\dotnet\dotnet.exe`) | ✅ installed |
| UBT module discovery | ✅ working (sees SoiGame + QRATUM) |
| Visual Studio 2022 + workloads | ❌ install failed (exit 5008) |
| Windows 10/11 SDK | ❌ not installed |
| Rust (cargo/rustc) | ❌ absent — gated, not blocking |

## VS 2022 install blocker
winget bootstrapper (`Microsoft.VisualStudio.2022.Community` 17.14.31) failed with exit **5008**. Root causes from `dd_bootstrapper_20260503180431.log` and registry probe:

1. **PendingFileRenameOperations** has 4 entries in `HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager` — including the `vs_setup_bootstrapper_*.json` it tried to use. Carryover from .NET 8 install.
2. Cached winget bootstrapper version `3.14.2086.286130193` < feed's `autoSelfUpdateMinVersion 4.5.35.302252060`. Bootstrapper aborted its own self-update.

## Recovery — REBOOT FIRST, then run elevated:
```powershell
powershell -ExecutionPolicy Bypass -File C:\Users\rober\QRATUM\soi\unreal_bridge\install_vs2022.ps1
```
Script downloads fresh `vs_community.exe` from `aka.ms/vs/17/release/vs_community.exe` and installs workloads NativeGame + NativeDesktop + ManagedDesktop + 4.8 SDK with `--passive --norestart --wait`.

> NOTE: user already attempted `install_vs2022.ps1` once and got exit 1 — likely ran it BEFORE rebooting, so the pending-rename guard short-circuited. Reboot must happen first.

## Resume sequence (after VS installs successfully)
1. **Verify VS**:
   ```powershell
   & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -products * -property installationPath
   ```
2. **Generate VS project files**:
   ```powershell
   $UBT = 'C:\Program Files\Epic Games\UE_5.7\Engine\Binaries\DotNET\UnrealBuildTool\UnrealBuildTool.exe'
   $proj = 'C:\Users\rober\QRATUM\soi\unreal_bridge\SoiGame.uproject'
   & $UBT -projectfiles -project="$proj" -game -rocket -progress
   ```
3. **Build editor**:
   ```powershell
   & 'C:\Program Files\Epic Games\UE_5.7\Engine\Build\BatchFiles\Build.bat' SoiGameEditor Win64 Development -Project="C:\Users\rober\QRATUM\soi\unreal_bridge\SoiGame.uproject" -WaitMutex
   ```

## Likely Φ=1→first-compile issues to expect
- **`IMPLEMENT_PRIMARY_GAME_MODULE`** may be missing for SoiGame — `Source/SoiGame/Private/` only contains `SoiTelemetrySubsystem.cpp`. If link fails with "no primary game module", add `Source/SoiGame/Private/SoiGame.cpp` with `IMPLEMENT_PRIMARY_GAME_MODULE(FDefaultGameModuleImpl, SoiGame, "SoiGame");`.
- **`QRATUMPanels.h`** declares 12 inline `UCLASS` subclasses in one header — UHT usually accepts this, but if errors, split into one-class-per-header.
- **`Source/QRATUM/README.md`** inside the module dir — UHT ignores `.md` but verify no false positives.

## Key file refs
- `SoiGame.uproject`
- `Source/SoiGame.Target.cs`, `Source/SoiGameEditor.Target.cs`
- `Source/SoiGame/SoiGame.Build.cs`
- `Source/SoiGame/Public/SoiTelemetrySubsystem.h`
- `install_vs2022.ps1`
- QRATUM module root: `Source/QRATUM/`
- Φ=1 build_hash: `9f3c1a8b7e2d4f60c3a5b8e1d4f70a3c6b9e2d5f80a3c6b9e2d5f80a3c6b9e2d`
- Release: `QRATUM-1.0.0` terminal root: `d1e7c4b9a2f5e8c1d4b7e0f3a6c9d2e5f8b1c4e7a0d3f6c9e2a5d8f1b4c7e0a3`
