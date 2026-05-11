# QRATUM Module — Φ=1 Skeleton

UE5 module materializing the QRATUM swarm dependency graph (DependencyGraph_v0)
as compileable subsystem shells. **No behavior** lives here; this is the Φ=1
seal artifact only.

## Layout

```
Source/QRATUM/
├── QRATUM.Build.cs
├── Private/
│   ├── QRATUMModule.cpp        ← IMPLEMENT_MODULE(QRATUM)
│   ├── RWB/    MVRI/    CIIR/  OBS/   CRS/   CGL/
│   ├── QuaSim/ QuBIC/   RMHD/  RIC/   GUI/
└── Public/
    ├── RWB/RWBPacket.h           FRWBPacket
    ├── RWB/RWBSubsystem.h        URWBSubsystem            (A-08)
    ├── MVRI/MVRIChannel.h        FMVRIChannel + status enum
    ├── MVRI/MVRISubsystem.h      UMVRISubsystem           (A-08)
    ├── CIIR/ObserverLoopState.h  FObserverLoopState
    ├── CIIR/CIIRSubsystem.h      UCIIRSubsystem           (A-02)
    ├── OBS/ObserverLoopDebugComponent.h                   (A-09)
    ├── CRS/WhitneyStratum.h      FWhitneyStratum
    ├── CRS/CRSSubsystem.h        UCRSSubsystem            (A-03)
    ├── CGL/CGLSubsystem.h        UCGLSubsystem            (A-09)
    ├── QuaSim/CHSHResult.h       FCHSHResult
    ├── QuaSim/QuaSimSubsystem.h  UQuaSimSubsystem         (A-05)
    ├── QuBIC/GenomicGraphState.h FGenomicNode/State
    ├── QuBIC/QuBICSubsystem.h    UQuBICSubsystem          (A-06)
    ├── RMHD/RMHDSubsystem.h      URMHDSubsystem           (A-07)
    ├── RIC/DispatchMatrix.h      FDispatchMatrix (4 cells)
    ├── RIC/RICSubsystem.h        URICSubsystem            (A-04)
    ├── GUI/QRATUMPanelWidget.h   UQRATUMPanelWidget (base)
    ├── GUI/QRATUMHUDLayer.h      UQRATUMHUDLayer          (A-10)
    └── GUI/Panels/QRATUMPanels.h P-01..P-12 widget classes
```

## Topological order (Initialize)

RWB → MVRI → CIIR → OBS → CRS → CGL → QuaSim → QuBIC → RMHD → RIC → GUI

## Teardown order (Deinitialize)

GUI → RIC → RMHD → QuBIC → QuaSim → CGL → CRS → OBS → CIIR → MVRI → RWB

## Build

The `QRATUM` module is registered in `SoiGame.uproject` alongside `SoiGame`.
Regenerate project files (right-click `.uproject` → *Generate Visual Studio
project files*) and build the `SoiGameEditor` target.

All subsystems are `UTickableWorldSubsystem` so they auto-register and tick
once a `UWorld` is created (e.g. PIE start). Tick bodies are intentionally
empty — Φ=2 lands behavior.

## Φ=2 entry points (not implemented)

- `URWBSubsystem::Tick`              — sensor poll → `FRWBPacket`
- `UMVRISubsystem::Ingest`           — ER/CS/SS/IC/AUD constraint gate
- `UCIIRSubsystem::OnTopologyUpdated`— fires on `‖ΔW‖ > ε`
- `UCGLSubsystem::OnRankDeficiencyDetected` → `UCRSSubsystem::UpdateStratumBoundary`
- `URICSubsystem` 4-cell dispatch    — MVRI/QuaSim/QuBIC/RMHD inbound
- `UQRATUMHUDLayer::RegisterPanel`   — flips `panel_live_flags` true

## Provenance

Materialized from A-01 PHASE_SEAL(Φ=1), build_hash
`9f3c1a8b7e2d4f60c3a5b8e1d4f70a3c6b9e2d5f80a3c6b9e2d5f80a3c6b9e2d`.
