# QRATUM Sovereign Operations Interface (SOI)

## Overview

The Sovereign Operations Interface (SOI) is QRATUM's deterministic, auditable substrate rendered as a real-time cinematic control plane. Designed for defense, genomics, aerospace, and ASI-bounded environments.

## Architectural Principle

**UI must never execute logic.** It is a purely reflective surface bound to cryptographic state streams.

Every pixel is derived from:

```
QRADLE State → Aethernet Provenance → ZK Proof Stream → UI Telemetry Bus
```

No UI action mutates state directly.

## Visual System Layers

### Legacy Stack (WebGL/DOM)

| Layer | Engine | Role |
|-------|--------|------|
| Core UI Runtime | Three.js + WebGL | Holographic 3D operational environment |
| Control Shell | React + Tauri | Desktop sovereign console |
| Telemetry Bus | WebSocket + JSON | Deterministic state feeds |
| Security Bridge | ZK-Verified Events | Prevents UI spoofing |
| Rendering Protocol | Read-only TXO mirror | UI cannot desync from QRADLE |

### **NEW: Unreal Engine 5 Stack** 🎬

| Layer | Engine | Role |
|-------|--------|------|
| Visual Cortex | Unreal Engine 5 | Cinematic-grade rendering (Niagara, Lumen, Nanite) |
| Nervous System | Rust (`soi_telemetry_core`) | High-frequency telemetry, ZK proof stream |
| Neural Bridge | C++ FFI (`USoiTelemetrySubsystem`) | 60Hz state polling, Blueprint integration |
| Holographic HUD | CommonUI + Materials | Glass-effect UI with chromatic aberration |
| Particle Systems | Niagara | Validator nodes, zone heatmaps, proof overlays |
| Procedural Generation | PCG Framework | Dynamic execution lattice |

**See: [`unreal_bridge/README_UE5_MIGRATION.md`](unreal_bridge/README_UE5_MIGRATION.md) for complete migration guide.**

## Sovereign UI Domains

### 1. Planetary Node Map

A holographic Earth visualization with:

- Live validator nodes
- Z-zones glowing by classification
- Air-gapped Z3 vaults as black monoliths
- BFT quorum flows animated in real time

### 2. QRADLE Execution Theater

Each execution is rendered as:

- Deterministic state machine lattice
- Rollback vectors visualized as reversible time branches
- Fatal invariant violations as red horizon fractures
- ZK proofs streaming as quantum-noise overlays

### 3. Aethernet Consensus War Room

- Validator lifecycle rings
- Slashing heat maps
- Trajectory-aware collapse precursors (amber → crimson)
- Self-suspension triggers as gravitational wells

### 4. Vertical Operations Bays

Each vertical is a cinematic chamber:

| Vertical | Visual Theme |
|----------|--------------|
| VITRA-E0 | DNA helix cathedral with provenance rays |
| CAPRA | Financial lattice towers |
| JURIS | Court-grade ledger halls |
| ECORA | Planetary energy mesh |
| FLUXA | Logistics hypergraphs |

## Technology Stack

### Legacy WebGL Stack

| Component | Stack |
|-----------|-------|
| 3D Runtime | Three.js + WebGL |
| Desktop Shell | Electron + React |
| Telemetry | WebSocket + Protobuf-compatible JSON |
| Proof Bridge | ZK-verified event streams |
| Security | Hardware-bound UI attestation |
| Rendering Sync | Deterministic TXO snapshots |

### **NEW: Unreal Engine 5 Stack** 🎬

| Component | Stack |
|-----------|-------|
| 3D Runtime | Unreal Engine 5.3+ (Lumen, Nanite) |
| Telemetry Core | Rust (`tokio`, `tungstenite`, `serde`) |
| FFI Bridge | C++ Subsystem with C ABI |
| Particle Systems | Niagara |
| UI Framework | CommonUI |
| Procedural Content | PCG (Procedural Content Generation) |
| Materials | Advanced shader materials (Fresnel, DepthFade) |

## File Structure

```
soi/
├── README.md                 # This file
├── index.html               # Legacy WebGL entry point
│
├── components/              # Legacy WebGL UI components (reference)
│   ├── planetary-map.js     # Holographic Earth
│   ├── execution-theater.js # QRADLE visualization
│   ├── war-room.js          # Consensus visualization
│   └── vertical-bays.js     # Vertical chambers
│
├── assets/                  # Legacy WebGL assets
│   ├── css/
│   │   └── soi.css          # Sovereign styling
│   └── js/
│       ├── telemetry-bus.js # State stream handler
│       ├── soi-renderer.js  # Main rendering engine
│       └── soi-api.js       # API integration
│
├── telemetry/
│   └── state-stream.py      # Python state stream server
│
├── rust_core/               # 🆕 Rust Telemetry Backend
│   └── soi_telemetry_core/
│       ├── Cargo.toml
│       ├── src/lib.rs       # FFI exports for Unreal
│       └── build.sh         # Build script
│
└── unreal_bridge/           # 🆕 Unreal Engine 5 Project
    ├── SoiGame.uproject     # UE5 project file
    ├── Source/
    │   └── SoiGame/
    │       ├── Public/SoiTelemetrySubsystem.h
    │       ├── Private/SoiTelemetrySubsystem.cpp
    │       └── SoiGame.Build.cs
    ├── Content/             # Blueprints, Materials, Niagara, PCG
    ├── README_UE5_MIGRATION.md        # Complete UE5 guide
    └── BLUEPRINT_IMPLEMENTATION_GUIDE.md  # Visual setup guide
```

## Determinism Preservation

The UI:

- Cannot generate entropy
- Cannot alter execution paths  
- Cannot bypass zones
- Cannot emit unsigned events

All rendering is post-factum reflective.

## Quick Start

### Legacy WebGL Version

1. Open `soi/index.html` in a browser
2. The interface will connect to the telemetry bus
3. Real-time state streams will render automatically

### **NEW: Unreal Engine 5 Version** 🎬

#### Prerequisites

- Rust 1.70+ ([Install](https://rustup.rs/))
- Unreal Engine 5.3+ ([Download](https://www.unrealengine.com/))
- Visual Studio 2022 (Windows) or Xcode (macOS)

#### Build Steps

```bash
# 1. Build Rust telemetry core
cd soi/rust_core/soi_telemetry_core
./build.sh

# 2. Open Unreal project
# Open soi/unreal_bridge/SoiGame.uproject in Unreal Editor

# 3. Build C++ code
# In Unreal Editor: Build → Compile (Ctrl+Alt+F11)

# 4. Run in editor
# Click Play or press Alt+P
```

**See detailed instructions:** [`unreal_bridge/README_UE5_MIGRATION.md`](unreal_bridge/README_UE5_MIGRATION.md)

**Visual implementation guide:** [`unreal_bridge/BLUEPRINT_IMPLEMENTATION_GUIDE.md`](unreal_bridge/BLUEPRINT_IMPLEMENTATION_GUIDE.md)

## Integration

The SOI integrates with:

- QRADLE Engine (`/qradle/engine.py`)
- Aethernet Consensus (`/Aethernet/core/consensus.py`)
- Validator Registry (`/Aethernet/core/validator.py`)
- Vertical Modules (`/verticals/`)
