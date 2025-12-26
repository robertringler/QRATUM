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

| Layer | Engine | Role |
|-------|--------|------|
| Core UI Runtime | Three.js + WebGL | Holographic 3D operational environment |
| Control Shell | React + Tauri | Desktop sovereign console |
| Telemetry Bus | WebSocket + JSON | Deterministic state feeds |
| Security Bridge | ZK-Verified Events | Prevents UI spoofing |
| Rendering Protocol | Read-only TXO mirror | UI cannot desync from QRADLE |

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

| Component | Stack |
|-----------|-------|
| 3D Runtime | Three.js + WebGL |
| Desktop Shell | Electron + React |
| Telemetry | WebSocket + Protobuf-compatible JSON |
| Proof Bridge | ZK-verified event streams |
| Security | Hardware-bound UI attestation |
| Rendering Sync | Deterministic TXO snapshots |

## File Structure

```
soi/
├── README.md                 # This file
├── index.html               # SOI main entry point
├── components/              # SOI UI components
│   ├── planetary-map.js     # Holographic Earth
│   ├── execution-theater.js # QRADLE visualization
│   ├── war-room.js          # Consensus visualization
│   └── vertical-bays.js     # Vertical chambers
├── assets/
│   ├── css/
│   │   └── soi.css          # Sovereign styling
│   └── js/
│       ├── telemetry-bus.js # State stream handler
│       ├── soi-renderer.js  # Main rendering engine
│       └── soi-api.js       # API integration
├── telemetry/
│   └── state-stream.py      # Python state stream server
└── api/
    └── soi-endpoints.py     # API endpoints for SOI
```

## Determinism Preservation

The UI:
- Cannot generate entropy
- Cannot alter execution paths  
- Cannot bypass zones
- Cannot emit unsigned events

All rendering is post-factum reflective.

## Quick Start

1. Open `soi/index.html` in a browser
2. The interface will connect to the telemetry bus
3. Real-time state streams will render automatically

## Integration

The SOI integrates with:
- QRADLE Engine (`/qradle/engine.py`)
- Aethernet Consensus (`/Aethernet/core/consensus.py`)
- Validator Registry (`/Aethernet/core/validator.py`)
- Vertical Modules (`/verticals/`)
