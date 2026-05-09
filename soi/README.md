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
    ├── IntentOS.uproject     # UE5 project file (renamed)
    ├── Source/
    │   └── IntentOS/
    │       ├── Public/SoiTelemetrySubsystem.h
    │       ├── Private/SoiTelemetrySubsystem.cpp
    │       └── IntentOS.Build.cs
    ├── Content/             # Blueprints, Materials, Niagara, PCG
    ├── README_UE5_MIGRATION.md        # Complete UE5 guide
    └── BLUEPRINT_IMPLEMENTATION_GUIDE.md  # Visual setup guide
