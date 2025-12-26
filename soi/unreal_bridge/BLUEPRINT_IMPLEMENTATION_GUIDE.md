# Unreal Engine 5 Blueprint Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing the SOI visual systems in Unreal Engine 5 using Blueprints, Niagara, Materials, and the PCG framework.

---

## 1. War Room Controller Blueprint (BP_WarRoomController)

### Purpose
Central controller that manages the cinematic "living notification" system and responds to telemetry state changes.

### Setup Instructions

1. **Create Blueprint**
   - Right-click in Content Browser → Blueprint Class → Actor
   - Name: `BP_WarRoomController`
   - Place in level

2. **Get Telemetry Subsystem**
   ```
   Event BeginPlay
   └─> Get Game Instance
       └─> Get Subsystem (USoiTelemetrySubsystem)
           └─> Promote to Variable: TelemetrySubsystem
   ```

3. **Bind to State Events**
   ```
   Event BeginPlay (continued)
   └─> Bind Event to OnStateUpdated (TelemetrySubsystem)
       └─> Custom Event: HandleStateUpdate
           Input: Epoch (int64), SlashingVector (float)
   ```

4. **Implement HandleStateUpdate Event**
   ```
   HandleStateUpdate Event
   ├─> Branch: SlashingVector > 0.5
   │   ├─> True: Trigger Red Alert Sequence
   │   │   ├─> Play Level Sequence: LevelSequence_RedAlert
   │   │   ├─> Set Light Color: (R=1.0, G=0.2, B=0.2)
   │   │   └─> Play Sound: Cue_WarningKlaxon
   │   └─> False: Normal Operations
   │       └─> Set Light Color: (R=0.3, G=0.6, B=1.0)
   └─> Print String: "Epoch: {Epoch}, Slashing: {SlashingVector}"
   ```

5. **Continuous Polling (Optional - for UI updates)**
   ```
   Event Tick
   └─> Get Current Epoch (TelemetrySubsystem)
       └─> Update UI Widget: EpochCounter
   ```

---

## 2. Holographic HUD (WBP_SovereignHUD)

### Purpose
Glass-effect UI overlay displaying telemetry data with chromatic aberration and blur.

### Widget Hierarchy
```
Canvas Panel
├─> Retainer Box (RetainerBox_GlassEffect)
│   └─> Overlay
│       ├─> Image: Background (Black, 0.3 opacity)
│       ├─> Text: EpochCounter
│       ├─> Text: ValidatorCount
│       └─> Text: QuorumPercentage
└─> Image: ChromaticBorder (Edge glow effect)
```

### Material Setup: M_HolographicGlass

1. **Create Material**
   - Base Material Domain: User Interface
   - Blend Mode: Translucent

2. **Material Graph**
   ```
   Scene Color Node
   ├─> Scene Depth Blur
   │   └─> Blur Amount: 5.0
   └─> Chromatic Aberration
       ├─> R Channel Offset: (0.02, 0.0)
       ├─> G Channel Offset: (0.0, 0.0)
       ├─> B Channel Offset: (-0.02, 0.0)
       └─> Fresnel
           └─> Emissive Color (Cyan glow at edges)
   ```

3. **Apply to Retainer Box**
   - Retainer Box → Effect Material: M_HolographicGlass

### Blueprint Data Binding

```
Event Construct (WBP_SovereignHUD)
└─> Get Game Instance
    └─> Get Subsystem (USoiTelemetrySubsystem)
        ├─> Bind to OnStateUpdated
        │   └─> Custom Event: UpdateEpochDisplay
        └─> Get Current Epoch
            └─> Set Text: EpochCounter
```

---

## 3. Planetary Map - Niagara Validator Particles

### Purpose
Replace Three.js sphere mesh with UE5 geosphere + Niagara particle validators.

### Planet Mesh Setup

1. **Create Geosphere**
   - Static Mesh: SM_Geosphere
   - Subdivision: 4 (high poly)
   - Scale: (80, 80, 80)

2. **Material: M_PlanetHolographic**
   ```
   Material Graph:
   ├─> Fresnel (Rim Light)
   │   ├─> Exponent: 3.0
   │   └─> Base Reflect Fraction: 0.5
   ├─> World Position Offset
   │   └─> Simplex Noise (Digital glitch effect)
   │       ├─> Scale: 0.1
   │       └─> Amplitude: 2.0 * sin(Time)
   └─> Base Color: (0.0, 0.2, 0.4)
       └─> Emissive: (0.0, 0.5, 0.7) * Fresnel
   ```

### Niagara System: NS_ValidatorParticles

1. **Create Niagara System**
   - Right-click → Niagara System → New system from selected emitters
   - Add Emitter: Simple Sprite

2. **Emitter Properties**
   - Spawn Rate: 256 particles (total validators)
   - Lifetime: Infinite
   - Particle Color: User Parameter binding

3. **User Parameters** (exposed to Blueprint)
   ```
   User.Zone0_Heat (float) - Default: 0.2
   User.Zone1_Heat (float) - Default: 0.2
   User.Zone2_Heat (float) - Default: 0.2
   User.Zone3_Heat (float) - Default: 0.2
   User.SlashingVector (float) - Default: 0.0
   ```

4. **Particle Color Module**
   ```
   Color from Curve:
   ├─> If Zone0_Heat < 0.3: Blue (0.0, 0.5, 1.0)
   ├─> If Zone0_Heat > 0.7: Red (1.0, 0.2, 0.2)
   └─> Lerp based on heat value
   ```

5. **Particle Position Module**
   ```
   Spherical Distribution:
   ├─> Zone 0: Radius = 100
   ├─> Zone 1: Radius = 115
   ├─> Zone 2: Radius = 130
   └─> Zone 3: Radius = 145
   ```

6. **Vortex Velocity Module** (responds to heat)
   ```
   Vortex Force:
   ├─> Center: Planet origin
   ├─> Radius: Zone radius
   └─> Strength: Zone_Heat * 1000
   ```

### Blueprint Integration (BP_PlanetaryMap)

```
Event BeginPlay
└─> Get Subsystem (USoiTelemetrySubsystem)
    └─> Bind to OnZoneHeatUpdated
        └─> Custom Event: UpdateZoneHeat
            Input: ZoneIndex (int), HeatValue (float)

UpdateZoneHeat Event
└─> Switch on Int: ZoneIndex
    ├─> Case 0: Set Niagara Variable (NS_ValidatorParticles, "User.Zone0_Heat", HeatValue)
    ├─> Case 1: Set Niagara Variable (NS_ValidatorParticles, "User.Zone1_Heat", HeatValue)
    ├─> Case 2: Set Niagara Variable (NS_ValidatorParticles, "User.Zone2_Heat", HeatValue)
    └─> Case 3: Set Niagara Variable (NS_ValidatorParticles, "User.Zone3_Heat", HeatValue)
```

---

## 4. Execution Theater - PCG Lattice

### Purpose
Replace WebGL line rendering with procedural crystalline lattice that grows with block height.

### PCG Graph: PCG_ExecutionLattice

1. **Create PCG Graph**
   - Right-click → PCG → PCG Graph
   - Name: PCG_ExecutionLattice

2. **Graph Structure**
   ```
   PCG Input
   └─> Grid Sampler
       ├─> Grid Size: (10, 5, 1)
       ├─> Cell Size: (30, 25, 50)
       └─> Spawn Points
           └─> Static Mesh Spawner
               ├─> Mesh: SM_Crystal_Node (custom octahedron)
               ├─> Scale: Random (0.8 - 1.2)
               └─> Material: M_CrystalGlow
   ```

3. **Block Height Integration**
   ```
   Get Game Instance
   └─> Get Subsystem (USoiTelemetrySubsystem)
       └─> Get Current Epoch
           └─> Modulo: 1000 (wrap for visualization)
               └─> Set PCG Seed
   ```

4. **Material: M_CrystalGlow**
   ```
   Material Graph:
   ├─> Base Color: (0.0, 0.7, 0.85)
   ├─> Emissive Color: (0.0, 0.7, 0.85) * 5.0
   ├─> Opacity: 0.8
   └─> Refraction: 1.5
   ```

### Dynamic Growth Blueprint (BP_LatticeController)

```
Event Tick
├─> Get Current Epoch (TelemetrySubsystem)
├─> Compute Lattice Height: Epoch / 1000.0
└─> Set Actor Scale 3D: (1.0, 1.0, LatticeHeight)
```

---

## 5. Red Alert Sequence

### Level Sequence: LevelSequence_RedAlert

1. **Create Level Sequence**
   - Content Browser → Right-click → Cinematics → Level Sequence
   - Name: LevelSequence_RedAlert

2. **Tracks**
   ```
   DirectionalLight_Main
   ├─> Light Color Track
   │   ├─> 0.0s: (0.3, 0.6, 1.0) - Normal blue
   │   ├─> 0.5s: (1.0, 0.2, 0.2) - Red alert
   │   └─> 1.0s: (1.0, 0.2, 0.2) - Hold red
   └─> Intensity Track
       └─> Pulse: sin wave (0.5 - 1.0)

   PostProcessVolume
   └─> Color Grading Track
       ├─> Saturation: 0.5 → 0.2 (desaturate)
       └─> Tint: (1.0, 0.8, 0.8) (red tint)

   Sound_Klaxon
   └─> Audio Track: Cue_WarningKlaxon
   ```

3. **Trigger from Blueprint**
   ```
   (When SlashingVector > 0.5)
   └─> Play Level Sequence: LevelSequence_RedAlert
       └─> Loop: True (until condition clears)
   ```

---

## 6. Shield Ripple Effect (Provenance Verified)

### Material: M_ShieldRipple

1. **Create Material**
   - Material Domain: Surface
   - Blend Mode: Translucent

2. **Material Graph**
   ```
   Radial Gradient
   ├─> Center: Actor Location
   ├─> Radius: Time * 500 (expand outward)
   └─> Opacity Mask
       ├─> Fresnel (Edge visibility)
       └─> Emissive: (0.2, 1.0, 0.8) * sin(Time)
   ```

3. **Blueprint Trigger** (BP_ShieldEffect)
   ```
   Bind to OnProofVerified (TelemetrySubsystem)
   └─> Custom Event: TriggerShieldRipple
       ├─> Spawn Decal at Player Location
       │   └─> Material: M_ShieldRipple
       ├─> Play Sound: Cue_ShieldActivate
       └─> Delay 2.0s
           └─> Destroy Decal
   ```

---

## 7. Integration Checklist

- [ ] Create BP_WarRoomController and place in level
- [ ] Connect TelemetrySubsystem in BeginPlay
- [ ] Create WBP_SovereignHUD with Retainer Box glass effect
- [ ] Set up M_HolographicGlass material with scene blur
- [ ] Create NS_ValidatorParticles Niagara system
- [ ] Expose User Parameters for zone heat
- [ ] Create PCG_ExecutionLattice graph
- [ ] Bind PCG seed to current epoch
- [ ] Create LevelSequence_RedAlert
- [ ] Add DirectionalLight and PostProcess tracks
- [ ] Create BP_ShieldEffect for provenance ripples
- [ ] Test all Blueprint bindings with demo data

---

## 8. Performance Optimization

### LOD Settings
- Planetary mesh: Enable LOD auto-generation (5 levels)
- Niagara particles: Max 256 particles, cull at distance > 5000 units
- PCG lattice: Max 50 nodes, disable at distance > 3000 units

### Lighting
- Use Lumen for real-time global illumination
- Virtual Shadow Maps for soft shadows
- Screen Space Reflections for holographic surfaces

### Materials
- Enable Material Quality switching (Low/Medium/High)
- Use Material Parameter Collections for global state
- Cache scene captures for glass effect (30 FPS update rate)

---

## 9. Testing

1. **Standalone Testing**
   ```
   Play in Editor → Standalone Game
   ```

2. **Connect to Local Telemetry**
   ```
   In BP_WarRoomController BeginPlay:
   └─> Connect To Aethernet: "ws://localhost:8000/soi/telemetry"
   ```

3. **Simulate State Changes**
   - Manually set User Parameters in Niagara
   - Test slashing threshold (0.5) triggers red alert
   - Verify proof verified event spawns shield ripple

---

## Blueprint Node Quick Reference

### Common Nodes
- `Get Game Instance` → `Get Subsystem` (Class: USoiTelemetrySubsystem)
- `Get Current Epoch` (Pure)
- `Get Zone Heat` (Pure, Input: Zone Index)
- `Get Slashing Vector` (Pure)
- `Bind Event to OnStateUpdated`
- `Bind Event to OnZoneHeatUpdated`
- `Bind Event to OnProofVerified`

### Niagara Nodes
- `Set Niagara Variable (Float)`
- `Spawn System at Location`
- `Set Emitter Enabled`

### Level Sequence Nodes
- `Play Level Sequence`
- `Set Playback Position`
- `Stop Level Sequence`

---

This guide provides the complete implementation blueprint for translating the WebGL/DOM SOI to Unreal Engine 5 with high-fidelity cinematic rendering.
