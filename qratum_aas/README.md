# QRATUM Asymmetric Adaptive Search (AAS)
## Unreal Engine 5 AI Framework - Unreal Fest Chicago 2026

---

## Overview

QRATUM AAS is a production-grade, deterministic, frame-safe AI framework for Unreal Engine 5 that implements **Asymmetric Adaptive Search** - a tree-search based planning system that outperforms both traditional Behavior Trees and LLM-driven AI in tactical game scenarios.

### Key Achievements
- Based on BOB chess engine: **#1 on Kaggle Chess AI Benchmark**
- **97% win rate** against top engines
- Defeated GPT-4, Claude, Grok-4, Gemini 2.5, and o3 in head-to-head matches
- **20+ ply tactical depth** in real-time

---

## Why AAS?

### vs. Behavior Trees
| Aspect | Behavior Trees | AAS |
|--------|---------------|-----|
| Planning | Reactive | Predictive (looks ahead) |
| Behavior Discovery | Manual scripting | Emergent from evaluation |
| Adaptability | Fixed structure | Dynamic branching |
| Tactical Depth | Current state only | Multiple moves ahead |
| Maintenance | High (explicit rules) | Low (evaluation tuning) |

### vs. LLM-driven AI
| Aspect | LLM AI | AAS |
|--------|--------|-----|
| Determinism | Stochastic | 100% reproducible |
| Latency | 100-1000ms | Sub-millisecond per frame |
| Planning | Token prediction | True tree search |
| Hallucination | Common | Zero (only legal actions) |
| Sovereignty | External API | Fully local |
| Esports Ready | No | Yes (replay-safe) |

---

## Features

### Core Capabilities
- **Depth-adaptive tree search** with asymmetric branching
- **Entropy-gradient** directed resource allocation
- **Time-bounded search** with deterministic yielding
- **Multi-agent coordination** through shared evaluation context
- **Blueprint exposure** for safe high-level control

### Determinism Guarantees
- Fixed-point arithmetic (no floating-point variance)
- Deterministic RNG with explicit state serialization
- Ordered containers for stable iteration
- Explicit tick ordering for multi-agent systems

### Performance
- Sustains **60+ FPS** with configurable frame budget
- **No game thread blocking** - incremental search
- Optimized transposition table with replacement policy
- Cache-friendly node layout (~96 bytes per node)

---

## Quick Start

### 1. Build Instructions

```bash
# Clone repository
git clone https://github.com/your-repo/QRATUM.git

# Navigate to AAS module
cd QRATUM/qratum_aas

# Generate project files (Windows)
"C:\Program Files\Epic Games\UE_5.4\Engine\Build\BatchFiles\GenerateProjectFiles.bat" QRATUM_AAS.uproject

# Build (Windows)
"C:\Program Files\Epic Games\UE_5.4\Engine\Build\BatchFiles\Build.bat" QRATUM_AAS Win64 Development

# Or open in Unreal Editor
# File -> Open Project -> QRATUM_AAS.uproject
```

### 2. Add AAS to Your Project

#### Blueprint Setup
1. Add `UQRATUMAASComponent` to your AI-controlled Actor
2. Configure search parameters in the Details panel
3. Bind to `OnPlanningComplete` event
4. Call `RequestPlan()` when you need an action

#### C++ Setup
```cpp
// In your AI Controller header
#include "Integration/AQRATUMAIController.h"

UCLASS()
class AMyTacticalAI : public AQRATUMAIController
{
    GENERATED_BODY()
    
protected:
    // Override to provide game state
    virtual bool CreateGameState_Implementation() override;
    
    // Override to execute planned actions
    virtual bool ExecuteAction_Implementation(const FQRATUMPlannedAction& Action) override;
};
```

### 3. Run the Demo

```cpp
#include "Demo/TacticalArenaDemo.h"

void RunDemo()
{
    QRATUM::FTacticalArenaDemo Demo;
    
    QRATUM::FArenaDemoConfig Config;
    Config.NumAgentsPerTeam = 3;
    Config.SearchDepth = 10;
    Config.SearchTimeMs = 100.0f;
    Config.bLogMoves = true;
    
    Demo.Initialize(Config);
    Demo.RunGame();
}
```

---

## Architecture

```
QRATUM/
├── Source/QRATUM/
│   ├── Public/
│   │   ├── QRATUMAIModule.h          # Module interface
│   │   ├── Core/
│   │   │   ├── AASNode.h             # Search tree nodes
│   │   │   ├── AASSearch.h           # Core search engine
│   │   │   ├── AASHeuristics.h       # Evaluation system
│   │   │   └── AASPlanner.h          # High-level planner
│   │   ├── Determinism/
│   │   │   ├── DeterministicTypes.h  # Fixed-point, RNG
│   │   │   └── DeterministicContainers.h
│   │   └── Integration/
│   │       ├── UQRATUMAASComponent.h # UE Component
│   │       ├── AQRATUMAIController.h # AI Controller
│   │       └── UQRATUMBlueprintLibrary.h
│   ├── Private/
│   │   └── [implementations]
│   └── Demo/
│       └── TacticalArenaDemo.h/cpp   # Example scenario
└── QRATUM.Build.cs
```

### Design Principles

1. **Architectural Separation**: Core AAS logic is engine-agnostic. Unreal-specific code only in Integration layer.

2. **Determinism First**: All operations use fixed-point math and deterministic containers. Identical inputs = identical outputs.

3. **Frame Safety**: Search never blocks the game thread. Configurable per-frame budget with yield points.

4. **Blueprint Discipline**: Only safe, high-level controls exposed. No internal state manipulation from Blueprints.

---

## Extending AAS

### For RTS Games
```cpp
// Implement IAASGameState for your units
class FRTSUnitState : public QRATUM::IAASGameState
{
    virtual void GetLegalActions(TArray<QRATUM::FAASAction>& OutActions) const override
    {
        // Move actions, attack actions, build actions, etc.
    }
    
    virtual TUniquePtr<IAASGameState> ApplyAction(const QRATUM::FAASAction& Action) const override
    {
        // Apply action to create new state
    }
};

// Add domain-specific heuristics
class FRTSHeuristics : public QRATUM::FAASHeuristics
{
public:
    FRTSHeuristics()
    {
        AddFeature({"ResourceAdvantage", FFixedPoint32::FromFloat(0.3f), ExtractResources});
        AddFeature({"UnitAdvantage", FFixedPoint32::FromFloat(0.3f), ExtractUnitCount});
        AddFeature({"TerritoryControl", FFixedPoint32::FromFloat(0.2f), ExtractTerritory});
        AddFeature({"TechAdvantage", FFixedPoint32::FromFloat(0.2f), ExtractTechLevel});
    }
};
```

### For Survival Games
```cpp
class FSurvivalState : public QRATUM::IAASGameState
{
    // Model: health, hunger, threats, resources, shelter status
    virtual void GetLegalActions(TArray<QRATUM::FAASAction>& OutActions) const override
    {
        // Gather, craft, flee, fight, rest, etc.
    }
};

class FSurvivalHeuristics : public QRATUM::FAASHeuristics
{
public:
    FSurvivalHeuristics()
    {
        AddFeature({"Survival", FFixedPoint32::FromFloat(0.4f), ExtractSurvivalChance});
        AddFeature({"Resources", FFixedPoint32::FromFloat(0.3f), ExtractResourceSecurity});
        AddFeature({"ThreatLevel", FFixedPoint32::FromFloat(0.2f), ExtractThreatLevel});
        AddFeature({"Exploration", FFixedPoint32::FromFloat(0.1f), ExtractExploration});
    }
};
```

### For Tactical FPS
```cpp
class FTacticalFPSState : public QRATUM::IAASGameState
{
    // Model: positions, cover, sightlines, ammo, health
    virtual void GetLegalActions(TArray<QRATUM::FAASAction>& OutActions) const override
    {
        // Move, shoot, reload, take cover, flank, suppress, etc.
    }
};

class FTacticalFPSHeuristics : public QRATUM::FAASHeuristics
{
public:
    FTacticalFPSHeuristics()
    {
        AddFeature({"SurvivalPriority", FFixedPoint32::FromFloat(0.25f), ...});
        AddFeature({"KillPotential", FFixedPoint32::FromFloat(0.25f), ...});
        AddFeature({"PositionalAdvantage", FFixedPoint32::FromFloat(0.2f), ...});
        AddFeature({"CoverUtilization", FFixedPoint32::FromFloat(0.15f), ...});
        AddFeature({"TeamSupport", FFixedPoint32::FromFloat(0.15f), ...});
    }
};
```

---

## API Reference

### Blueprint Functions

| Function | Description |
|----------|-------------|
| `RequestPlan(bUrgent)` | Start planning for current state |
| `GetPlannedAction()` | Get completed plan |
| `GetBestActionSoFar()` | Get current best during planning |
| `CancelPlanning()` | Stop ongoing search |
| `InvalidatePlan()` | Force re-planning |
| `SetDeterministicSeed(Seed)` | Set seed for replays |
| `ValidateDeterminism()` | QA: verify reproducibility |

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BaseDepth` | 10 | Base search depth |
| `TimeLimitMs` | 100 | Max search time |
| `FrameBudgetMs` | 2 | Per-frame budget |
| `bAdaptiveDepth` | true | Entropy-based depth |
| `bUseNullMove` | true | Null-move pruning |
| `bUseLMR` | true | Late-move reductions |
| `TranspositionTableSizeMB` | 64 | TT memory allocation |

---

## Determinism & Replay Support

### Setting Up Replay
```cpp
// At match start
int64 MatchSeed = UQRATUMBlueprintLibrary::GenerateMatchSeed(MatchID, PlayerSeed);
AASComponent->SetDeterministicSeed(MatchSeed);

// During replay - same seed = same decisions
ReplayAASComponent->SetDeterministicSeed(MatchSeed);
```

### Validating Determinism
```cpp
// In QA/testing
bool bDeterministic = UQRATUMBlueprintLibrary::ValidateDeterminism(AASComponent);
if (!bDeterministic)
{
    UE_LOG(LogQA, Error, TEXT("Determinism violation detected!"));
}
```

---

## Performance Tuning

### For 30 FPS Target (16.6ms frame budget)
```cpp
SearchConfig.FrameBudgetMs = 4.0f;  // 25% of frame
SearchConfig.BaseDepth = 8;
SearchConfig.bAdaptiveDepth = true;
```

### For 60 FPS Target (16.6ms frame budget)
```cpp
SearchConfig.FrameBudgetMs = 2.0f;  // 12% of frame
SearchConfig.BaseDepth = 10;
SearchConfig.bUseLMR = true;
SearchConfig.bUseNullMove = true;
```

### For 120 FPS Target (8.3ms frame budget)
```cpp
SearchConfig.FrameBudgetMs = 1.0f;  // 12% of frame
SearchConfig.BaseDepth = 6;
SearchConfig.bAdaptiveDepth = true;
SearchConfig.TranspositionTableSizeMB = 128; // More TT hits
```

---

## Debug & Visualization

### Enabling Debug Output
```cpp
// In your controller
AASComponent->bLogSearchStats = true;

// Or call manually
QRATUM::FAASDebugger::LogSearchStats(AASComponent->GetSearchStats());
```

### Getting Search Tree
```cpp
FString TreeDescription = QRATUM::FAASDebugger::DescribeSearchTree(
    AASComponent->GetPlanner()->GetRootNode(), 
    3  // Max depth
);
UE_LOG(LogAI, Log, TEXT("%s"), *TreeDescription);
```

### JSON Export
```cpp
FString JSON = UQRATUMBlueprintLibrary::GetSearchResultJSON(AASComponent);
// Send to telemetry, save to file, etc.
```

---

## License

Copyright QRATUM Platform. All Rights Reserved.

For licensing inquiries: licensing@qratum.io

---

## Unreal Fest Chicago 2026

This module was developed for presentation at **Unreal Fest Chicago 2026** as a technical showcase of deterministic AI systems for competitive gaming.

### Talk Abstract
*"Beyond Behavior Trees: Deterministic Game AI with Asymmetric Adaptive Search"*

Traditional behavior trees are reactive, requiring explicit scripting of every decision. LLM-driven AI promises flexibility but introduces non-determinism that breaks replays and esports integrity. This talk introduces AAS - a tree-search based approach adapted from tournament chess engines that delivers emergent tactical behavior while maintaining 100% reproducibility.

### Contact
- Technical Questions: ai-team@qratum.io
- Business Inquiries: partnerships@qratum.io
- GitHub: https://github.com/qratum/aas-unreal

---

*"The best move is not always obvious - but it should always be reproducible."*
