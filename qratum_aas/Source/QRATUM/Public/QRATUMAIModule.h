/// File: Source/QRATUM/Public/QRATUMAIModule.h
// Copyright QRATUM Platform. All Rights Reserved.
// Asymmetric Adaptive Search (AAS) Module for Unreal Engine 5
// Unreal Fest Chicago 2026

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

/**
 * QRATUM AI Module
 * 
 * Production-grade, deterministic, frame-safe AI framework implementing
 * Asymmetric Adaptive Search (AAS). Designed for competitive game AI
 * that requires:
 * 
 * - Deterministic behavior for replays and esports
 * - Real-time performance at 60+ FPS
 * - Deep tactical planning (20+ ply equivalent)
 * - Multi-agent coordination
 * 
 * Architecture Rationale vs. Alternatives:
 * 
 * Why AAS beats Behavior Trees:
 * - BTs are reactive, not predictive - they respond to states, not plan ahead
 * - AAS searches future game states, enabling look-ahead tactical decisions
 * - BTs require explicit scripting of every behavior; AAS discovers optimal actions
 * - AAS adapts branching dynamically; BTs have fixed structure
 * 
 * Why AAS beats LLM-driven AI:
 * - Determinism: LLMs are inherently stochastic; AAS is 100% reproducible
 * - Latency: LLM inference is 100-1000ms; AAS runs sub-millisecond per frame slice
 * - Tactical depth: LLMs lack true planning; AAS performs actual tree search
 * - No hallucination: AAS only considers legal actions from the game state
 * - Sovereignty: AAS runs locally with no external API dependency
 * 
 * Key Features:
 * - Depth-adaptive tree search with asymmetric branching
 * - Entropy-gradient directed resource allocation
 * - Time-bounded search with deterministic yielding
 * - Multi-agent coordination through shared evaluation context
 * - Blueprint exposure for safe high-level control
 */
class QRATUM_API FQRATUMAIModule : public IModuleInterface
{
public:
    /** IModuleInterface implementation */
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

    /**
     * Get the singleton instance of this module.
     * @return Reference to the QRATUM AI module
     */
    static FQRATUMAIModule& Get();

    /**
     * Check if the module has been loaded.
     * @return True if the module is loaded and initialized
     */
    static bool IsAvailable();

    /**
     * Get the global deterministic seed used for all AAS operations.
     * @return Current global seed value
     */
    uint64 GetGlobalSeed() const { return GlobalSeed; }

    /**
     * Set the global deterministic seed.
     * Call this at session start for deterministic replay support.
     * @param NewSeed - The seed value to use for all AAS operations
     */
    void SetGlobalSeed(uint64 NewSeed);

    /**
     * Get the current frame's deterministic tick counter.
     * Used for stable execution ordering within a frame.
     * @return Current tick counter value
     */
    uint64 GetDeterministicTickCounter() const { return DeterministicTickCounter; }

    /**
     * Advance the deterministic tick counter.
     * Called automatically by the module; do not call manually.
     */
    void AdvanceTickCounter();

private:
    /** Global seed for deterministic operations */
    uint64 GlobalSeed;

    /** Per-frame tick counter for execution ordering */
    uint64 DeterministicTickCounter;

    /** Whether the module has been initialized */
    bool bIsInitialized;
};
