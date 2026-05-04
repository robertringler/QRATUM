/// File: Source/QRATUM/Public/Integration/UQRATUMBlueprintLibrary.h
// Copyright QRATUM Platform. All Rights Reserved.
// Unreal Engine integration - Blueprint Function Library

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "UQRATUMAASComponent.h"
#include "UQRATUMBlueprintLibrary.generated.h"

/**
 * UQRATUMBlueprintLibrary - Blueprint function library for AAS
 * 
 * Provides safe, high-level access to AAS functionality from Blueprints.
 * All exposed functions are:
 * - Deterministic: Same inputs = same outputs
 * - Replay-safe: No hidden state that breaks replays
 * - Frame-safe: Won't block the game thread
 * 
 * Exposed Functionality:
 * - EvaluateMove: Quick action evaluation
 * - RunSearchStep: Manual search control
 * - GetPlannedAction: Retrieve computed action
 * - GetSearchStats: Debug information
 * - ValidateDeterminism: QA helper
 * 
 * NOT Exposed (unsafe for Blueprint):
 * - Internal node access
 * - Direct heuristic modification
 * - Mutable search state
 */
UCLASS()
class QRATUM_API UQRATUMBlueprintLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    //--------------------------------------------------------------------------
    // Action Evaluation
    //--------------------------------------------------------------------------

    /**
     * Quick evaluation of a potential action.
     * 
     * Returns a score indicating how good this action is.
     * Does NOT perform full search - use for filtering/ordering.
     * 
     * @param Component - The AAS component
     * @param FromLocation - Source position/entity
     * @param ToLocation - Target position/entity
     * @return Score in range [-1, 1], higher is better
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Evaluation",
        meta = (DisplayName = "Quick Evaluate Action"))
    static float EvaluateAction(UQRATUMAASComponent* Component,
        int32 FromLocation, int32 ToLocation);

    /**
     * Compare two actions without full search.
     * 
     * @param Component - The AAS component
     * @param ActionA_From - First action source
     * @param ActionA_To - First action target
     * @param ActionB_From - Second action source
     * @param ActionB_To - Second action target
     * @return Positive if A is better, negative if B is better, 0 if equal
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Evaluation",
        meta = (DisplayName = "Compare Actions"))
    static float CompareActions(UQRATUMAASComponent* Component,
        int32 ActionA_From, int32 ActionA_To,
        int32 ActionB_From, int32 ActionB_To);

    //--------------------------------------------------------------------------
    // Search Control
    //--------------------------------------------------------------------------

    /**
     * Execute one step of search (manual control).
     * 
     * Use this for custom planning loops instead of auto-tick.
     * Set Component->bAutoTickPlanning = false first.
     * 
     * @param Component - The AAS component
     * @param OutProgress - Planning progress [0, 1]
     * @return True when search is complete
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Search",
        meta = (DisplayName = "Run Search Step"))
    static bool RunSearchStep(UQRATUMAASComponent* Component, float& OutProgress);

    /**
     * Get the best action found so far.
     * 
     * Can be called during search to get current best.
     * 
     * @param Component - The AAS component
     * @return Current best action (may change until search completes)
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Search",
        meta = (DisplayName = "Get Best Action So Far"))
    static FQRATUMPlannedAction GetBestActionSoFar(UQRATUMAASComponent* Component);

    /**
     * Get the final planned action.
     * 
     * Call after planning completes for deterministic result.
     * 
     * @param Component - The AAS component
     * @return Final planned action
     */
    UFUNCTION(BlueprintPure, Category = "QRATUM|Search",
        meta = (DisplayName = "Get Planned Action"))
    static FQRATUMPlannedAction GetPlannedAction(UQRATUMAASComponent* Component);

    //--------------------------------------------------------------------------
    // Statistics & Debug
    //--------------------------------------------------------------------------

    /**
     * Get search statistics.
     * 
     * @param Component - The AAS component
     * @return Search statistics from last/current search
     */
    UFUNCTION(BlueprintPure, Category = "QRATUM|Debug",
        meta = (DisplayName = "Get Search Statistics"))
    static FQRATUMSearchStats GetSearchStats(UQRATUMAASComponent* Component);

    /**
     * Get nodes per second (search performance).
     * 
     * @param Component - The AAS component
     * @return Nodes searched per second (NPS)
     */
    UFUNCTION(BlueprintPure, Category = "QRATUM|Debug",
        meta = (DisplayName = "Get Nodes Per Second"))
    static int64 GetNodesPerSecond(UQRATUMAASComponent* Component);

    /**
     * Get the principal variation (best move sequence).
     * 
     * @param Component - The AAS component
     * @param MaxMoves - Maximum moves to return
     * @return Array of planned actions representing best line
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Debug",
        meta = (DisplayName = "Get Principal Variation"))
    static TArray<FQRATUMPlannedAction> GetPrincipalVariation(
        UQRATUMAASComponent* Component, int32 MaxMoves = 5);

    /**
     * Validate determinism by running search twice.
     * 
     * Use in QA to verify AI reproducibility.
     * 
     * @param Component - The AAS component
     * @return True if results are identical
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Debug",
        meta = (DisplayName = "Validate Determinism"))
    static bool ValidateDeterminism(UQRATUMAASComponent* Component);

    /**
     * Get search result as JSON string.
     * 
     * Useful for logging, telemetry, or external analysis.
     * 
     * @param Component - The AAS component
     * @return JSON string with search result
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Debug",
        meta = (DisplayName = "Get Search Result JSON"))
    static FString GetSearchResultJSON(UQRATUMAASComponent* Component);

    //--------------------------------------------------------------------------
    // Configuration
    //--------------------------------------------------------------------------

    /**
     * Set search depth.
     * 
     * @param Component - The AAS component
     * @param Depth - New search depth (1-30)
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Configuration",
        meta = (DisplayName = "Set Search Depth"))
    static void SetSearchDepth(UQRATUMAASComponent* Component, int32 Depth);

    /**
     * Set time limit for search.
     * 
     * @param Component - The AAS component
     * @param TimeLimitMs - Time limit in milliseconds (0 = no limit)
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Configuration",
        meta = (DisplayName = "Set Time Limit"))
    static void SetTimeLimit(UQRATUMAASComponent* Component, float TimeLimitMs);

    /**
     * Set frame budget for incremental search.
     * 
     * @param Component - The AAS component
     * @param FrameBudgetMs - Budget per frame in milliseconds
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Configuration",
        meta = (DisplayName = "Set Frame Budget"))
    static void SetFrameBudget(UQRATUMAASComponent* Component, float FrameBudgetMs);

    //--------------------------------------------------------------------------
    // Determinism
    //--------------------------------------------------------------------------

    /**
     * Set deterministic seed for AI.
     * 
     * Call at match start with same seed on all clients for deterministic replay.
     * 
     * @param Component - The AAS component
     * @param Seed - 64-bit seed value
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Determinism",
        meta = (DisplayName = "Set Deterministic Seed"))
    static void SetDeterministicSeed(UQRATUMAASComponent* Component, int64 Seed);

    /**
     * Get current deterministic seed.
     * 
     * @param Component - The AAS component
     * @return Current seed value
     */
    UFUNCTION(BlueprintPure, Category = "QRATUM|Determinism",
        meta = (DisplayName = "Get Deterministic Seed"))
    static int64 GetDeterministicSeed(UQRATUMAASComponent* Component);

    /**
     * Generate a seed from match parameters.
     * 
     * Use this to create consistent seeds across clients.
     * 
     * @param MatchID - Unique match identifier
     * @param PlayerSeed - Player-specific seed (e.g., from lobby)
     * @return Combined seed value
     */
    UFUNCTION(BlueprintPure, Category = "QRATUM|Determinism",
        meta = (DisplayName = "Generate Match Seed"))
    static int64 GenerateMatchSeed(int64 MatchID, int64 PlayerSeed);

    //--------------------------------------------------------------------------
    // Utility
    //--------------------------------------------------------------------------

    /**
     * Convert action to human-readable string.
     * 
     * @param Action - The action to describe
     * @return Human-readable description
     */
    UFUNCTION(BlueprintPure, Category = "QRATUM|Utility",
        meta = (DisplayName = "Action To String"))
    static FString ActionToString(const FQRATUMPlannedAction& Action);

    /**
     * Check if an action is valid.
     * 
     * @param Action - The action to check
     * @return True if action is valid
     */
    UFUNCTION(BlueprintPure, Category = "QRATUM|Utility",
        meta = (DisplayName = "Is Action Valid"))
    static bool IsActionValid(const FQRATUMPlannedAction& Action);

    /**
     * Get QRATUM module version.
     * 
     * @return Version string
     */
    UFUNCTION(BlueprintPure, Category = "QRATUM|Utility",
        meta = (DisplayName = "Get QRATUM Version"))
    static FString GetQRATUMVersion();
};
