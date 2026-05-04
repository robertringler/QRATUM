/// File: Source/QRATUM/Public/Integration/UQRATUMAASComponent.h
// Copyright QRATUM Platform. All Rights Reserved.
// Unreal Engine integration - AAS Component

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Core/AASPlanner.h"
#include "UQRATUMAASComponent.generated.h"

/**
 * Search configuration exposed to Blueprints
 */
USTRUCT(BlueprintType)
struct QRATUM_API FQRATUMSearchConfig
{
    GENERATED_BODY()

    /** Base search depth */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Search", meta = (ClampMin = "1", ClampMax = "30"))
    int32 BaseDepth = 10;

    /** Time limit in milliseconds (0 = no limit) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Search", meta = (ClampMin = "0"))
    float TimeLimitMs = 100.0f;

    /** Per-frame time budget for incremental search */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Search", meta = (ClampMin = "0.1", ClampMax = "16.0"))
    float FrameBudgetMs = 2.0f;

    /** Enable adaptive depth based on position entropy */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Search")
    bool bAdaptiveDepth = true;

    /** Enable null-move pruning optimization */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Search|Advanced")
    bool bUseNullMove = true;

    /** Enable late-move reductions */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Search|Advanced")
    bool bUseLMR = true;

    /** Transposition table size in MB */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Search|Memory", meta = (ClampMin = "1", ClampMax = "256"))
    int32 TranspositionTableSizeMB = 64;

    /** Convert to internal config */
    QRATUM::FAASSearchConfig ToInternalConfig() const;
};

/**
 * Planned action exposed to Blueprints
 */
USTRUCT(BlueprintType)
struct QRATUM_API FQRATUMPlannedAction
{
    GENERATED_BODY()

    /** Source location/entity ID */
    UPROPERTY(BlueprintReadOnly, Category = "Action")
    int32 From = 0;

    /** Target location/entity ID */
    UPROPERTY(BlueprintReadOnly, Category = "Action")
    int32 To = 0;

    /** Action type flags */
    UPROPERTY(BlueprintReadOnly, Category = "Action")
    int32 TypeFlags = 0;

    /** Confidence in this action [0, 1] */
    UPROPERTY(BlueprintReadOnly, Category = "Action")
    float Confidence = 0.0f;

    /** Expected value of executing this action */
    UPROPERTY(BlueprintReadOnly, Category = "Action")
    float ExpectedValue = 0.0f;

    /** Number of moves lookahead used */
    UPROPERTY(BlueprintReadOnly, Category = "Action")
    int32 LookaheadDepth = 0;

    /** Whether this action is valid */
    UPROPERTY(BlueprintReadOnly, Category = "Action")
    bool bIsValid = false;

    FQRATUMPlannedAction() = default;

    /** Construct from internal action */
    explicit FQRATUMPlannedAction(const QRATUM::FAASPlannedAction& InternalAction);
};

/**
 * Search statistics exposed to Blueprints
 */
USTRUCT(BlueprintType)
struct QRATUM_API FQRATUMSearchStats
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "Stats")
    int64 NodesSearched = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Stats")
    int32 DepthReached = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Stats")
    float TimeMs = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Stats")
    float TTHitRate = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Stats")
    float Entropy = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Stats")
    bool bCompleted = false;

    FQRATUMSearchStats() = default;

    /** Construct from internal result */
    explicit FQRATUMSearchStats(const QRATUM::FAASSearchResult& InternalResult);
};

/**
 * Delegate fired when planning completes
 */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnPlanningComplete, FQRATUMPlannedAction, PlannedAction);

/**
 * Delegate fired on each planning step (for UI updates)
 */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnPlanningProgress, float, Progress, FQRATUMPlannedAction, BestSoFar);

/**
 * UQRATUMAASComponent - Asymmetric Adaptive Search Component
 * 
 * Add this component to any Actor that needs AI-driven tactical planning.
 * The component handles:
 * - Incremental search within frame budget
 * - State management for the AI planner
 * - Blueprint-friendly interface
 * - Deterministic replay support
 * 
 * Usage:
 * 1. Add component to your AI-controlled actor
 * 2. Implement a game state adapter (see documentation)
 * 3. Call UpdateState() when world state changes
 * 4. Call RequestPlan() to begin planning
 * 5. Check GetPlannedAction() or bind to OnPlanningComplete
 * 
 * The search runs incrementally each tick, respecting frame budget
 * to maintain 60+ FPS.
 */
UCLASS(ClassGroup=(QRATUM), meta=(BlueprintSpawnableComponent))
class QRATUM_API UQRATUMAASComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UQRATUMAASComponent();

    //--------------------------------------------------------------------------
    // Configuration
    //--------------------------------------------------------------------------

    /** Search configuration */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "QRATUM|Configuration")
    FQRATUMSearchConfig SearchConfig;

    /** Whether to automatically tick planning */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "QRATUM|Configuration")
    bool bAutoTickPlanning = true;

    /** Debug: log search statistics */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "QRATUM|Debug")
    bool bLogSearchStats = false;

    //--------------------------------------------------------------------------
    // Events
    //--------------------------------------------------------------------------

    /** Fired when planning completes */
    UPROPERTY(BlueprintAssignable, Category = "QRATUM|Events")
    FOnPlanningComplete OnPlanningComplete;

    /** Fired on each planning step */
    UPROPERTY(BlueprintAssignable, Category = "QRATUM|Events")
    FOnPlanningProgress OnPlanningProgress;

    //--------------------------------------------------------------------------
    // Blueprint Interface
    //--------------------------------------------------------------------------

    /**
     * Request a new plan for the current state.
     * Planning will proceed incrementally over subsequent ticks.
     * 
     * @param bUrgent - If true, use faster but shallower search
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Planning")
    void RequestPlan(bool bUrgent = false);

    /**
     * Cancel ongoing planning.
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Planning")
    void CancelPlanning();

    /**
     * Get the current planned action.
     * Returns an invalid action if no plan is ready.
     */
    UFUNCTION(BlueprintPure, Category = "QRATUM|Planning")
    FQRATUMPlannedAction GetPlannedAction() const;

    /**
     * Get the best action found so far (even if planning not complete).
     */
    UFUNCTION(BlueprintPure, Category = "QRATUM|Planning")
    FQRATUMPlannedAction GetBestActionSoFar() const;

    /**
     * Check if planning is in progress.
     */
    UFUNCTION(BlueprintPure, Category = "QRATUM|Planning")
    bool IsPlanning() const;

    /**
     * Check if a valid plan is ready.
     */
    UFUNCTION(BlueprintPure, Category = "QRATUM|Planning")
    bool HasPlan() const;

    /**
     * Get search statistics from last planning cycle.
     */
    UFUNCTION(BlueprintPure, Category = "QRATUM|Planning")
    FQRATUMSearchStats GetSearchStats() const;

    /**
     * Invalidate current plan (force re-planning on next request).
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Planning")
    void InvalidatePlan();

    /**
     * Set the deterministic seed for this component's planner.
     * Call at session start for replay support.
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Determinism")
    void SetDeterministicSeed(int64 Seed);

    /**
     * Get the current deterministic seed.
     */
    UFUNCTION(BlueprintPure, Category = "QRATUM|Determinism")
    int64 GetDeterministicSeed() const;

    /**
     * Validate determinism by running search twice.
     * Returns true if results are identical.
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Debug")
    bool ValidateDeterminism();

    //--------------------------------------------------------------------------
    // State Management (called from game code)
    //--------------------------------------------------------------------------

    /**
     * Set the game state adapter.
     * Must be called before planning can begin.
     */
    void SetGameState(TUniquePtr<QRATUM::IAASGameState> InState);

    /**
     * Update the game state (call when world changes).
     */
    void UpdateGameState(TUniquePtr<QRATUM::IAASGameState> NewState);

    /**
     * Set the heuristics for evaluation.
     */
    void SetHeuristics(TSharedPtr<QRATUM::FAASHeuristics> InHeuristics);

    /**
     * Get the internal planner (for advanced usage).
     */
    QRATUM::FAASPlanner* GetPlanner() const { return Planner.Get(); }

protected:
    //--------------------------------------------------------------------------
    // UActorComponent Interface
    //--------------------------------------------------------------------------

    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, 
        FActorComponentTickFunction* ThisTickFunction) override;

private:
    /** The internal planner */
    TUniquePtr<QRATUM::FAASPlanner> Planner;

    /** Current game state */
    TUniquePtr<QRATUM::IAASGameState> CurrentGameState;

    /** Heuristics for evaluation */
    TSharedPtr<QRATUM::FAASHeuristics> Heuristics;

    /** Planning context */
    QRATUM::FAASPlanningContext PlanningContext;

    /** Whether a plan has been requested */
    bool bPlanRequested;

    /** Deterministic seed */
    int64 DeterministicSeed;

    /** Execute one planning step (called from tick) */
    void ExecutePlanningStep();

    /** Apply configuration to internal planner */
    void ApplyConfiguration();
};
