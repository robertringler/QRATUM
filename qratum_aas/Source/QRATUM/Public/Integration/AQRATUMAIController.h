/// File: Source/QRATUM/Public/Integration/AQRATUMAIController.h
// Copyright QRATUM Platform. All Rights Reserved.
// Unreal Engine integration - AAS AI Controller

#pragma once

#include "CoreMinimal.h"
#include "AIController.h"
#include "UQRATUMAASComponent.h"
#include "AQRATUMAIController.generated.h"

/**
 * AI state for tracking behavior
 */
UENUM(BlueprintType)
enum class EQRATUMAIState : uint8
{
    /** AI is idle, no planning in progress */
    Idle,
    /** AI is planning next action */
    Planning,
    /** AI is executing planned action */
    Executing,
    /** AI is waiting (cooldown, animation, etc.) */
    Waiting,
    /** AI is disabled */
    Disabled
};

/**
 * Delegate fired when AI state changes
 */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnAIStateChanged, EQRATUMAIState, OldState, EQRATUMAIState, NewState);

/**
 * Delegate fired when an action is about to be executed
 */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnActionExecute, FQRATUMPlannedAction, Action);

/**
 * AQRATUMAIController - AI Controller using Asymmetric Adaptive Search
 * 
 * This controller replaces traditional Behavior Tree-based AI with
 * AAS-driven tactical planning. Key advantages:
 * 
 * - Predictive: Looks ahead multiple moves instead of reacting to current state
 * - Adaptive: Automatically adjusts strategy based on situation
 * - Deterministic: Same situation = same decision (crucial for replays)
 * - Emergent: Complex behaviors emerge from evaluation, not scripting
 * 
 * Usage:
 * 1. Create a Blueprint subclass of this controller
 * 2. Implement CreateGameState() to provide current tactical state
 * 3. Implement ExecuteAction() to translate planned action to game commands
 * 4. Assign this controller to your AI pawn
 * 
 * The controller automatically handles:
 * - State polling and update
 * - Planning requests and timing
 * - Action execution flow
 * - Deterministic replay support
 */
UCLASS()
class QRATUM_API AQRATUMAIController : public AAIController
{
    GENERATED_BODY()

public:
    AQRATUMAIController();

    //--------------------------------------------------------------------------
    // Configuration
    //--------------------------------------------------------------------------

    /** Search configuration (passed to component) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "QRATUM|Configuration")
    FQRATUMSearchConfig SearchConfig;

    /** How often to check for state updates (seconds) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "QRATUM|Configuration")
    float StateUpdateInterval = 0.1f;

    /** Minimum time between action executions (seconds) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "QRATUM|Configuration")
    float ActionCooldown = 0.5f;

    /** Whether to automatically start planning on possess */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "QRATUM|Configuration")
    bool bAutoStartPlanning = true;

    /** Debug visualization */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "QRATUM|Debug")
    bool bShowDebugInfo = false;

    //--------------------------------------------------------------------------
    // Events
    //--------------------------------------------------------------------------

    /** Fired when AI state changes */
    UPROPERTY(BlueprintAssignable, Category = "QRATUM|Events")
    FOnAIStateChanged OnAIStateChanged;

    /** Fired when an action is about to be executed */
    UPROPERTY(BlueprintAssignable, Category = "QRATUM|Events")
    FOnActionExecute OnActionExecute;

    //--------------------------------------------------------------------------
    // Blueprint Interface
    //--------------------------------------------------------------------------

    /**
     * Start AI processing.
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Control")
    void StartAI();

    /**
     * Stop AI processing.
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Control")
    void StopAI();

    /**
     * Pause AI (temporarily stop, can resume).
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Control")
    void PauseAI();

    /**
     * Resume AI after pause.
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Control")
    void ResumeAI();

    /**
     * Get current AI state.
     */
    UFUNCTION(BlueprintPure, Category = "QRATUM|Control")
    EQRATUMAIState GetAIState() const { return CurrentState; }

    /**
     * Force immediate re-planning.
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Control")
    void ForceReplan();

    /**
     * Get the AAS component.
     */
    UFUNCTION(BlueprintPure, Category = "QRATUM|Control")
    UQRATUMAASComponent* GetAASComponent() const { return AASComponent; }

    /**
     * Set deterministic seed for this controller.
     */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Determinism")
    void SetDeterministicSeed(int64 Seed);

    //--------------------------------------------------------------------------
    // Overridable Methods
    //--------------------------------------------------------------------------

    /**
     * Create game state representation.
     * 
     * Override in Blueprint or C++ to provide tactical state.
     * This is called whenever state needs to be updated.
     */
    UFUNCTION(BlueprintNativeEvent, Category = "QRATUM|Override")
    bool CreateGameState();
    virtual bool CreateGameState_Implementation();

    /**
     * Execute a planned action.
     * 
     * Override to translate the planned action to game commands.
     * Return true if action was successfully initiated.
     */
    UFUNCTION(BlueprintNativeEvent, Category = "QRATUM|Override")
    bool ExecuteAction(const FQRATUMPlannedAction& Action);
    virtual bool ExecuteAction_Implementation(const FQRATUMPlannedAction& Action);

    /**
     * Check if current action execution is complete.
     * 
     * Override to provide action completion status.
     * Return true when the action has finished executing.
     */
    UFUNCTION(BlueprintNativeEvent, Category = "QRATUM|Override")
    bool IsActionComplete();
    virtual bool IsActionComplete_Implementation();

    /**
     * Get urgency level for current situation.
     * 
     * Override to indicate when faster planning is needed.
     * Return true if situation is urgent (enemy nearby, low health, etc.)
     */
    UFUNCTION(BlueprintNativeEvent, Category = "QRATUM|Override")
    bool IsUrgentSituation();
    virtual bool IsUrgentSituation_Implementation();

protected:
    //--------------------------------------------------------------------------
    // AAIController Interface
    //--------------------------------------------------------------------------

    virtual void OnPossess(APawn* InPawn) override;
    virtual void OnUnPossess() override;
    virtual void Tick(float DeltaSeconds) override;

    //--------------------------------------------------------------------------
    // Internal Methods
    //--------------------------------------------------------------------------

    /** Update AI state machine */
    void UpdateAIStateMachine(float DeltaSeconds);

    /** Set new AI state */
    void SetAIState(EQRATUMAIState NewState);

    /** Handle planning complete callback */
    UFUNCTION()
    void OnPlanningCompleteInternal(FQRATUMPlannedAction PlannedAction);

    /** Draw debug visualization */
    void DrawDebugInfo() const;

private:
    /** The AAS component */
    UPROPERTY()
    UQRATUMAASComponent* AASComponent;

    /** Current AI state */
    EQRATUMAIState CurrentState;

    /** Time since last state update */
    float TimeSinceStateUpdate;

    /** Time since last action */
    float TimeSinceLastAction;

    /** Currently executing action */
    FQRATUMPlannedAction CurrentAction;

    /** Whether AI is paused */
    bool bIsPaused;
};
