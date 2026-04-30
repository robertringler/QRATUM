/// File: Source/QRATUM/Private/Integration/AQRATUMAIController.cpp
// Copyright QRATUM Platform. All Rights Reserved.
// Unreal Engine integration - AAS AI Controller implementation

#include "Integration/AQRATUMAIController.h"
#include "DrawDebugHelpers.h"
#include "Engine/World.h"

AQRATUMAIController::AQRATUMAIController()
    : StateUpdateInterval(0.1f)
    , ActionCooldown(0.5f)
    , bAutoStartPlanning(true)
    , bShowDebugInfo(false)
    , AASComponent(nullptr)
    , CurrentState(EQRATUMAIState::Idle)
    , TimeSinceStateUpdate(0.0f)
    , TimeSinceLastAction(0.0f)
    , bIsPaused(false)
{
    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.bStartWithTickEnabled = true;
}

void AQRATUMAIController::OnPossess(APawn* InPawn)
{
    Super::OnPossess(InPawn);

    if (!InPawn)
    {
        return;
    }

    // Create and attach AAS component
    AASComponent = NewObject<UQRATUMAASComponent>(this, TEXT("AASComponent"));
    if (AASComponent)
    {
        AASComponent->RegisterComponent();
        AASComponent->SearchConfig = SearchConfig;

        // Bind to planning complete event
        AASComponent->OnPlanningComplete.AddDynamic(this, 
            &AQRATUMAIController::OnPlanningCompleteInternal);

        UE_LOG(LogTemp, Log, TEXT("[QRATUM] AI Controller possessed %s"), *InPawn->GetName());
    }

    // Auto-start if configured
    if (bAutoStartPlanning)
    {
        StartAI();
    }
}

void AQRATUMAIController::OnUnPossess()
{
    StopAI();

    if (AASComponent)
    {
        AASComponent->OnPlanningComplete.RemoveDynamic(this, 
            &AQRATUMAIController::OnPlanningCompleteInternal);
        AASComponent->DestroyComponent();
        AASComponent = nullptr;
    }

    Super::OnUnPossess();
}

void AQRATUMAIController::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);

    if (bIsPaused || CurrentState == EQRATUMAIState::Disabled)
    {
        return;
    }

    UpdateAIStateMachine(DeltaSeconds);

    if (bShowDebugInfo)
    {
        DrawDebugInfo();
    }
}

void AQRATUMAIController::StartAI()
{
    if (CurrentState == EQRATUMAIState::Disabled)
    {
        SetAIState(EQRATUMAIState::Idle);
        bIsPaused = false;
        
        // Initial state creation
        CreateGameState();
        
        UE_LOG(LogTemp, Log, TEXT("[QRATUM] AI started"));
    }
}

void AQRATUMAIController::StopAI()
{
    SetAIState(EQRATUMAIState::Disabled);
    
    if (AASComponent)
    {
        AASComponent->CancelPlanning();
    }
    
    UE_LOG(LogTemp, Log, TEXT("[QRATUM] AI stopped"));
}

void AQRATUMAIController::PauseAI()
{
    bIsPaused = true;
    
    if (AASComponent)
    {
        AASComponent->CancelPlanning();
    }
    
    UE_LOG(LogTemp, Log, TEXT("[QRATUM] AI paused"));
}

void AQRATUMAIController::ResumeAI()
{
    bIsPaused = false;
    UE_LOG(LogTemp, Log, TEXT("[QRATUM] AI resumed"));
}

void AQRATUMAIController::ForceReplan()
{
    if (AASComponent)
    {
        AASComponent->InvalidatePlan();
    }
    
    SetAIState(EQRATUMAIState::Idle);
}

void AQRATUMAIController::SetDeterministicSeed(int64 Seed)
{
    if (AASComponent)
    {
        AASComponent->SetDeterministicSeed(Seed);
    }
}

bool AQRATUMAIController::CreateGameState_Implementation()
{
    // Default implementation: no state created
    // Override in subclass to provide actual game state
    UE_LOG(LogTemp, Warning, 
        TEXT("[QRATUM] CreateGameState not implemented - override in subclass"));
    return false;
}

bool AQRATUMAIController::ExecuteAction_Implementation(const FQRATUMPlannedAction& Action)
{
    // Default implementation: no action executed
    // Override in subclass to handle actual game actions
    UE_LOG(LogTemp, Warning, 
        TEXT("[QRATUM] ExecuteAction not implemented - override in subclass"));
    return false;
}

bool AQRATUMAIController::IsActionComplete_Implementation()
{
    // Default: actions complete immediately
    return true;
}

bool AQRATUMAIController::IsUrgentSituation_Implementation()
{
    // Default: not urgent
    return false;
}

void AQRATUMAIController::UpdateAIStateMachine(float DeltaSeconds)
{
    TimeSinceStateUpdate += DeltaSeconds;
    TimeSinceLastAction += DeltaSeconds;

    switch (CurrentState)
    {
    case EQRATUMAIState::Idle:
        // Check for state updates
        if (TimeSinceStateUpdate >= StateUpdateInterval)
        {
            TimeSinceStateUpdate = 0.0f;
            
            // Update game state
            if (CreateGameState())
            {
                // Check if we can plan
                if (AASComponent && TimeSinceLastAction >= ActionCooldown)
                {
                    bool bUrgent = IsUrgentSituation();
                    AASComponent->RequestPlan(bUrgent);
                    SetAIState(EQRATUMAIState::Planning);
                }
            }
        }
        break;

    case EQRATUMAIState::Planning:
        // Planning happens automatically via component tick
        // State transitions via OnPlanningCompleteInternal callback
        break;

    case EQRATUMAIState::Executing:
        // Check if action is complete
        if (IsActionComplete())
        {
            SetAIState(EQRATUMAIState::Idle);
            TimeSinceLastAction = 0.0f;
        }
        break;

    case EQRATUMAIState::Waiting:
        // Waiting for cooldown
        if (TimeSinceLastAction >= ActionCooldown)
        {
            SetAIState(EQRATUMAIState::Idle);
        }
        break;

    case EQRATUMAIState::Disabled:
        // Do nothing
        break;
    }
}

void AQRATUMAIController::SetAIState(EQRATUMAIState NewState)
{
    if (CurrentState != NewState)
    {
        EQRATUMAIState OldState = CurrentState;
        CurrentState = NewState;
        OnAIStateChanged.Broadcast(OldState, NewState);

        UE_LOG(LogTemp, Verbose, TEXT("[QRATUM] AI State: %d -> %d"), 
            static_cast<int32>(OldState), static_cast<int32>(NewState));
    }
}

void AQRATUMAIController::OnPlanningCompleteInternal(FQRATUMPlannedAction PlannedAction)
{
    if (CurrentState != EQRATUMAIState::Planning)
    {
        return;
    }

    if (!PlannedAction.bIsValid)
    {
        UE_LOG(LogTemp, Warning, TEXT("[QRATUM] Planning produced invalid action"));
        SetAIState(EQRATUMAIState::Idle);
        return;
    }

    // Store current action
    CurrentAction = PlannedAction;

    // Fire event
    OnActionExecute.Broadcast(PlannedAction);

    // Execute the action
    if (ExecuteAction(PlannedAction))
    {
        SetAIState(EQRATUMAIState::Executing);
        
        UE_LOG(LogTemp, Verbose, TEXT("[QRATUM] Executing action: %d->%d (conf: %.2f)"),
            PlannedAction.From, PlannedAction.To, PlannedAction.Confidence);
    }
    else
    {
        // Action couldn't be executed, go back to idle
        SetAIState(EQRATUMAIState::Waiting);
        
        UE_LOG(LogTemp, Warning, TEXT("[QRATUM] Failed to execute action: %d->%d"),
            PlannedAction.From, PlannedAction.To);
    }
}

void AQRATUMAIController::DrawDebugInfo() const
{
    APawn* ControlledPawn = GetPawn();
    if (!ControlledPawn)
    {
        return;
    }

    const FVector PawnLocation = ControlledPawn->GetActorLocation();

    // State string
    FString StateStr;
    switch (CurrentState)
    {
    case EQRATUMAIState::Idle:     StateStr = TEXT("IDLE"); break;
    case EQRATUMAIState::Planning: StateStr = TEXT("PLANNING"); break;
    case EQRATUMAIState::Executing:StateStr = TEXT("EXECUTING"); break;
    case EQRATUMAIState::Waiting:  StateStr = TEXT("WAITING"); break;
    case EQRATUMAIState::Disabled: StateStr = TEXT("DISABLED"); break;
    }

    // Draw state above pawn
    DrawDebugString(GetWorld(), PawnLocation + FVector(0, 0, 100),
        FString::Printf(TEXT("QRATUM: %s"), *StateStr),
        nullptr, FColor::Cyan, 0.0f, true);

    // Draw search stats if planning
    if (AASComponent && CurrentState == EQRATUMAIState::Planning)
    {
        FQRATUMSearchStats Stats = AASComponent->GetSearchStats();
        DrawDebugString(GetWorld(), PawnLocation + FVector(0, 0, 80),
            FString::Printf(TEXT("Depth: %d  Nodes: %lld"), 
                Stats.DepthReached, Stats.NodesSearched),
            nullptr, FColor::Yellow, 0.0f, true);
    }

    // Draw current action if executing
    if (CurrentState == EQRATUMAIState::Executing && CurrentAction.bIsValid)
    {
        DrawDebugString(GetWorld(), PawnLocation + FVector(0, 0, 60),
            FString::Printf(TEXT("Action: %d->%d (%.2f)"),
                CurrentAction.From, CurrentAction.To, CurrentAction.Confidence),
            nullptr, FColor::Green, 0.0f, true);
    }
}
