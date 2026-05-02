/// File: Source/QRATUM/Private/Integration/UQRATUMAASComponent.cpp
// Copyright QRATUM Platform. All Rights Reserved.
// Unreal Engine integration - AAS Component implementation

#include "Integration/UQRATUMAASComponent.h"
#include "QRATUMAIModule.h"

//------------------------------------------------------------------------------
// FQRATUMSearchConfig Implementation
//------------------------------------------------------------------------------

QRATUM::FAASSearchConfig FQRATUMSearchConfig::ToInternalConfig() const
{
    QRATUM::FAASSearchConfig Config;
    Config.BaseDepth = BaseDepth;
    Config.MaxDepth = BaseDepth + 10;
    Config.TimeLimitMs = TimeLimitMs;
    Config.FrameBudgetMs = FrameBudgetMs;
    Config.bAdaptiveDepth = bAdaptiveDepth;
    Config.bUseNullMove = bUseNullMove;
    Config.bUseLMR = bUseLMR;
    Config.TranspositionTableSizeMB = TranspositionTableSizeMB;
    return Config;
}

//------------------------------------------------------------------------------
// FQRATUMPlannedAction Implementation
//------------------------------------------------------------------------------

FQRATUMPlannedAction::FQRATUMPlannedAction(const QRATUM::FAASPlannedAction& InternalAction)
    : From(InternalAction.PrimaryAction.From)
    , To(InternalAction.PrimaryAction.To)
    , TypeFlags(InternalAction.PrimaryAction.TypeFlags)
    , Confidence(InternalAction.Confidence.ToFloat())
    , ExpectedValue(InternalAction.ExpectedValue.ToFloat())
    , LookaheadDepth(InternalAction.LookaheadDepth)
    , bIsValid(InternalAction.IsValid())
{
}

//------------------------------------------------------------------------------
// FQRATUMSearchStats Implementation
//------------------------------------------------------------------------------

FQRATUMSearchStats::FQRATUMSearchStats(const QRATUM::FAASSearchResult& InternalResult)
    : NodesSearched(InternalResult.NodesSearched)
    , DepthReached(InternalResult.DepthReached)
    , TimeMs(static_cast<float>(InternalResult.TimeMs))
    , TTHitRate(InternalResult.TTHitRate)
    , Entropy(InternalResult.Entropy.ToFloat())
    , bCompleted(InternalResult.bCompleted)
{
}

//------------------------------------------------------------------------------
// UQRATUMAASComponent Implementation
//------------------------------------------------------------------------------

UQRATUMAASComponent::UQRATUMAASComponent()
    : bAutoTickPlanning(true)
    , bLogSearchStats(false)
    , bPlanRequested(false)
    , DeterministicSeed(0x51415455) // "QATU"
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.bStartWithTickEnabled = true;
    PrimaryComponentTick.TickGroup = TG_PrePhysics;
}

void UQRATUMAASComponent::BeginPlay()
{
    Super::BeginPlay();

    // Initialize planner
    Planner = MakeUnique<QRATUM::FAASPlanner>();

    // Create default heuristics if none provided
    if (!Heuristics)
    {
        Heuristics = MakeShared<QRATUM::FAASHeuristics>();
    }

    // Apply configuration
    ApplyConfiguration();

    // Initialize planner with heuristics
    Planner->Initialize(Heuristics, SearchConfig.ToInternalConfig());

    UE_LOG(LogTemp, Log, TEXT("[QRATUM] AAS Component initialized on %s"), 
        *GetOwner()->GetName());
}

void UQRATUMAASComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    // Clean up
    if (Planner)
    {
        Planner->Reset();
    }

    Super::EndPlay(EndPlayReason);
}

void UQRATUMAASComponent::TickComponent(float DeltaTime, ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    // Auto-advance deterministic tick counter
    if (FQRATUMAIModule::IsAvailable())
    {
        FQRATUMAIModule::Get().AdvanceTickCounter();
    }

    // Execute planning step if requested and auto-tick enabled
    if (bAutoTickPlanning && bPlanRequested)
    {
        ExecutePlanningStep();
    }
}

void UQRATUMAASComponent::RequestPlan(bool bUrgent)
{
    if (!Planner || !CurrentGameState)
    {
        UE_LOG(LogTemp, Warning, 
            TEXT("[QRATUM] Cannot request plan: Planner or GameState not initialized"));
        return;
    }

    // Invalidate any existing plan
    Planner->InvalidatePlan();

    // Set up planning context
    PlanningContext.CurrentState = CurrentGameState.Get();
    PlanningContext.AvailableTimeMs = SearchConfig.TimeLimitMs;
    PlanningContext.FrameBudgetMs = SearchConfig.FrameBudgetMs;
    PlanningContext.bUrgent = bUrgent;

    bPlanRequested = true;

    UE_LOG(LogTemp, Verbose, TEXT("[QRATUM] Plan requested (urgent: %s)"),
        bUrgent ? TEXT("true") : TEXT("false"));
}

void UQRATUMAASComponent::CancelPlanning()
{
    if (Planner)
    {
        Planner->InvalidatePlan();
    }
    bPlanRequested = false;
}

FQRATUMPlannedAction UQRATUMAASComponent::GetPlannedAction() const
{
    if (!Planner)
    {
        return FQRATUMPlannedAction();
    }
    return FQRATUMPlannedAction(Planner->GetPlannedAction());
}

FQRATUMPlannedAction UQRATUMAASComponent::GetBestActionSoFar() const
{
    if (!Planner)
    {
        return FQRATUMPlannedAction();
    }
    return FQRATUMPlannedAction(Planner->GetBestActionSoFar());
}

bool UQRATUMAASComponent::IsPlanning() const
{
    return Planner ? Planner->IsPlanning() : false;
}

bool UQRATUMAASComponent::HasPlan() const
{
    return Planner ? Planner->HasPlan() : false;
}

FQRATUMSearchStats UQRATUMAASComponent::GetSearchStats() const
{
    if (!Planner)
    {
        return FQRATUMSearchStats();
    }
    return FQRATUMSearchStats(Planner->GetSearchStats());
}

void UQRATUMAASComponent::InvalidatePlan()
{
    if (Planner)
    {
        Planner->InvalidatePlan();
    }
    bPlanRequested = false;
}

void UQRATUMAASComponent::SetDeterministicSeed(int64 Seed)
{
    DeterministicSeed = Seed;
    
    // Update global seed in module
    if (FQRATUMAIModule::IsAvailable())
    {
        FQRATUMAIModule::Get().SetGlobalSeed(static_cast<uint64>(Seed));
    }
    
    // Reset planner with new seed
    if (Planner)
    {
        Planner->Reset();
    }

    UE_LOG(LogTemp, Log, TEXT("[QRATUM] Deterministic seed set to: 0x%016llX"), Seed);
}

int64 UQRATUMAASComponent::GetDeterministicSeed() const
{
    return DeterministicSeed;
}

bool UQRATUMAASComponent::ValidateDeterminism()
{
    if (!Planner || !CurrentGameState)
    {
        UE_LOG(LogTemp, Warning, 
            TEXT("[QRATUM] Cannot validate determinism: Planner or GameState not initialized"));
        return false;
    }

    return QRATUM::FAASDebugger::ValidateDeterminism(*Planner, *CurrentGameState);
}

void UQRATUMAASComponent::SetGameState(TUniquePtr<QRATUM::IAASGameState> InState)
{
    CurrentGameState = MoveTemp(InState);
    
    // Invalidate plan when state changes
    if (Planner)
    {
        Planner->InvalidatePlan();
    }
}

void UQRATUMAASComponent::UpdateGameState(TUniquePtr<QRATUM::IAASGameState> NewState)
{
    SetGameState(MoveTemp(NewState));
}

void UQRATUMAASComponent::SetHeuristics(TSharedPtr<QRATUM::FAASHeuristics> InHeuristics)
{
    Heuristics = InHeuristics;
    
    // Re-initialize planner if it exists
    if (Planner)
    {
        Planner->Initialize(Heuristics, SearchConfig.ToInternalConfig());
    }
}

void UQRATUMAASComponent::ExecutePlanningStep()
{
    if (!Planner || !CurrentGameState)
    {
        return;
    }

    // Update context in case config changed
    PlanningContext.CurrentState = CurrentGameState.Get();
    PlanningContext.FrameBudgetMs = SearchConfig.FrameBudgetMs;

    // Execute one planning step
    bool bPlanComplete = Planner->PlanStep(PlanningContext);

    // Fire progress event
    QRATUM::FAASPlannedAction BestSoFar = Planner->GetBestActionSoFar();
    float Progress = bPlanComplete ? 1.0f : 
        static_cast<float>(Planner->GetSearchStats().DepthReached) / 
        static_cast<float>(SearchConfig.BaseDepth);
    
    OnPlanningProgress.Broadcast(Progress, FQRATUMPlannedAction(BestSoFar));

    // Check if planning complete
    if (bPlanComplete)
    {
        bPlanRequested = false;

        QRATUM::FAASPlannedAction FinalAction = Planner->GetPlannedAction();
        
        // Log stats if enabled
        if (bLogSearchStats)
        {
            QRATUM::FAASDebugger::LogSearchStats(Planner->GetSearchStats());
        }

        // Fire completion event
        OnPlanningComplete.Broadcast(FQRATUMPlannedAction(FinalAction));

        UE_LOG(LogTemp, Verbose, TEXT("[QRATUM] Planning complete. Action: %d->%d, Value: %.3f"),
            FinalAction.PrimaryAction.From, FinalAction.PrimaryAction.To,
            FinalAction.ExpectedValue.ToFloat());
    }
}

void UQRATUMAASComponent::ApplyConfiguration()
{
    if (Planner)
    {
        Planner->SetConfig(SearchConfig.ToInternalConfig());
    }
}
