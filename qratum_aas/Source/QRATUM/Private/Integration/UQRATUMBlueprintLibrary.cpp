/// File: Source/QRATUM/Private/Integration/UQRATUMBlueprintLibrary.cpp
// Copyright QRATUM Platform. All Rights Reserved.
// Unreal Engine integration - Blueprint Function Library implementation

#include "Integration/UQRATUMBlueprintLibrary.h"
#include "Core/AASPlanner.h"

//------------------------------------------------------------------------------
// Action Evaluation
//------------------------------------------------------------------------------

float UQRATUMBlueprintLibrary::EvaluateAction(UQRATUMAASComponent* Component,
    int32 FromLocation, int32 ToLocation)
{
    if (!Component || !Component->GetPlanner())
    {
        return 0.0f;
    }

    // Create action
    QRATUM::FAASAction Action(FromLocation, ToLocation);
    
    // Quick evaluation using planner
    // Note: This requires a valid game state to be set
    // For now, return prior-based estimate
    return Action.Prior.ToFloat();
}

float UQRATUMBlueprintLibrary::CompareActions(UQRATUMAASComponent* Component,
    int32 ActionA_From, int32 ActionA_To,
    int32 ActionB_From, int32 ActionB_To)
{
    float ScoreA = EvaluateAction(Component, ActionA_From, ActionA_To);
    float ScoreB = EvaluateAction(Component, ActionB_From, ActionB_To);
    return ScoreA - ScoreB;
}

//------------------------------------------------------------------------------
// Search Control
//------------------------------------------------------------------------------

bool UQRATUMBlueprintLibrary::RunSearchStep(UQRATUMAASComponent* Component, 
    float& OutProgress)
{
    if (!Component)
    {
        OutProgress = 0.0f;
        return true;
    }

    // Note: This is a convenience wrapper. For custom search loops,
    // disable auto-tick and call this manually each frame.
    
    if (Component->IsPlanning())
    {
        FQRATUMSearchStats Stats = Component->GetSearchStats();
        OutProgress = FMath::Clamp(
            static_cast<float>(Stats.DepthReached) / 
            static_cast<float>(Component->SearchConfig.BaseDepth),
            0.0f, 1.0f);
        return false;
    }
    else if (Component->HasPlan())
    {
        OutProgress = 1.0f;
        return true;
    }
    else
    {
        OutProgress = 0.0f;
        return true; // Not planning, nothing to do
    }
}

FQRATUMPlannedAction UQRATUMBlueprintLibrary::GetBestActionSoFar(
    UQRATUMAASComponent* Component)
{
    if (!Component)
    {
        return FQRATUMPlannedAction();
    }
    return Component->GetBestActionSoFar();
}

FQRATUMPlannedAction UQRATUMBlueprintLibrary::GetPlannedAction(
    UQRATUMAASComponent* Component)
{
    if (!Component)
    {
        return FQRATUMPlannedAction();
    }
    return Component->GetPlannedAction();
}

//------------------------------------------------------------------------------
// Statistics & Debug
//------------------------------------------------------------------------------

FQRATUMSearchStats UQRATUMBlueprintLibrary::GetSearchStats(
    UQRATUMAASComponent* Component)
{
    if (!Component)
    {
        return FQRATUMSearchStats();
    }
    return Component->GetSearchStats();
}

int64 UQRATUMBlueprintLibrary::GetNodesPerSecond(UQRATUMAASComponent* Component)
{
    if (!Component)
    {
        return 0;
    }

    FQRATUMSearchStats Stats = Component->GetSearchStats();
    if (Stats.TimeMs > 0.0f)
    {
        return static_cast<int64>(Stats.NodesSearched / (Stats.TimeMs / 1000.0f));
    }
    return 0;
}

TArray<FQRATUMPlannedAction> UQRATUMBlueprintLibrary::GetPrincipalVariation(
    UQRATUMAASComponent* Component, int32 MaxMoves)
{
    TArray<FQRATUMPlannedAction> PV;
    
    if (!Component || !Component->GetPlanner())
    {
        return PV;
    }

    const QRATUM::FAASSearchResult& Result = Component->GetPlanner()->GetSearchStats();
    
    const int32 NumMoves = FMath::Min(Result.PrincipalVariation.Num(), MaxMoves);
    for (int32 i = 0; i < NumMoves; ++i)
    {
        FQRATUMPlannedAction Action;
        Action.From = Result.PrincipalVariation[i].From;
        Action.To = Result.PrincipalVariation[i].To;
        Action.TypeFlags = Result.PrincipalVariation[i].TypeFlags;
        Action.bIsValid = true;
        PV.Add(Action);
    }

    return PV;
}

bool UQRATUMBlueprintLibrary::ValidateDeterminism(UQRATUMAASComponent* Component)
{
    if (!Component)
    {
        return false;
    }
    return Component->ValidateDeterminism();
}

FString UQRATUMBlueprintLibrary::GetSearchResultJSON(UQRATUMAASComponent* Component)
{
    if (!Component || !Component->GetPlanner())
    {
        return TEXT("{}");
    }

    return QRATUM::FAASDebugger::SearchResultToJSON(
        Component->GetPlanner()->GetSearchStats());
}

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------

void UQRATUMBlueprintLibrary::SetSearchDepth(UQRATUMAASComponent* Component, 
    int32 Depth)
{
    if (!Component)
    {
        return;
    }
    Component->SearchConfig.BaseDepth = FMath::Clamp(Depth, 1, 30);
}

void UQRATUMBlueprintLibrary::SetTimeLimit(UQRATUMAASComponent* Component, 
    float TimeLimitMs)
{
    if (!Component)
    {
        return;
    }
    Component->SearchConfig.TimeLimitMs = FMath::Max(0.0f, TimeLimitMs);
}

void UQRATUMBlueprintLibrary::SetFrameBudget(UQRATUMAASComponent* Component, 
    float FrameBudgetMs)
{
    if (!Component)
    {
        return;
    }
    Component->SearchConfig.FrameBudgetMs = FMath::Clamp(FrameBudgetMs, 0.1f, 16.0f);
}

//------------------------------------------------------------------------------
// Determinism
//------------------------------------------------------------------------------

void UQRATUMBlueprintLibrary::SetDeterministicSeed(UQRATUMAASComponent* Component, 
    int64 Seed)
{
    if (!Component)
    {
        return;
    }
    Component->SetDeterministicSeed(Seed);
}

int64 UQRATUMBlueprintLibrary::GetDeterministicSeed(UQRATUMAASComponent* Component)
{
    if (!Component)
    {
        return 0;
    }
    return Component->GetDeterministicSeed();
}

int64 UQRATUMBlueprintLibrary::GenerateMatchSeed(int64 MatchID, int64 PlayerSeed)
{
    // Combine using hash
    uint64 Combined = QRATUM::HashCombine(
        static_cast<uint64>(MatchID),
        static_cast<uint64>(PlayerSeed)
    );
    return static_cast<int64>(Combined);
}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

FString UQRATUMBlueprintLibrary::ActionToString(const FQRATUMPlannedAction& Action)
{
    if (!Action.bIsValid)
    {
        return TEXT("(invalid action)");
    }

    return FString::Printf(
        TEXT("Action[%d->%d, flags=0x%X, conf=%.2f, val=%.3f, depth=%d]"),
        Action.From, Action.To, Action.TypeFlags,
        Action.Confidence, Action.ExpectedValue, Action.LookaheadDepth);
}

bool UQRATUMBlueprintLibrary::IsActionValid(const FQRATUMPlannedAction& Action)
{
    return Action.bIsValid;
}

FString UQRATUMBlueprintLibrary::GetQRATUMVersion()
{
    return TEXT("QRATUM AAS 1.0.0 - Unreal Fest Chicago 2026");
}
