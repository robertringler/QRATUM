/// File: Source/QRATUM/Private/Core/AASPlanner.cpp
// Copyright QRATUM Platform. All Rights Reserved.
// Asymmetric Adaptive Search - High-level planner implementation

#include "Core/AASPlanner.h"
#include "HAL/PlatformTime.h"
#include "Serialization/JsonSerializer.h"
#include "Dom/JsonObject.h"

namespace QRATUM
{
    //--------------------------------------------------------------------------
    // FAASPlanner Implementation
    //--------------------------------------------------------------------------

    FAASPlanner::FAASPlanner()
        : bIsPlanning(false)
        , bPlanValid(false)
        , LastStateHash(0)
        , TotalPlanningTimeMs(0.0)
        , PlanningIterations(0)
    {
        Search = MakeUnique<FAASSearch>();
    }

    FAASPlanner::~FAASPlanner()
    {
        InvalidatePlan();
    }

    void FAASPlanner::Initialize(TSharedPtr<FAASHeuristics> InHeuristics, 
        const FAASSearchConfig& InConfig)
    {
        Heuristics = InHeuristics;
        Search->SetConfig(InConfig);
        Reset();
    }

    bool FAASPlanner::PlanStep(const FAASPlanningContext& Context)
    {
        if (!Context.CurrentState || !Heuristics)
        {
            return false;
        }

        const uint64 CurrentStateHash = Context.CurrentState->GetStateHash();

        // Check if state changed (invalidate plan)
        if (CurrentStateHash != LastStateHash)
        {
            InvalidatePlan();
            LastStateHash = CurrentStateHash;
        }

        // If we already have a valid plan, return immediately
        if (bPlanValid && !bIsPlanning)
        {
            return true;
        }

        // Begin planning if not already in progress
        if (!bIsPlanning)
        {
            BeginPlanning(Context);
        }

        // Execute one search step
        FAASSearchResult StepResult;
        const double StepStartTime = FPlatformTime::Seconds() * 1000.0;

        // Configure search for this step
        FAASSearchConfig StepConfig = Search->GetConfig();
        StepConfig.FrameBudgetMs = Context.FrameBudgetMs;
        StepConfig.TimeLimitMs = Context.AvailableTimeMs;
        
        // Adjust depth based on urgency
        if (Context.bUrgent)
        {
            StepConfig.BaseDepth = FMath::Max(4, StepConfig.BaseDepth - 2);
        }

        Search->SetConfig(StepConfig);

        bool bSearchComplete = Search->SearchStep(*Context.CurrentState, *Heuristics, StepResult);

        const double StepTimeMs = (FPlatformTime::Seconds() * 1000.0) - StepStartTime;
        TotalPlanningTimeMs += StepTimeMs;
        ++PlanningIterations;

        // Update current plan with best found so far
        if (StepResult.BestAction.From != 0 || StepResult.BestAction.To != 0)
        {
            CurrentPlan.PrimaryAction = StepResult.BestAction;
            CurrentPlan.ExpectedValue = StepResult.Evaluation;
            CurrentPlan.LookaheadDepth = StepResult.DepthReached;
            CurrentPlan.PlanningTimeMs = TotalPlanningTimeMs;
            CurrentPlan.Confidence = ComputeConfidence(StepResult);
            
            // Extract alternative actions
            ExtractAlternatives(CurrentPlan.Alternatives);
        }

        // Check if planning is complete
        if (bSearchComplete)
        {
            bIsPlanning = false;
            bPlanValid = CurrentPlan.IsValid();
            
            UE_LOG(LogTemp, Verbose, 
                TEXT("[QRATUM] Planning complete. Depth: %d, Confidence: %.2f, Time: %.1fms"),
                CurrentPlan.LookaheadDepth, CurrentPlan.Confidence.ToFloat(), 
                CurrentPlan.PlanningTimeMs);

            return true;
        }

        return false;
    }

    void FAASPlanner::BeginPlanning(const FAASPlanningContext& Context)
    {
        bIsPlanning = true;
        bPlanValid = false;
        TotalPlanningTimeMs = 0.0;
        PlanningIterations = 0;
        CurrentPlan = FAASPlannedAction();

        // Clear move ordering heuristics if state changed significantly
        Heuristics->ClearMoveOrderingData();

        // Begin search
        Search->BeginSearch(*Context.CurrentState, *Heuristics);
    }

    FAASPlannedAction FAASPlanner::GetPlannedAction() const
    {
        return CurrentPlan;
    }

    FAASPlannedAction FAASPlanner::GetBestActionSoFar() const
    {
        if (bIsPlanning)
        {
            // Get current best from ongoing search
            const FAASSearchResult& Result = Search->GetLastResult();
            FAASPlannedAction BestSoFar;
            BestSoFar.PrimaryAction = Result.BestAction;
            BestSoFar.ExpectedValue = Result.Evaluation;
            BestSoFar.LookaheadDepth = Result.DepthReached;
            BestSoFar.PlanningTimeMs = TotalPlanningTimeMs;
            BestSoFar.Confidence = ComputeConfidence(Result);
            return BestSoFar;
        }
        return CurrentPlan;
    }

    FFixedPoint32 FAASPlanner::EvaluateAction(const IAASGameState& State, 
        const FAASAction& Action) const
    {
        if (!Heuristics)
        {
            return FFixedPoint32::Zero();
        }

        // Apply action and evaluate resulting state
        TUniquePtr<IAASGameState> ChildState = State.ApplyAction(Action);
        if (!ChildState)
        {
            return FFixedPoint32::Min();
        }

        return -Heuristics->Evaluate(*ChildState);
    }

    void FAASPlanner::InvalidatePlan()
    {
        if (bIsPlanning)
        {
            Search->CancelSearch();
        }
        bIsPlanning = false;
        bPlanValid = false;
        CurrentPlan = FAASPlannedAction();
    }

    const FAASSearchResult& FAASPlanner::GetSearchStats() const
    {
        return Search->GetLastResult();
    }

    void FAASPlanner::SetConfig(const FAASSearchConfig& InConfig)
    {
        Search->SetConfig(InConfig);
    }

    const FAASSearchConfig& FAASPlanner::GetConfig() const
    {
        return Search->GetConfig();
    }

    void FAASPlanner::Reset()
    {
        InvalidatePlan();
        Search->Reset();
        LastStateHash = 0;
        TotalPlanningTimeMs = 0.0;
        PlanningIterations = 0;
    }

    FFixedPoint32 FAASPlanner::ComputeConfidence(const FAASSearchResult& Result) const
    {
        // Confidence is based on:
        // - Search depth achieved
        // - Evaluation stability (how much it changed)
        // - Time spent (more time = more confident)

        float DepthFactor = FMath::Clamp(Result.DepthReached / 10.0f, 0.0f, 1.0f);
        float TimeFactor = FMath::Clamp(static_cast<float>(Result.TimeMs) / 1000.0f, 0.0f, 1.0f);
        
        // Higher absolute evaluation = more confident
        float EvalFactor = FMath::Clamp(FMath::Abs(Result.Evaluation.ToFloat()), 0.0f, 1.0f);
        
        // Combine factors
        float Confidence = 0.4f * DepthFactor + 0.3f * TimeFactor + 0.3f * EvalFactor;
        
        return FFixedPoint32::FromFloat(FMath::Clamp(Confidence, 0.0f, 1.0f));
    }

    void FAASPlanner::ExtractAlternatives(TArray<FAASAction>& OutAlternatives) const
    {
        OutAlternatives.Empty();

        const FAASNode* Root = Search->GetRootNode();
        if (!Root)
        {
            return;
        }

        // Get children sorted by value (skip best which is primary)
        TArray<TPair<FFixedPoint32, FAASAction>> ChildValues;
        
        for (const FAASNode* Child : Root->GetChildren())
        {
            if (Child->GetAction() != CurrentPlan.PrimaryAction)
            {
                ChildValues.Emplace(Child->GetValue(), Child->GetAction());
            }
        }

        // Sort by value (highest first)
        Algo::Sort(ChildValues, [](const TPair<FFixedPoint32, FAASAction>& A, 
                                   const TPair<FFixedPoint32, FAASAction>& B)
        {
            return A.Key > B.Key;
        });

        // Extract top alternatives
        const int32 MaxAlternatives = 3;
        for (int32 i = 0; i < FMath::Min(ChildValues.Num(), MaxAlternatives); ++i)
        {
            OutAlternatives.Add(ChildValues[i].Value);
        }
    }

    //--------------------------------------------------------------------------
    // FAASBehaviorTreeCompare Implementation
    //--------------------------------------------------------------------------

    void FAASBehaviorTreeCompare::LogComparison(const FAASAction& AASAction, 
        const FAASAction& BTAction, const FString& Context)
    {
        const bool bSameTarget = AASAction.To == BTAction.To;
        const bool bSameSource = AASAction.From == BTAction.From;

        UE_LOG(LogTemp, Log, TEXT("[QRATUM] Decision Comparison - %s"), *Context);
        UE_LOG(LogTemp, Log, TEXT("  AAS: %d -> %d (flags: 0x%X)"), 
            AASAction.From, AASAction.To, AASAction.TypeFlags);
        UE_LOG(LogTemp, Log, TEXT("  BT:  %d -> %d (flags: 0x%X)"), 
            BTAction.From, BTAction.To, BTAction.TypeFlags);
        UE_LOG(LogTemp, Log, TEXT("  Match: Source=%s, Target=%s"),
            bSameSource ? TEXT("Yes") : TEXT("No"),
            bSameTarget ? TEXT("Yes") : TEXT("No"));

        if (!bSameTarget || !bSameSource)
        {
            UE_LOG(LogTemp, Log, 
                TEXT("  Analysis: AAS uses predictive tree search while BT uses reactive rules."));
            UE_LOG(LogTemp, Log,
                TEXT("  AAS may see tactical opportunities that BT cannot anticipate."));
        }
    }

    FFixedPoint32 FAASBehaviorTreeCompare::ComputeDifference(const FAASAction& AASAction, 
        const FAASAction& BTAction)
    {
        // Compute Manhattan distance-like difference
        int32 SourceDiff = (AASAction.From != BTAction.From) ? 1 : 0;
        int32 TargetDiff = FMath::Abs(static_cast<int32>(AASAction.To) - 
                                       static_cast<int32>(BTAction.To));
        int32 TypeDiff = (AASAction.TypeFlags != BTAction.TypeFlags) ? 1 : 0;

        // Normalize to [0, 1]
        float Difference = (SourceDiff + FMath::Min(TargetDiff, 10) / 10.0f + TypeDiff) / 3.0f;
        
        return FFixedPoint32::FromFloat(FMath::Clamp(Difference, 0.0f, 1.0f));
    }

    //--------------------------------------------------------------------------
    // FAASDebugger Implementation
    //--------------------------------------------------------------------------

    FString FAASDebugger::DescribeSearchTree(const FAASNode* Root, int32 MaxDepth)
    {
        if (!Root)
        {
            return TEXT("(empty tree)");
        }

        FString Result;
        TFunction<void(const FAASNode*, int32, FString&)> DescribeNode = 
            [&DescribeNode, MaxDepth](const FAASNode* Node, int32 Depth, FString& Out)
        {
            if (!Node || Depth > MaxDepth)
            {
                return;
            }

            FString Indent;
            for (int32 i = 0; i < Depth; ++i)
            {
                Indent += TEXT("  ");
            }

            const FAASAction& Action = Node->GetAction();
            Out += FString::Printf(TEXT("%s[%d->%d] Value: %.3f, Visits: %d\n"),
                *Indent, Action.From, Action.To, 
                Node->GetValue().ToFloat(), Node->GetVisitCount());

            for (const FAASNode* Child : Node->GetChildren())
            {
                DescribeNode(Child, Depth + 1, Out);
            }
        };

        Result += FString::Printf(TEXT("Search Tree (Root: hash=0x%016llX, depth=%d)\n"),
            Root->GetStateHash(), Root->GetDepth());
        
        for (const FAASNode* Child : Root->GetChildren())
        {
            DescribeNode(Child, 0, Result);
        }

        return Result;
    }

    FString FAASDebugger::SearchResultToJSON(const FAASSearchResult& Result)
    {
        TSharedPtr<FJsonObject> JsonObj = MakeShared<FJsonObject>();

        // Best action
        TSharedPtr<FJsonObject> ActionObj = MakeShared<FJsonObject>();
        ActionObj->SetNumberField(TEXT("from"), Result.BestAction.From);
        ActionObj->SetNumberField(TEXT("to"), Result.BestAction.To);
        ActionObj->SetNumberField(TEXT("typeFlags"), Result.BestAction.TypeFlags);
        JsonObj->SetObjectField(TEXT("bestAction"), ActionObj);

        // Statistics
        JsonObj->SetNumberField(TEXT("evaluation"), Result.Evaluation.ToFloat());
        JsonObj->SetNumberField(TEXT("nodesSearched"), Result.NodesSearched);
        JsonObj->SetNumberField(TEXT("depthReached"), Result.DepthReached);
        JsonObj->SetNumberField(TEXT("timeMs"), Result.TimeMs);
        JsonObj->SetNumberField(TEXT("ttHitRate"), Result.TTHitRate);
        JsonObj->SetBoolField(TEXT("completed"), Result.bCompleted);
        JsonObj->SetNumberField(TEXT("entropy"), Result.Entropy.ToFloat());

        // Principal variation
        TArray<TSharedPtr<FJsonValue>> PVArray;
        for (const FAASAction& PVAction : Result.PrincipalVariation)
        {
            TSharedPtr<FJsonObject> PVObj = MakeShared<FJsonObject>();
            PVObj->SetNumberField(TEXT("from"), PVAction.From);
            PVObj->SetNumberField(TEXT("to"), PVAction.To);
            PVArray.Add(MakeShared<FJsonValueObject>(PVObj));
        }
        JsonObj->SetArrayField(TEXT("principalVariation"), PVArray);

        FString OutputString;
        TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutputString);
        FJsonSerializer::Serialize(JsonObj.ToSharedRef(), Writer);
        
        return OutputString;
    }

    void FAASDebugger::LogSearchStats(const FAASSearchResult& Result)
    {
        UE_LOG(LogTemp, Log, TEXT("[QRATUM] Search Statistics:"));
        UE_LOG(LogTemp, Log, TEXT("  Best Action: %d -> %d"), 
            Result.BestAction.From, Result.BestAction.To);
        UE_LOG(LogTemp, Log, TEXT("  Evaluation: %.4f"), Result.Evaluation.ToFloat());
        UE_LOG(LogTemp, Log, TEXT("  Nodes Searched: %lld"), Result.NodesSearched);
        UE_LOG(LogTemp, Log, TEXT("  Depth Reached: %d"), Result.DepthReached);
        UE_LOG(LogTemp, Log, TEXT("  Time: %.2f ms"), Result.TimeMs);
        UE_LOG(LogTemp, Log, TEXT("  TT Hit Rate: %.1f%%"), Result.TTHitRate * 100.0f);
        UE_LOG(LogTemp, Log, TEXT("  Entropy: %.3f"), Result.Entropy.ToFloat());
        UE_LOG(LogTemp, Log, TEXT("  Completed: %s"), Result.bCompleted ? TEXT("Yes") : TEXT("No"));
        
        if (!Result.PrincipalVariation.IsEmpty())
        {
            FString PVStr;
            for (int32 i = 0; i < FMath::Min(Result.PrincipalVariation.Num(), 5); ++i)
            {
                const FAASAction& A = Result.PrincipalVariation[i];
                PVStr += FString::Printf(TEXT(" %d->%d"), A.From, A.To);
            }
            UE_LOG(LogTemp, Log, TEXT("  PV:%s"), *PVStr);
        }
    }

    bool FAASDebugger::ValidateDeterminism(FAASPlanner& Planner, const IAASGameState& State)
    {
        // Run search twice
        FAASPlanningContext Context;
        Context.CurrentState = &State;
        Context.AvailableTimeMs = 1000.0;
        Context.FrameBudgetMs = 1000.0; // Single frame

        Planner.Reset();
        while (!Planner.PlanStep(Context)) {}
        FAASPlannedAction Result1 = Planner.GetPlannedAction();

        Planner.Reset();
        while (!Planner.PlanStep(Context)) {}
        FAASPlannedAction Result2 = Planner.GetPlannedAction();

        // Compare results
        bool bDeterministic = 
            Result1.PrimaryAction == Result2.PrimaryAction &&
            Result1.ExpectedValue == Result2.ExpectedValue &&
            Result1.LookaheadDepth == Result2.LookaheadDepth;

        if (!bDeterministic)
        {
            UE_LOG(LogTemp, Error, TEXT("[QRATUM] DETERMINISM VIOLATION DETECTED!"));
            UE_LOG(LogTemp, Error, TEXT("  Run 1: Action %d->%d, Value %.4f, Depth %d"),
                Result1.PrimaryAction.From, Result1.PrimaryAction.To,
                Result1.ExpectedValue.ToFloat(), Result1.LookaheadDepth);
            UE_LOG(LogTemp, Error, TEXT("  Run 2: Action %d->%d, Value %.4f, Depth %d"),
                Result2.PrimaryAction.From, Result2.PrimaryAction.To,
                Result2.ExpectedValue.ToFloat(), Result2.LookaheadDepth);
        }
        else
        {
            UE_LOG(LogTemp, Log, TEXT("[QRATUM] Determinism validation PASSED"));
        }

        return bDeterministic;
    }
}
