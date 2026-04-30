/// File: Source/QRATUM/Private/Core/AASSearch.cpp
// Copyright QRATUM Platform. All Rights Reserved.
// Asymmetric Adaptive Search - Core search engine implementation

#include "Core/AASSearch.h"
#include "HAL/PlatformTime.h"
#include "Algo/Sort.h"

namespace QRATUM
{
    // Evaluation bounds - margins prevent overflow during score adjustments
    static constexpr int32 EVAL_SAFETY_MARGIN = 1000;  // Buffer to prevent overflow
    static constexpr int32 EVAL_INF = FFixedPoint32::Max().RawValue - EVAL_SAFETY_MARGIN;
    static constexpr int32 EVAL_MATE = EVAL_INF - EVAL_SAFETY_MARGIN;  // Mate scores distinct from infinity

    //--------------------------------------------------------------------------
    // FAASSearch Implementation
    //--------------------------------------------------------------------------

    FAASSearch::FAASSearch()
        : CachedHeuristics(nullptr)
        , bIsSearching(false)
        , bShouldCancel(false)
        , CurrentDepth(0)
        , NodesSearched(0)
        , SearchStartTime(0.0)
    {
        TranspositionTable = MakeUnique<FAASTranspositionTable>(Config.TranspositionTableSizeMB);
    }

    FAASSearch::FAASSearch(const FAASSearchConfig& InConfig)
        : Config(InConfig)
        , CachedHeuristics(nullptr)
        , bIsSearching(false)
        , bShouldCancel(false)
        , CurrentDepth(0)
        , NodesSearched(0)
        , SearchStartTime(0.0)
    {
        TranspositionTable = MakeUnique<FAASTranspositionTable>(Config.TranspositionTableSizeMB);
    }

    FAASSearch::~FAASSearch()
    {
        CancelSearch();
    }

    void FAASSearch::SetConfig(const FAASSearchConfig& InConfig)
    {
        if (!bIsSearching)
        {
            Config = InConfig;
            
            // Recreate TT if size changed
            if (TranspositionTable == nullptr || 
                Config.TranspositionTableSizeMB != InConfig.TranspositionTableSizeMB)
            {
                TranspositionTable = MakeUnique<FAASTranspositionTable>(InConfig.TranspositionTableSizeMB);
            }
        }
    }

    void FAASSearch::Reset()
    {
        CancelSearch();
        TranspositionTable->Clear();
        RootNode.Reset();
        CachedRootState.Reset();
        CachedHeuristics = nullptr;
        LastResult = FAASSearchResult();
        CurrentPV.Empty();
    }

    void FAASSearch::CancelSearch()
    {
        bShouldCancel = true;
        bIsSearching = false;
    }

    FAASSearchResult FAASSearch::Search(const IAASGameState& RootState, const FAASHeuristics& Heuristics)
    {
        BeginSearch(RootState, Heuristics);

        FAASSearchResult Result;
        while (!SearchStep(RootState, Heuristics, Result))
        {
            // Continue until complete or cancelled
        }

        return Result;
    }

    void FAASSearch::BeginSearch(const IAASGameState& RootState, const FAASHeuristics& Heuristics)
    {
        bIsSearching = true;
        bShouldCancel = false;
        CurrentDepth = 1;
        NodesSearched = 0;
        SearchStartTime = FPlatformTime::Seconds() * 1000.0;
        
        CachedRootState = RootState.Clone();
        CachedHeuristics = &Heuristics;
        
        // Initialize root node
        RootNode = MakeUnique<FAASNode>();
        RootNode->SetStateHash(RootState.GetStateHash());
        
        // Clear previous PV
        CurrentPV.Empty();
        
        // Initialize result
        LastResult = FAASSearchResult();
        LastResult.Entropy = Heuristics.CalculateEntropy(RootState);
        
        UE_LOG(LogTemp, Verbose, TEXT("[QRATUM] Search started. Entropy: %.3f"), 
            LastResult.Entropy.ToFloat());
    }

    bool FAASSearch::SearchStep(const IAASGameState& RootState, const FAASHeuristics& Heuristics, 
        FAASSearchResult& OutResult)
    {
        if (!bIsSearching || bShouldCancel)
        {
            OutResult = LastResult;
            bIsSearching = false;
            return true;
        }

        const double StepStartTime = FPlatformTime::Seconds() * 1000.0;

        // Get mutable copy of move orderer
        FAASMoveOrderer MoveOrderer = Heuristics.GetMoveOrderer();

        // Iterative deepening with aspiration windows
        while (CurrentDepth <= Config.MaxDepth && !ShouldStop())
        {
            // Check frame budget
            const double StepElapsed = (FPlatformTime::Seconds() * 1000.0) - StepStartTime;
            if (StepElapsed >= Config.FrameBudgetMs && Config.FrameBudgetMs > 0)
            {
                // Yield to next frame
                OutResult = LastResult;
                return false;
            }

            // Calculate adaptive depth based on entropy
            int32 EffectiveDepth = CurrentDepth;
            if (Config.bAdaptiveDepth)
            {
                EffectiveDepth = GetAdaptiveDepth(LastResult.Entropy);
                EffectiveDepth = FMath::Max(EffectiveDepth, CurrentDepth);
            }

            // Execute search at this depth
            FFixedPoint32 Value = SearchRoot(RootState, EffectiveDepth, MoveOrderer);

            if (ShouldStop())
            {
                break;
            }

            // Update result
            LastResult.DepthReached = CurrentDepth;
            LastResult.Evaluation = Value;
            LastResult.NodesSearched = NodesSearched;
            LastResult.TimeMs = GetElapsedMs();
            LastResult.TTHitRate = TranspositionTable->GetHitRate();

            // Extract PV from search
            if (RootNode)
            {
                RootNode->GetPrincipalVariation(CurrentPV, 20);
                LastResult.PrincipalVariation = CurrentPV;
                
                if (!CurrentPV.IsEmpty())
                {
                    LastResult.BestAction = CurrentPV[0];
                }
            }

            // Age history heuristics for next iteration
            MoveOrderer.AgeHistory();

            ++CurrentDepth;
        }

        // Search complete
        LastResult.bCompleted = !bShouldCancel;
        OutResult = LastResult;
        bIsSearching = false;

        UE_LOG(LogTemp, Verbose, 
            TEXT("[QRATUM] Search complete. Depth: %d, Nodes: %lld, Time: %.1fms, Eval: %.3f"),
            LastResult.DepthReached, LastResult.NodesSearched, 
            LastResult.TimeMs, LastResult.Evaluation.ToFloat());

        return true;
    }

    FFixedPoint32 FAASSearch::SearchRoot(const IAASGameState& State, int32 Depth, 
        FAASMoveOrderer& MoveOrderer)
    {
        TArray<FAASAction> LegalActions;
        State.GetLegalActions(LegalActions);

        if (LegalActions.IsEmpty())
        {
            return State.GetTerminalValue();
        }

        // Single move: return immediately
        if (LegalActions.Num() == 1)
        {
            LastResult.BestAction = LegalActions[0];
            TUniquePtr<IAASGameState> ChildState = State.ApplyAction(LegalActions[0]);
            return -CachedHeuristics->Evaluate(*ChildState);
        }

        // Order moves using PV from previous iteration
        FAASAction* HashMove = CurrentPV.Num() > 0 ? &CurrentPV[0] : nullptr;
        MoveOrderer.OrderMoves(LegalActions, State, 0, HashMove);

        // Aspiration window search
        FFixedPoint32 Alpha = FFixedPoint32::FromInt(-EVAL_INF);
        FFixedPoint32 Beta = FFixedPoint32::FromInt(EVAL_INF);

        if (Config.bUseAspirationWindows && CurrentDepth > 1 && LastResult.Evaluation.RawValue != 0)
        {
            Alpha = LastResult.Evaluation - Config.AspirationWindow;
            Beta = LastResult.Evaluation + Config.AspirationWindow;
        }

        FAASAction BestAction = LegalActions[0];
        FFixedPoint32 BestValue = FFixedPoint32::FromInt(-EVAL_INF);
        bool bFailedLow = false;
        bool bFailedHigh = false;

        for (int32 MoveIndex = 0; MoveIndex < LegalActions.Num(); ++MoveIndex)
        {
            if (ShouldStop())
            {
                break;
            }

            const FAASAction& Action = LegalActions[MoveIndex];
            TUniquePtr<IAASGameState> ChildState = State.ApplyAction(Action);

            FFixedPoint32 Value;

            if (MoveIndex == 0)
            {
                // Full window search for first move
                Value = -AlphaBeta(*ChildState, Depth - 1, -Beta, -Alpha, 1, MoveOrderer, false);
            }
            else
            {
                // Principal variation search: null window first
                Value = -AlphaBeta(*ChildState, Depth - 1, 
                    FFixedPoint32(-(Alpha.RawValue + 1)), -Alpha, 1, MoveOrderer, false);

                // Re-search if fails high
                if (Value > Alpha && Value < Beta)
                {
                    Value = -AlphaBeta(*ChildState, Depth - 1, -Beta, -Alpha, 1, MoveOrderer, false);
                }
            }

            if (Value > BestValue)
            {
                BestValue = Value;
                BestAction = Action;

                // Update tree
                if (RootNode)
                {
                    FAASNode* ChildNode = RootNode->FindChild(Action);
                    if (!ChildNode)
                    {
                        ChildNode = RootNode->AddChild(Action);
                    }
                    ChildNode->SetValue(Value);
                    ChildNode->SetStateHash(ChildState->GetStateHash());
                }
            }

            if (Value > Alpha)
            {
                Alpha = Value;
            }

            if (Alpha >= Beta)
            {
                MoveOrderer.RecordKiller(Action, 0);
                MoveOrderer.RecordHistory(Action, Depth);
                break;
            }
        }

        // Handle aspiration window failures
        if (Config.bUseAspirationWindows)
        {
            if (BestValue <= LastResult.Evaluation - Config.AspirationWindow)
            {
                bFailedLow = true;
            }
            if (BestValue >= LastResult.Evaluation + Config.AspirationWindow)
            {
                bFailedHigh = true;
            }

            // Re-search with full window if aspiration failed
            if (bFailedLow || bFailedHigh)
            {
                Alpha = FFixedPoint32::FromInt(-EVAL_INF);
                Beta = FFixedPoint32::FromInt(EVAL_INF);
                BestValue = FFixedPoint32::FromInt(-EVAL_INF);

                for (int32 MoveIndex = 0; MoveIndex < LegalActions.Num(); ++MoveIndex)
                {
                    if (ShouldStop())
                    {
                        break;
                    }

                    const FAASAction& Action = LegalActions[MoveIndex];
                    TUniquePtr<IAASGameState> ChildState = State.ApplyAction(Action);

                    FFixedPoint32 Value = -AlphaBeta(*ChildState, Depth - 1, -Beta, -Alpha, 1, MoveOrderer, false);

                    if (Value > BestValue)
                    {
                        BestValue = Value;
                        BestAction = Action;
                    }

                    if (Value > Alpha)
                    {
                        Alpha = Value;
                    }

                    if (Alpha >= Beta)
                    {
                        MoveOrderer.RecordKiller(Action, 0);
                        MoveOrderer.RecordHistory(Action, Depth);
                        break;
                    }
                }
            }
        }

        // Store in TT
        StoreTT(State.GetStateHash(), BestValue, Depth, ETranspositionType::Exact, BestAction);

        LastResult.BestAction = BestAction;
        return BestValue;
    }

    FFixedPoint32 FAASSearch::AlphaBeta(const IAASGameState& State, int32 Depth,
        FFixedPoint32 Alpha, FFixedPoint32 Beta, int32 Ply, FAASMoveOrderer& MoveOrderer, bool bIsNull)
    {
        ++NodesSearched;

        if (ShouldStop())
        {
            return FFixedPoint32::Zero();
        }

        // Terminal check
        if (State.IsTerminal())
        {
            FFixedPoint32 TermValue = State.GetTerminalValue();
            // Adjust mate scores by ply for shortest mate preference
            if (TermValue.RawValue > EVAL_MATE - 100)
            {
                return FFixedPoint32(TermValue.RawValue - Ply);
            }
            if (TermValue.RawValue < -EVAL_MATE + 100)
            {
                return FFixedPoint32(TermValue.RawValue + Ply);
            }
            return TermValue;
        }

        // Depth limit: go to quiescence
        if (Depth <= 0)
        {
            return Quiescence(State, Alpha, Beta, 0);
        }

        // Transposition table probe
        const uint64 StateHash = State.GetStateHash();
        FAASAction TTAction;
        FFixedPoint32 TTValue;
        if (ProbeTT(StateHash, Depth, Alpha, Beta, TTValue, TTAction))
        {
            return TTValue;
        }

        // Null-move pruning
        if (Config.bUseNullMove && !bIsNull && Depth >= Config.NullMoveReduction + 1)
        {
            // Skip null move in tactical positions (when in check, etc.)
            TArray<FAASAction> Actions;
            State.GetLegalActions(Actions);
            
            if (Actions.Num() > 5) // Not too few moves
            {
                // Apply null move (pass)
                FFixedPoint32 NullValue = -AlphaBeta(State, Depth - Config.NullMoveReduction - 1,
                    -Beta, FFixedPoint32(-(Beta.RawValue - 1)), Ply + 1, MoveOrderer, true);

                if (NullValue >= Beta)
                {
                    // Null move failed high - this position is probably good
                    return Beta;
                }
            }
        }

        // Generate and order moves
        TArray<FAASAction> LegalActions;
        State.GetLegalActions(LegalActions);

        if (LegalActions.IsEmpty())
        {
            // No legal moves but not terminal - stalemate
            return FFixedPoint32::Zero();
        }

        // Order moves
        FAASAction* HashMove = (TTAction.From != 0 || TTAction.To != 0) ? &TTAction : nullptr;
        MoveOrderer.OrderMoves(LegalActions, State, Ply, HashMove);

        FAASAction BestAction = LegalActions[0];
        FFixedPoint32 BestValue = FFixedPoint32::FromInt(-EVAL_INF);
        ETranspositionType TTType = ETranspositionType::UpperBound;

        for (int32 MoveIndex = 0; MoveIndex < LegalActions.Num(); ++MoveIndex)
        {
            if (ShouldStop())
            {
                break;
            }

            const FAASAction& Action = LegalActions[MoveIndex];
            TUniquePtr<IAASGameState> ChildState = State.ApplyAction(Action);

            FFixedPoint32 Value;
            int32 SearchDepth = Depth - 1;

            // Late Move Reductions
            bool bDoLMR = Config.bUseLMR && 
                          MoveIndex >= 4 && 
                          Depth >= 3 && 
                          !(Action.TypeFlags & 0x03); // Not capture or check

            if (bDoLMR)
            {
                // Reduced depth search
                int32 Reduction = 1;
                if (MoveIndex >= 6) Reduction = 2;
                if (MoveIndex >= 12) Reduction = 3;

                Value = -AlphaBeta(*ChildState, Depth - 1 - Reduction,
                    FFixedPoint32(-(Alpha.RawValue + 1)), -Alpha, Ply + 1, MoveOrderer, false);

                // Re-search if fails high
                if (Value > Alpha)
                {
                    Value = -AlphaBeta(*ChildState, SearchDepth, -Beta, -Alpha, Ply + 1, MoveOrderer, false);
                }
            }
            else if (MoveIndex > 0)
            {
                // PVS: null window search first
                Value = -AlphaBeta(*ChildState, SearchDepth,
                    FFixedPoint32(-(Alpha.RawValue + 1)), -Alpha, Ply + 1, MoveOrderer, false);

                if (Value > Alpha && Value < Beta)
                {
                    Value = -AlphaBeta(*ChildState, SearchDepth, -Beta, -Alpha, Ply + 1, MoveOrderer, false);
                }
            }
            else
            {
                // Full window for first move
                Value = -AlphaBeta(*ChildState, SearchDepth, -Beta, -Alpha, Ply + 1, MoveOrderer, false);
            }

            if (Value > BestValue)
            {
                BestValue = Value;
                BestAction = Action;
            }

            if (Value > Alpha)
            {
                Alpha = Value;
                TTType = ETranspositionType::Exact;
            }

            if (Alpha >= Beta)
            {
                // Beta cutoff
                MoveOrderer.RecordKiller(Action, Ply);
                MoveOrderer.RecordHistory(Action, Depth);
                TTType = ETranspositionType::LowerBound;
                break;
            }
        }

        // Store in TT
        StoreTT(StateHash, BestValue, Depth, TTType, BestAction);

        return BestValue;
    }

    FFixedPoint32 FAASSearch::Quiescence(const IAASGameState& State, FFixedPoint32 Alpha, 
        FFixedPoint32 Beta, int32 QDepth)
    {
        ++NodesSearched;

        if (ShouldStop() || QDepth >= Config.QuiescenceDepth)
        {
            return CachedHeuristics->Evaluate(State);
        }

        // Stand pat
        FFixedPoint32 StandPat = CachedHeuristics->Evaluate(State);

        if (StandPat >= Beta)
        {
            return Beta;
        }

        if (StandPat > Alpha)
        {
            Alpha = StandPat;
        }

        // Generate tactical moves only (captures, promotions)
        TArray<FAASAction> TacticalMoves;
        State.GetLegalActions(TacticalMoves);

        // Filter to captures only (TypeFlags bit 0)
        TacticalMoves.RemoveAll([](const FAASAction& Action)
        {
            return (Action.TypeFlags & 0x01) == 0; // Keep only captures
        });

        // Order by MVV-LVA (captured value is in StaticScore for captures)
        Algo::Sort(TacticalMoves, [](const FAASAction& A, const FAASAction& B)
        {
            return A.StaticScore > B.StaticScore; // Higher capture value first
        });

        for (const FAASAction& Action : TacticalMoves)
        {
            if (ShouldStop())
            {
                break;
            }

            // Delta pruning: skip if capture can't possibly improve alpha
            FFixedPoint32 DeltaMargin = FFixedPoint32::FromFloat(0.2f);
            if (StandPat + Action.StaticScore + DeltaMargin < Alpha)
            {
                continue;
            }

            TUniquePtr<IAASGameState> ChildState = State.ApplyAction(Action);
            FFixedPoint32 Value = -Quiescence(*ChildState, -Beta, -Alpha, QDepth + 1);

            if (Value >= Beta)
            {
                return Beta;
            }

            if (Value > Alpha)
            {
                Alpha = Value;
            }
        }

        return Alpha;
    }

    bool FAASSearch::ShouldStop() const
    {
        if (bShouldCancel)
        {
            return true;
        }

        if (Config.TimeLimitMs > 0 && GetElapsedMs() >= Config.TimeLimitMs)
        {
            return true;
        }

        return false;
    }

    double FAASSearch::GetElapsedMs() const
    {
        return (FPlatformTime::Seconds() * 1000.0) - SearchStartTime;
    }

    int32 FAASSearch::GetAdaptiveDepth(FFixedPoint32 Entropy) const
    {
        // Low entropy: extend depth (position is clear)
        if (Entropy < Config.LowEntropyThreshold)
        {
            return Config.BaseDepth + 2;
        }

        // High entropy: reduce depth but search wider
        if (Entropy > Config.HighEntropyThreshold)
        {
            return FMath::Max(Config.BaseDepth - 2, 4);
        }

        return Config.BaseDepth;
    }

    void FAASSearch::StoreTT(uint64 StateHash, FFixedPoint32 Value, int32 Depth,
        ETranspositionType Type, const FAASAction& BestAction)
    {
        FAASTranspositionEntry Entry;
        Entry.StateHash = StateHash;
        Entry.Value = Value;
        Entry.Depth = Depth;
        Entry.Type = Type;
        Entry.BestAction = BestAction;
        TranspositionTable->Store(Entry);
    }

    bool FAASSearch::ProbeTT(uint64 StateHash, int32 Depth, FFixedPoint32 Alpha, FFixedPoint32 Beta,
        FFixedPoint32& OutValue, FAASAction& OutBestAction) const
    {
        FAASTranspositionEntry Entry;
        if (!TranspositionTable->Probe(StateHash, Entry))
        {
            return false;
        }

        OutBestAction = Entry.BestAction;

        // Only use value if depth is sufficient
        if (Entry.Depth < Depth)
        {
            return false;
        }

        OutValue = Entry.Value;

        switch (Entry.Type)
        {
        case ETranspositionType::Exact:
            return true;

        case ETranspositionType::LowerBound:
            if (Entry.Value >= Beta)
            {
                return true;
            }
            break;

        case ETranspositionType::UpperBound:
            if (Entry.Value <= Alpha)
            {
                return true;
            }
            break;
        }

        return false;
    }

    //--------------------------------------------------------------------------
    // FAASMultiAgentCoordinator Implementation
    //--------------------------------------------------------------------------

    FAASMultiAgentCoordinator::FAASMultiAgentCoordinator()
    {
    }

    void FAASMultiAgentCoordinator::AddAgent(int32 AgentID, TSharedPtr<FAASSearch> Search)
    {
        AgentSearches.Add(AgentID, Search);
    }

    void FAASMultiAgentCoordinator::RemoveAgent(int32 AgentID)
    {
        AgentSearches.Remove(AgentID);
    }

    TDeterministicMap<int32, FAASAction> FAASMultiAgentCoordinator::CoordinatedSearch(
        const IAASGameState& SharedState,
        const TDeterministicMap<int32, TUniquePtr<IAASGameState>>& AgentStates,
        const FAASHeuristics& SharedHeuristics)
    {
        TDeterministicMap<int32, FAASAction> Results;

        // Phase 1: Independent searches
        TDeterministicMap<int32, FAASSearchResult> AgentResults;
        
        for (const auto& Pair : AgentSearches)
        {
            const int32 AgentID = Pair.Key;
            FAASSearch* Search = Pair.Value.Get();

            if (!Search)
            {
                continue;
            }

            // Get agent's perspective state
            const TUniquePtr<IAASGameState>* AgentState = AgentStates.Find(AgentID);
            if (!AgentState || !AgentState->IsValid())
            {
                continue;
            }

            FAASSearchResult Result = Search->Search(**AgentState, SharedHeuristics);
            AgentResults.Add(AgentID, Result);

            // Update blackboard with agent's intended action
            if (Result.BestAction.From != 0 || Result.BestAction.To != 0)
            {
                FName ActionKey(*FString::Printf(TEXT("Agent_%d_Target"), AgentID));
                UpdateBlackboard(AgentID, ActionKey, FFixedPoint32::FromInt(Result.BestAction.To));
            }
        }

        // Phase 2: Coordination refinement
        // Agents can see each other's intentions through blackboard and adjust
        for (const auto& ResultPair : AgentResults)
        {
            const int32 AgentID = ResultPair.Key;
            const FAASSearchResult& Result = ResultPair.Value;

            // Check for conflicts with other agents
            bool bHasConflict = false;
            for (const auto& OtherPair : AgentResults)
            {
                if (OtherPair.Key == AgentID)
                {
                    continue;
                }

                // Simple conflict: same target location
                if (Result.BestAction.To == OtherPair.Value.BestAction.To)
                {
                    bHasConflict = true;
                    break;
                }
            }

            // For now, just use the original result
            // Advanced: could re-search with conflict penalty
            Results.Add(AgentID, Result.BestAction);
        }

        return Results;
    }

    void FAASMultiAgentCoordinator::UpdateBlackboard(int32 AgentID, const FName& Key, FFixedPoint32 Value)
    {
        Blackboard.Add(Key, Value);
    }

    FFixedPoint32 FAASMultiAgentCoordinator::ReadBlackboard(const FName& Key) const
    {
        return Blackboard.FindOrDefault(Key, FFixedPoint32::Zero());
    }
}
