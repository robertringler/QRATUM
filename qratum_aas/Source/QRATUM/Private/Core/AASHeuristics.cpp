/// File: Source/QRATUM/Private/Core/AASHeuristics.cpp
// Copyright QRATUM Platform. All Rights Reserved.
// Asymmetric Adaptive Search - Heuristic evaluation implementation

#include "Core/AASHeuristics.h"
#include "Algo/Sort.h"

namespace QRATUM
{
    //--------------------------------------------------------------------------
    // FAASMoveOrderer Implementation
    //--------------------------------------------------------------------------

    FAASMoveOrderer::FAASMoveOrderer()
    {
        KillerMoves.SetNum(MaxPly * KillersPerPly);
    }

    void FAASMoveOrderer::OrderMoves(TArray<FAASAction>& Moves, const IAASGameState& State,
        int32 Ply, const FAASAction* HashMove) const
    {
        if (Moves.Num() <= 1)
        {
            return; // Nothing to order
        }

        // Compute scores for all moves
        TArray<TPair<int32, int32>> Scores; // (score, original index)
        Scores.Reserve(Moves.Num());

        for (int32 i = 0; i < Moves.Num(); ++i)
        {
            int32 Score = ComputeMoveScore(Moves[i], State, Ply, HashMove);
            Scores.Emplace(Score, i);
        }

        // Sort by score (highest first), with stable original index tiebreaker
        Algo::Sort(Scores, [](const TPair<int32, int32>& A, const TPair<int32, int32>& B)
        {
            if (A.Key != B.Key)
            {
                return A.Key > B.Key; // Higher score first
            }
            return A.Value < B.Value; // Lower original index as tiebreaker
        });

        // Reorder moves array
        TArray<FAASAction> OrderedMoves;
        OrderedMoves.Reserve(Moves.Num());
        for (const auto& Pair : Scores)
        {
            OrderedMoves.Add(Moves[Pair.Value]);
        }
        Moves = MoveTemp(OrderedMoves);
    }

    int32 FAASMoveOrderer::ComputeMoveScore(const FAASAction& Action, const IAASGameState& State,
        int32 Ply, const FAASAction* HashMove) const
    {
        // Priority order (higher score = searched first):
        // 1. Hash move: 1,000,000
        // 2. High-prior captures: 500,000 + capture_value
        // 3. Killers: 400,000 - slot
        // 4. History: history_score
        // 5. Prior: prior * 1000
        // 6. Other: 0

        // Hash move gets highest priority
        if (HashMove && Action == *HashMove)
        {
            return 1000000;
        }

        // Check if action is a capture (TypeFlags bit 0 indicates capture)
        const bool bIsCapture = (Action.TypeFlags & 0x01) != 0;
        if (bIsCapture)
        {
            // Capture ordering by estimated gain
            int32 CaptureValue = Action.StaticScore.RawValue / 100;
            return 500000 + CaptureValue;
        }

        // Killer move check
        if (Ply >= 0 && Ply < MaxPly)
        {
            for (int32 Slot = 0; Slot < KillersPerPly; ++Slot)
            {
                const int32 KillerIdx = GetKillerIndex(Ply, Slot);
                if (KillerMoves[KillerIdx] == Action)
                {
                    return 400000 - Slot * 100;
                }
            }
        }

        // History heuristic
        const uint64 HistoryKey = HashCombine(Action.From, Action.To);
        const int32* HistoryScore = HistoryScores.Find(HistoryKey);
        if (HistoryScore && *HistoryScore > 0)
        {
            return FMath::Min(*HistoryScore, 399999);
        }

        // Default: use prior probability
        return Action.Prior.RawValue / 32; // Scale to reasonable range
    }

    void FAASMoveOrderer::RecordKiller(const FAASAction& Action, int32 Ply)
    {
        if (Ply < 0 || Ply >= MaxPly)
        {
            return;
        }

        // Don't record captures as killers
        if ((Action.TypeFlags & 0x01) != 0)
        {
            return;
        }

        const int32 Slot0 = GetKillerIndex(Ply, 0);
        const int32 Slot1 = GetKillerIndex(Ply, 1);

        // Don't duplicate
        if (KillerMoves[Slot0] == Action)
        {
            return;
        }

        // Shift slot 0 to slot 1, add new to slot 0
        KillerMoves[Slot1] = KillerMoves[Slot0];
        KillerMoves[Slot0] = Action;
    }

    void FAASMoveOrderer::RecordHistory(const FAASAction& Action, int32 Depth)
    {
        // Don't record captures in history
        if ((Action.TypeFlags & 0x01) != 0)
        {
            return;
        }

        const uint64 HistoryKey = HashCombine(Action.From, Action.To);
        int32* Existing = HistoryScores.Find(HistoryKey);
        
        // Depth-squared bonus
        const int32 Bonus = Depth * Depth;
        
        if (Existing)
        {
            *Existing += Bonus;
            // Cap to prevent overflow
            *Existing = FMath::Min(*Existing, 100000);
        }
        else
        {
            HistoryScores.Add(HistoryKey, Bonus);
        }
    }

    void FAASMoveOrderer::Clear()
    {
        for (FAASAction& Killer : KillerMoves)
        {
            Killer = FAASAction();
        }
        HistoryScores.Empty();
    }

    void FAASMoveOrderer::AgeHistory()
    {
        // Reduce all history scores by half
        for (auto& Pair : HistoryScores)
        {
            Pair.Value /= 2;
        }
    }

    //--------------------------------------------------------------------------
    // FAASHeuristics Implementation
    //--------------------------------------------------------------------------

    FAASHeuristics::FAASHeuristics()
    {
    }

    FFixedPoint32 FAASHeuristics::Evaluate(const IAASGameState& State) const
    {
        if (State.IsTerminal())
        {
            return State.GetTerminalValue();
        }

        if (Features.IsEmpty())
        {
            // No features registered, return neutral
            return FFixedPoint32::Zero();
        }

        // Sum weighted features
        FFixedPoint32 TotalValue = FFixedPoint32::Zero();
        FFixedPoint32 TotalWeight = FFixedPoint32::Zero();

        for (const FAASHeuristicFeature& Feature : Features)
        {
            if (Feature.ExtractFeature)
            {
                FFixedPoint32 FeatureValue = Feature.ExtractFeature(State);
                TotalValue = TotalValue + Feature.Weight * FeatureValue;
                TotalWeight = TotalWeight + FFixedPoint32(FMath::Abs(Feature.Weight.RawValue));
            }
        }

        // Normalize by total weight to get [-1, 1] range
        if (TotalWeight.RawValue > 0)
        {
            TotalValue = TotalValue / TotalWeight;
        }

        // Clamp to valid range
        return Normalize(TotalValue, FFixedPoint32::One());
    }

    FFixedPoint32 FAASHeuristics::EvaluateAction(const IAASGameState& State, const FAASAction& Action) const
    {
        // Return action's prior if set
        if (Action.Prior.RawValue != 0)
        {
            return Action.Prior;
        }

        // Otherwise compute based on action properties
        FFixedPoint32 Score = FFixedPoint32::FromFloat(0.5f); // Base prior

        // Capture bonus
        if (Action.TypeFlags & 0x01)
        {
            Score = Score + FFixedPoint32::FromFloat(0.2f);
        }

        // Check/forcing move bonus
        if (Action.TypeFlags & 0x02)
        {
            Score = Score + FFixedPoint32::FromFloat(0.15f);
        }

        return Normalize(Score, FFixedPoint32::One());
    }

    FFixedPoint32 FAASHeuristics::CalculateEntropy(const IAASGameState& State) const
    {
        TArray<FAASAction> Actions;
        State.GetLegalActions(Actions);

        if (Actions.Num() <= 1)
        {
            return FFixedPoint32::Zero(); // No uncertainty with forced move
        }

        // Compute entropy based on action prior distribution
        // H = -sum(p * log(p))
        
        FFixedPoint32 TotalPrior = FFixedPoint32::Zero();
        for (const FAASAction& Action : Actions)
        {
            FFixedPoint32 Prior = EvaluateAction(State, Action);
            TotalPrior = TotalPrior + Prior;
        }

        if (TotalPrior.RawValue <= 0)
        {
            // Uniform distribution
            const float UniformEntropy = FMath::Loge(static_cast<float>(Actions.Num()));
            return FFixedPoint32::FromFloat(UniformEntropy);
        }

        // Normalized entropy calculation
        float Entropy = 0.0f;
        for (const FAASAction& Action : Actions)
        {
            FFixedPoint32 Prior = EvaluateAction(State, Action);
            float P = Prior.ToFloat() / TotalPrior.ToFloat();
            if (P > 0.001f)
            {
                Entropy -= P * FMath::Loge(P);
            }
        }

        return FFixedPoint32::FromFloat(Entropy);
    }

    void FAASHeuristics::AddFeature(const FAASHeuristicFeature& Feature)
    {
        // Check for duplicate
        for (FAASHeuristicFeature& Existing : Features)
        {
            if (Existing.FeatureName == Feature.FeatureName)
            {
                Existing = Feature;
                return;
            }
        }
        Features.Add(Feature);
    }

    void FAASHeuristics::SetFeatureWeight(FName FeatureName, FFixedPoint32 Weight)
    {
        for (FAASHeuristicFeature& Feature : Features)
        {
            if (Feature.FeatureName == FeatureName)
            {
                Feature.Weight = Weight;
                return;
            }
        }
    }

    void FAASHeuristics::GetFeatureWeights(TDeterministicMap<FName, FFixedPoint32>& OutWeights) const
    {
        OutWeights.Empty();
        for (const FAASHeuristicFeature& Feature : Features)
        {
            OutWeights.Add(Feature.FeatureName, Feature.Weight);
        }
    }

    void FAASHeuristics::SetFeatureWeights(const TDeterministicMap<FName, FFixedPoint32>& Weights)
    {
        for (const auto& Pair : Weights)
        {
            SetFeatureWeight(Pair.Key, Pair.Value);
        }
    }

    FFixedPoint32 FAASHeuristics::Normalize(FFixedPoint32 Value, FFixedPoint32 Scale)
    {
        // Clamp to [-Scale, Scale]
        if (Value.RawValue > Scale.RawValue)
        {
            return Scale;
        }
        if (Value.RawValue < -Scale.RawValue)
        {
            return FFixedPoint32(-Scale.RawValue);
        }
        return Value;
    }

    //--------------------------------------------------------------------------
    // FAASMultiPhaseHeuristics Implementation
    //--------------------------------------------------------------------------

    FAASMultiPhaseHeuristics::FAASMultiPhaseHeuristics()
    {
        DefaultHeuristics = MakeShared<FAASHeuristics>();
    }

    void FAASMultiPhaseHeuristics::SetPhaseHeuristics(EGamePhase Phase, TSharedPtr<FAASHeuristics> Heuristics)
    {
        PhaseHeuristics.Add(Phase, Heuristics);
    }

    void FAASMultiPhaseHeuristics::SetPhaseDetector(TSharedPtr<FAASPhaseDetector> Detector)
    {
        PhaseDetector = Detector;
    }

    FFixedPoint32 FAASMultiPhaseHeuristics::Evaluate(const IAASGameState& State) const
    {
        FAASHeuristics* Heuristics = GetCurrentHeuristics(State);
        return Heuristics ? Heuristics->Evaluate(State) : FFixedPoint32::Zero();
    }

    EGamePhase FAASMultiPhaseHeuristics::GetCurrentPhase(const IAASGameState& State) const
    {
        if (State.IsTerminal())
        {
            return EGamePhase::Terminal;
        }

        if (PhaseDetector)
        {
            return PhaseDetector->DetectPhase(State);
        }

        // Default to middlegame
        return EGamePhase::Middlegame;
    }

    FAASHeuristics* FAASMultiPhaseHeuristics::GetCurrentHeuristics(const IAASGameState& State) const
    {
        EGamePhase Phase = GetCurrentPhase(State);
        
        TSharedPtr<FAASHeuristics>* Found = PhaseHeuristics.Find(Phase);
        if (Found && Found->IsValid())
        {
            return Found->Get();
        }

        return DefaultHeuristics.Get();
    }
}
