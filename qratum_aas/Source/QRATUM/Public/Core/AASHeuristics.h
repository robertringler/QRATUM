/// File: Source/QRATUM/Public/Core/AASHeuristics.h
// Copyright QRATUM Platform. All Rights Reserved.
// Asymmetric Adaptive Search - Heuristic evaluation system

#pragma once

#include "CoreMinimal.h"
#include "AASNode.h"
#include "Determinism/DeterministicTypes.h"

namespace QRATUM
{
    /**
     * Abstract game state interface for AAS heuristics.
     * 
     * Domain-specific game implementations must provide a concrete
     * subclass that implements these methods.
     * 
     * Design Note: This interface is engine-agnostic. Unreal-specific
     * state (actors, components) is wrapped by integration layer.
     */
    class QRATUM_API IAASGameState
    {
    public:
        virtual ~IAASGameState() = default;

        /** Get hash of current state for transposition detection */
        virtual uint64 GetStateHash() const = 0;

        /** Generate all legal actions from current state */
        virtual void GetLegalActions(TArray<FAASAction>& OutActions) const = 0;

        /** Apply action and return new state (immutable semantics) */
        virtual TUniquePtr<IAASGameState> ApplyAction(const FAASAction& Action) const = 0;

        /** Check if current state is terminal (game over) */
        virtual bool IsTerminal() const = 0;

        /** Get terminal value if IsTerminal() is true */
        virtual FFixedPoint32 GetTerminalValue() const = 0;

        /** Get the ID of the agent whose turn it is (for multi-agent) */
        virtual int32 GetActiveAgentID() const = 0;

        /** Clone this state (for search tree manipulation) */
        virtual TUniquePtr<IAASGameState> Clone() const = 0;
    };

    /**
     * Heuristic feature for position evaluation.
     * 
     * Each feature extracts a scalar value from the game state.
     * Features are combined by the evaluator using learned weights.
     */
    struct QRATUM_API FAASHeuristicFeature
    {
        /** Feature identifier */
        FName FeatureName;

        /** Feature weight (can be negative) */
        FFixedPoint32 Weight;

        /** Feature extraction function */
        TFunction<FFixedPoint32(const IAASGameState&)> ExtractFeature;

        FAASHeuristicFeature()
            : FeatureName(NAME_None)
            , Weight(FFixedPoint32::One())
        {}

        FAASHeuristicFeature(FName InName, FFixedPoint32 InWeight,
            TFunction<FFixedPoint32(const IAASGameState&)> InExtractor)
            : FeatureName(InName)
            , Weight(InWeight)
            , ExtractFeature(MoveTemp(InExtractor))
        {}
    };

    /**
     * Move ordering heuristic for efficient pruning.
     * 
     * Better move ordering leads to more pruning in alpha-beta search,
     * dramatically improving search efficiency. The order is:
     * 
     * 1. Hash move (from transposition table)
     * 2. Winning captures (MVV-LVA positive)
     * 3. Killer moves (caused beta cutoffs at this ply)
     * 4. History heuristic (caused cutoffs historically)
     * 5. Other moves (by prior probability)
     */
    class QRATUM_API FAASMoveOrderer
    {
    public:
        FAASMoveOrderer();

        /** Order moves for search efficiency */
        void OrderMoves(TArray<FAASAction>& Moves, const IAASGameState& State,
            int32 Ply, const FAASAction* HashMove = nullptr) const;

        /** Record a killer move (caused beta cutoff) */
        void RecordKiller(const FAASAction& Action, int32 Ply);

        /** Record history score (move caused cutoff) */
        void RecordHistory(const FAASAction& Action, int32 Depth);

        /** Clear all heuristic data (for new search) */
        void Clear();

        /** Age history scores (for iterative deepening) */
        void AgeHistory();

    private:
        // Killer moves: 2 slots per ply, up to max depth
        static constexpr int32 MaxPly = 128;
        static constexpr int32 KillersPerPly = 2;
        TArray<FAASAction> KillerMoves;

        // History heuristic: maps (from, to) -> score
        TDeterministicMap<uint64, int32> HistoryScores;

        /** Compute move score for ordering */
        int32 ComputeMoveScore(const FAASAction& Action, const IAASGameState& State,
            int32 Ply, const FAASAction* HashMove) const;

        /** Get killer move index */
        int32 GetKillerIndex(int32 Ply, int32 Slot) const
        {
            return Ply * KillersPerPly + Slot;
        }
    };

    /**
     * FAASHeuristics - Domain-agnostic heuristic evaluation
     * 
     * Combines multiple features with learned weights for position evaluation.
     * Supports:
     * - Static evaluation (leaf nodes)
     * - Action ordering (internal nodes)
     * - Entropy calculation (for resource allocation)
     * 
     * Evaluation is performed from the perspective of the active agent.
     * Values are in range [-1, 1] where 1 is winning and -1 is losing.
     */
    class QRATUM_API FAASHeuristics
    {
    public:
        FAASHeuristics();

        /**
         * Evaluate a game state statically.
         * @param State - The state to evaluate
         * @return Evaluation in [-1, 1] from active agent's perspective
         */
        FFixedPoint32 Evaluate(const IAASGameState& State) const;

        /**
         * Evaluate an action before search (for move ordering).
         * @param State - Current state
         * @param Action - Action to evaluate
         * @return Prior probability in [0, 1]
         */
        FFixedPoint32 EvaluateAction(const IAASGameState& State, const FAASAction& Action) const;

        /**
         * Calculate state entropy (uncertainty measure).
         * Used for adaptive resource allocation.
         * @param State - The state to analyze
         * @return Entropy value >= 0, higher = more uncertain
         */
        FFixedPoint32 CalculateEntropy(const IAASGameState& State) const;

        /** Add a heuristic feature */
        void AddFeature(const FAASHeuristicFeature& Feature);

        /** Set feature weight by name */
        void SetFeatureWeight(FName FeatureName, FFixedPoint32 Weight);

        /** Get current feature weights (for serialization) */
        void GetFeatureWeights(TDeterministicMap<FName, FFixedPoint32>& OutWeights) const;

        /** Set feature weights (for loading) */
        void SetFeatureWeights(const TDeterministicMap<FName, FFixedPoint32>& Weights);

        /** Get move orderer for this heuristic */
        FAASMoveOrderer& GetMoveOrderer() { return MoveOrderer; }
        const FAASMoveOrderer& GetMoveOrderer() const { return MoveOrderer; }

        /** Clear move ordering heuristics */
        void ClearMoveOrderingData() { MoveOrderer.Clear(); }

    private:
        /** Registered heuristic features */
        TArray<FAASHeuristicFeature> Features;

        /** Move ordering heuristic */
        FAASMoveOrderer MoveOrderer;

        /** Normalize value to [-1, 1] range */
        static FFixedPoint32 Normalize(FFixedPoint32 Value, FFixedPoint32 Scale);
    };

    /**
     * Specialized heuristics for different game phases.
     * 
     * AAS uses phase detection to switch between heuristic sets:
     * - Opening: Emphasize development, space, initiative
     * - Middlegame: Emphasize tactics, king safety, piece activity
     * - Endgame: Emphasize material, pawn structure, king activity
     */
    enum class EGamePhase : uint8
    {
        Opening,
        Middlegame,
        Endgame,
        Terminal
    };

    class QRATUM_API FAASPhaseDetector
    {
    public:
        virtual ~FAASPhaseDetector() = default;

        /** Detect current game phase */
        virtual EGamePhase DetectPhase(const IAASGameState& State) const = 0;
    };

    /**
     * Multi-phase heuristic manager.
     * Switches between phase-specific heuristics automatically.
     */
    class QRATUM_API FAASMultiPhaseHeuristics
    {
    public:
        FAASMultiPhaseHeuristics();

        /** Set heuristics for a specific phase */
        void SetPhaseHeuristics(EGamePhase Phase, TSharedPtr<FAASHeuristics> Heuristics);

        /** Set phase detector */
        void SetPhaseDetector(TSharedPtr<FAASPhaseDetector> Detector);

        /** Evaluate using phase-appropriate heuristics */
        FFixedPoint32 Evaluate(const IAASGameState& State) const;

        /** Get current detected phase */
        EGamePhase GetCurrentPhase(const IAASGameState& State) const;

        /** Get heuristics for current phase */
        FAASHeuristics* GetCurrentHeuristics(const IAASGameState& State) const;

    private:
        TMap<EGamePhase, TSharedPtr<FAASHeuristics>> PhaseHeuristics;
        TSharedPtr<FAASPhaseDetector> PhaseDetector;
        TSharedPtr<FAASHeuristics> DefaultHeuristics;
    };
}
