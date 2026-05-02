/// File: Source/QRATUM/Public/Core/AASSearch.h
// Copyright QRATUM Platform. All Rights Reserved.
// Asymmetric Adaptive Search - Core search engine

#pragma once

#include "CoreMinimal.h"
#include "AASNode.h"
#include "AASHeuristics.h"
#include "Determinism/DeterministicTypes.h"
#include "Determinism/DeterministicContainers.h"

namespace QRATUM
{
    /**
     * Search result from AAS.
     */
    struct QRATUM_API FAASSearchResult
    {
        /** Best action found */
        FAASAction BestAction;

        /** Evaluation of best action (from searcher's perspective) */
        FFixedPoint32 Evaluation;

        /** Principal variation (sequence of best moves) */
        TArray<FAASAction> PrincipalVariation;

        /** Total nodes searched */
        int64 NodesSearched;

        /** Maximum depth reached */
        int32 DepthReached;

        /** Time spent in milliseconds */
        double TimeMs;

        /** Transposition table hit rate */
        float TTHitRate;

        /** Whether search completed without timeout */
        bool bCompleted;

        /** Entropy of root position */
        FFixedPoint32 Entropy;

        FAASSearchResult()
            : Evaluation(FFixedPoint32::Zero())
            , NodesSearched(0)
            , DepthReached(0)
            , TimeMs(0.0)
            , TTHitRate(0.0f)
            , bCompleted(true)
            , Entropy(FFixedPoint32::Zero())
        {}
    };

    /**
     * Search configuration.
     */
    struct QRATUM_API FAASSearchConfig
    {
        /** Base search depth */
        int32 BaseDepth = 10;

        /** Maximum search depth (including extensions) */
        int32 MaxDepth = 30;

        /** Quiescence search depth */
        int32 QuiescenceDepth = 8;

        /** Time limit in milliseconds (0 = no limit) */
        double TimeLimitMs = 0.0;

        /** Per-frame time budget for incremental search */
        double FrameBudgetMs = 2.0;

        /** Exploration constant for UCB selection */
        FFixedPoint32 ExplorationConstant = FFixedPoint32::FromFloat(1.414f);

        /** Enable null-move pruning */
        bool bUseNullMove = true;

        /** Null-move reduction depth */
        int32 NullMoveReduction = 3;

        /** Enable late-move reductions */
        bool bUseLMR = true;

        /** Enable aspiration windows */
        bool bUseAspirationWindows = true;

        /** Aspiration window size */
        FFixedPoint32 AspirationWindow = FFixedPoint32::FromFloat(0.25f);

        /** Enable multi-cut pruning */
        bool bUseMultiCut = true;

        /** Transposition table size in MB */
        int32 TranspositionTableSizeMB = 64;

        /** Enable entropy-adaptive depth */
        bool bAdaptiveDepth = true;

        /** Minimum entropy for depth reduction */
        FFixedPoint32 LowEntropyThreshold = FFixedPoint32::FromFloat(0.5f);

        /** Maximum entropy for depth extension */
        FFixedPoint32 HighEntropyThreshold = FFixedPoint32::FromFloat(2.5f);
    };

    /**
     * FAASSearch - Asymmetric Adaptive Search Engine
     * 
     * Production-grade tree search implementing:
     * - Alpha-beta pruning with fail-soft
     * - Iterative deepening with aspiration windows
     * - Transposition table with replacement scheme
     * - Late Move Reductions (LMR)
     * - Null-move pruning
     * - Killer/history heuristics for move ordering
     * - Quiescence search for tactical stability
     * - Entropy-gradient adaptive depth allocation
     * 
     * The "asymmetric" aspect refers to non-uniform branching:
     * - Promising branches are searched deeper
     * - Unpromising branches are pruned aggressively
     * - Resource allocation adapts based on position entropy
     * 
     * The "adaptive" aspect refers to runtime adjustment:
     * - Search parameters adapt based on time pressure
     * - Move ordering improves during iterative deepening
     * - History heuristics learn from search
     * 
     * Determinism Guarantees:
     * - Same state + same config = same result (always)
     * - No use of wall-clock for decisions (only for limits)
     * - All containers iterate in deterministic order
     * - Floating-point replaced with fixed-point
     */
    class QRATUM_API FAASSearch
    {
    public:
        FAASSearch();
        explicit FAASSearch(const FAASSearchConfig& InConfig);
        ~FAASSearch();

        // Non-copyable
        FAASSearch(const FAASSearch&) = delete;
        FAASSearch& operator=(const FAASSearch&) = delete;

        /**
         * Execute full search to find best action.
         * 
         * @param RootState - The game state to search from
         * @param Heuristics - Evaluation heuristics to use
         * @return Search result with best action and statistics
         */
        FAASSearchResult Search(const IAASGameState& RootState, const FAASHeuristics& Heuristics);

        /**
         * Execute one step of incremental search.
         * 
         * Call this each frame for time-budgeted search.
         * Returns true when search is complete.
         * 
         * @param RootState - The game state to search from
         * @param Heuristics - Evaluation heuristics to use
         * @param OutResult - Result so far
         * @return True if search is complete
         */
        bool SearchStep(const IAASGameState& RootState, const FAASHeuristics& Heuristics, 
            FAASSearchResult& OutResult);

        /**
         * Prepare for a new search (call before SearchStep loop).
         */
        void BeginSearch(const IAASGameState& RootState, const FAASHeuristics& Heuristics);

        /**
         * Cancel ongoing search.
         */
        void CancelSearch();

        /**
         * Check if search is in progress.
         */
        bool IsSearching() const { return bIsSearching; }

        /**
         * Get current configuration.
         */
        const FAASSearchConfig& GetConfig() const { return Config; }

        /**
         * Set configuration (only when not searching).
         */
        void SetConfig(const FAASSearchConfig& InConfig);

        /**
         * Clear transposition table and search state.
         */
        void Reset();

        /**
         * Get the search tree root (for debugging/visualization).
         */
        const FAASNode* GetRootNode() const { return RootNode.Get(); }

        /**
         * Get last search statistics.
         */
        const FAASSearchResult& GetLastResult() const { return LastResult; }

    private:
        /** Search configuration */
        FAASSearchConfig Config;

        /** Transposition table */
        TUniquePtr<FAASTranspositionTable> TranspositionTable;

        /** Root node of search tree */
        TUniquePtr<FAASNode> RootNode;

        /** Cached root state for incremental search */
        TUniquePtr<IAASGameState> CachedRootState;

        /** Cached heuristics pointer */
        const FAASHeuristics* CachedHeuristics;

        /** Search state */
        bool bIsSearching;
        bool bShouldCancel;
        int32 CurrentDepth;
        int64 NodesSearched;
        double SearchStartTime;

        /** Last search result */
        FAASSearchResult LastResult;

        /** Principal variation from last complete iteration */
        TArray<FAASAction> CurrentPV;

        /** Deterministic RNG for search decisions */
        FDeterministicRNG SearchRNG;

        /**
         * Alpha-beta search with all optimizations.
         * Returns value from perspective of side to move.
         */
        FFixedPoint32 AlphaBeta(
            const IAASGameState& State,
            int32 Depth,
            FFixedPoint32 Alpha,
            FFixedPoint32 Beta,
            int32 Ply,
            FAASMoveOrderer& MoveOrderer,
            bool bIsNull
        );

        /**
         * Quiescence search for tactical stability.
         */
        FFixedPoint32 Quiescence(
            const IAASGameState& State,
            FFixedPoint32 Alpha,
            FFixedPoint32 Beta,
            int32 QDepth
        );

        /**
         * Root search with aspiration windows.
         */
        FFixedPoint32 SearchRoot(
            const IAASGameState& State,
            int32 Depth,
            FAASMoveOrderer& MoveOrderer
        );

        /**
         * Check if we should stop searching (time/cancellation).
         */
        bool ShouldStop() const;

        /**
         * Get elapsed time since search start.
         */
        double GetElapsedMs() const;

        /**
         * Calculate adaptive depth based on entropy.
         */
        int32 GetAdaptiveDepth(FFixedPoint32 Entropy) const;

        /**
         * Store result in transposition table.
         */
        void StoreTT(uint64 StateHash, FFixedPoint32 Value, int32 Depth,
            ETranspositionType Type, const FAASAction& BestAction);

        /**
         * Probe transposition table.
         */
        bool ProbeTT(uint64 StateHash, int32 Depth, FFixedPoint32 Alpha, FFixedPoint32 Beta,
            FFixedPoint32& OutValue, FAASAction& OutBestAction) const;
    };

    /**
     * Multi-agent search coordinator.
     * 
     * Manages multiple AAS instances searching from different perspectives.
     * Used for squad-level tactical coordination.
     * 
     * Coordination emerges from:
     * - Shared evaluation of world state
     * - Cost functions that reward cooperation
     * - Information sharing through blackboard
     */
    class QRATUM_API FAASMultiAgentCoordinator
    {
    public:
        FAASMultiAgentCoordinator();

        /** Add an agent's search engine */
        void AddAgent(int32 AgentID, TSharedPtr<FAASSearch> Search);

        /** Remove an agent */
        void RemoveAgent(int32 AgentID);

        /**
         * Execute coordinated search for all agents.
         * 
         * @param SharedState - Shared game state
         * @param AgentStates - Per-agent perspective states
         * @param SharedHeuristics - Common evaluation
         * @return Map of agent ID to their best action
         */
        TDeterministicMap<int32, FAASAction> CoordinatedSearch(
            const IAASGameState& SharedState,
            const TDeterministicMap<int32, TUniquePtr<IAASGameState>>& AgentStates,
            const FAASHeuristics& SharedHeuristics
        );

        /**
         * Update shared blackboard with coordination data.
         */
        void UpdateBlackboard(int32 AgentID, const FName& Key, FFixedPoint32 Value);

        /**
         * Read from shared blackboard.
         */
        FFixedPoint32 ReadBlackboard(const FName& Key) const;

    private:
        /** Per-agent search engines */
        TDeterministicMap<int32, TSharedPtr<FAASSearch>> AgentSearches;

        /** Shared coordination blackboard */
        TDeterministicMap<FName, FFixedPoint32> Blackboard;

        /** Coordination weights for multi-agent scoring */
        struct FCoordinationParams
        {
            FFixedPoint32 SelfWeight = FFixedPoint32::FromFloat(0.7f);
            FFixedPoint32 TeamWeight = FFixedPoint32::FromFloat(0.3f);
            FFixedPoint32 ProximityBonus = FFixedPoint32::FromFloat(0.1f);
        };
        FCoordinationParams CoordParams;
    };
}
