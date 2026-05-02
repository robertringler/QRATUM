/// File: Source/QRATUM/Public/Core/AASPlanner.h
// Copyright QRATUM Platform. All Rights Reserved.
// Asymmetric Adaptive Search - High-level planner interface

#pragma once

#include "CoreMinimal.h"
#include "AASSearch.h"
#include "AASHeuristics.h"
#include "Determinism/DeterministicTypes.h"

namespace QRATUM
{
    /**
     * Planned action with confidence and alternatives.
     */
    struct QRATUM_API FAASPlannedAction
    {
        /** The primary recommended action */
        FAASAction PrimaryAction;

        /** Confidence in this action [0, 1] */
        FFixedPoint32 Confidence;

        /** Alternative actions if primary is blocked */
        TArray<FAASAction> Alternatives;

        /** Expected value of executing this action */
        FFixedPoint32 ExpectedValue;

        /** Number of moves lookahead used to determine this action */
        int32 LookaheadDepth;

        /** Time taken to plan this action (ms) */
        double PlanningTimeMs;

        FAASPlannedAction()
            : Confidence(FFixedPoint32::Zero())
            , ExpectedValue(FFixedPoint32::Zero())
            , LookaheadDepth(0)
            , PlanningTimeMs(0.0)
        {}

        bool IsValid() const
        {
            return PrimaryAction.From != 0 || PrimaryAction.To != 0;
        }
    };

    /**
     * Planning context providing world information to the planner.
     */
    struct QRATUM_API FAASPlanningContext
    {
        /** Current game state */
        const IAASGameState* CurrentState;

        /** Available time for planning (ms) */
        double AvailableTimeMs;

        /** Target quality level [0, 1] - higher = more thorough search */
        FFixedPoint32 QualityTarget;

        /** Whether this is an urgent situation (enemy nearby, etc.) */
        bool bUrgent;

        /** Frame budget for incremental planning */
        double FrameBudgetMs;

        FAASPlanningContext()
            : CurrentState(nullptr)
            , AvailableTimeMs(100.0)
            , QualityTarget(FFixedPoint32::FromFloat(0.8f))
            , bUrgent(false)
            , FrameBudgetMs(2.0)
        {}
    };

    /**
     * FAASPlanner - High-level planning interface
     * 
     * The planner provides a clean interface for game AI to request
     * tactical decisions without managing search internals.
     * 
     * Key Features:
     * - Incremental planning that respects frame budget
     * - Plan caching and reuse
     * - Multi-phase planning with different strategies
     * - Resource-aware quality adaptation
     * 
     * Usage Pattern:
     * 1. Create planner with domain-specific state/heuristics
     * 2. Each frame: call PlanStep() with current context
     * 3. When ready: retrieve action with GetPlannedAction()
     * 4. On state change: call InvalidatePlan() to force re-planning
     */
    class QRATUM_API FAASPlanner
    {
    public:
        FAASPlanner();
        ~FAASPlanner();

        // Non-copyable
        FAASPlanner(const FAASPlanner&) = delete;
        FAASPlanner& operator=(const FAASPlanner&) = delete;

        /**
         * Initialize planner with heuristics.
         * @param InHeuristics - Evaluation heuristics (ownership shared)
         * @param InConfig - Search configuration
         */
        void Initialize(TSharedPtr<FAASHeuristics> InHeuristics, 
            const FAASSearchConfig& InConfig = FAASSearchConfig());

        /**
         * Execute one step of planning within frame budget.
         * 
         * Call this each frame. Returns true when planning is complete.
         * 
         * @param Context - Current planning context
         * @return True if a plan is ready
         */
        bool PlanStep(const FAASPlanningContext& Context);

        /**
         * Get the current planned action.
         * 
         * Valid after PlanStep returns true or after GetBestActionSoFar.
         */
        FAASPlannedAction GetPlannedAction() const;

        /**
         * Get the best action found so far (even if planning not complete).
         * 
         * Useful when forced to act before planning completes.
         */
        FAASPlannedAction GetBestActionSoFar() const;

        /**
         * Evaluate a specific action without full search.
         * 
         * Quick evaluation for action validation or comparison.
         */
        FFixedPoint32 EvaluateAction(const IAASGameState& State, const FAASAction& Action) const;

        /**
         * Invalidate current plan (force re-planning).
         * 
         * Call when the world state changes significantly.
         */
        void InvalidatePlan();

        /**
         * Check if currently planning.
         */
        bool IsPlanning() const { return bIsPlanning; }

        /**
         * Check if a valid plan is ready.
         */
        bool HasPlan() const { return CurrentPlan.IsValid(); }

        /**
         * Get search statistics from last planning cycle.
         */
        const FAASSearchResult& GetSearchStats() const;

        /**
         * Set search configuration.
         */
        void SetConfig(const FAASSearchConfig& InConfig);

        /**
         * Get current configuration.
         */
        const FAASSearchConfig& GetConfig() const;

        /**
         * Reset planner state completely.
         */
        void Reset();

    private:
        /** Search engine */
        TUniquePtr<FAASSearch> Search;

        /** Evaluation heuristics */
        TSharedPtr<FAASHeuristics> Heuristics;

        /** Current plan */
        FAASPlannedAction CurrentPlan;

        /** Planning state */
        bool bIsPlanning;
        bool bPlanValid;

        /** Cached state hash for plan invalidation */
        uint64 LastStateHash;

        /** Planning statistics */
        double TotalPlanningTimeMs;
        int32 PlanningIterations;

        /** Begin planning with new context */
        void BeginPlanning(const FAASPlanningContext& Context);

        /** Compute confidence from search result */
        FFixedPoint32 ComputeConfidence(const FAASSearchResult& Result) const;

        /** Extract alternatives from search tree */
        void ExtractAlternatives(TArray<FAASAction>& OutAlternatives) const;
    };

    /**
     * Behavior Tree comparison interface.
     * 
     * Provides methods to compare AAS decisions with behavior tree outputs
     * for validation and demonstration purposes.
     */
    class QRATUM_API FAASBehaviorTreeCompare
    {
    public:
        /**
         * Log comparison between AAS and BT decision.
         * 
         * @param AASAction - Action chosen by AAS
         * @param BTAction - Action chosen by Behavior Tree
         * @param Context - Description of the decision context
         */
        static void LogComparison(const FAASAction& AASAction, const FAASAction& BTAction,
            const FString& Context);

        /**
         * Compute strategic difference metric.
         * 
         * Returns a value indicating how different the two approaches are.
         * 0 = identical, 1 = completely different
         */
        static FFixedPoint32 ComputeDifference(const FAASAction& AASAction, const FAASAction& BTAction);
    };

    /**
     * Debug/introspection utilities for AAS.
     */
    class QRATUM_API FAASDebugger
    {
    public:
        /**
         * Generate human-readable description of search tree.
         */
        static FString DescribeSearchTree(const FAASNode* Root, int32 MaxDepth = 3);

        /**
         * Generate JSON representation of search result.
         */
        static FString SearchResultToJSON(const FAASSearchResult& Result);

        /**
         * Log search statistics.
         */
        static void LogSearchStats(const FAASSearchResult& Result);

        /**
         * Validate determinism by running search twice.
         * Returns true if results are identical.
         */
        static bool ValidateDeterminism(FAASPlanner& Planner, const IAASGameState& State);
    };
}
