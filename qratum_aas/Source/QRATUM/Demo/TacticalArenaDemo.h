/// File: Source/QRATUM/Demo/TacticalArenaDemo.h
// Copyright QRATUM Platform. All Rights Reserved.
// Demo: Tactical Arena with AAS-driven AI agents

#pragma once

#include "CoreMinimal.h"
#include "Core/AASHeuristics.h"
#include "Core/AASPlanner.h"
#include "Determinism/DeterministicTypes.h"

namespace QRATUM
{
    //--------------------------------------------------------------------------
    // Tactical Arena Game State
    //--------------------------------------------------------------------------

    /**
     * Arena cell types
     */
    enum class EArenaCell : uint8
    {
        Empty,
        Wall,
        Cover,       // Half cover - reduces incoming damage
        HighGround,  // Elevation - bonus to attacks
        Objective    // Control point
    };

    /**
     * Agent data in the arena
     */
    struct FArenaAgent
    {
        int32 AgentID;
        int32 TeamID;
        int32 PositionX;
        int32 PositionY;
        int32 Health;
        int32 MaxHealth;
        int32 ActionPoints;
        int32 MaxActionPoints;
        bool bInCover;
        bool bOnHighGround;

        FArenaAgent()
            : AgentID(-1), TeamID(0)
            , PositionX(0), PositionY(0)
            , Health(100), MaxHealth(100)
            , ActionPoints(2), MaxActionPoints(2)
            , bInCover(false), bOnHighGround(false)
        {}
    };

    /**
     * Tactical arena game state implementation
     */
    class FArenaGameState : public IAASGameState
    {
    public:
        static constexpr int32 ArenaWidth = 16;
        static constexpr int32 ArenaHeight = 16;

        FArenaGameState();
        FArenaGameState(const FArenaGameState& Other);

        // IAASGameState interface
        virtual uint64 GetStateHash() const override;
        virtual void GetLegalActions(TArray<FAASAction>& OutActions) const override;
        virtual TUniquePtr<IAASGameState> ApplyAction(const FAASAction& Action) const override;
        virtual bool IsTerminal() const override;
        virtual FFixedPoint32 GetTerminalValue() const override;
        virtual int32 GetActiveAgentID() const override;
        virtual TUniquePtr<IAASGameState> Clone() const override;

        // Arena-specific methods
        void Initialize();
        void SetCell(int32 X, int32 Y, EArenaCell Cell);
        EArenaCell GetCell(int32 X, int32 Y) const;
        void AddAgent(const FArenaAgent& Agent);
        FArenaAgent* GetAgent(int32 AgentID);
        const FArenaAgent* GetAgent(int32 AgentID) const;
        void SetActiveAgent(int32 AgentID);
        int32 GetTeamScore(int32 TeamID) const;
        
        // Action types
        static constexpr uint32 ACTION_MOVE = 0x01;
        static constexpr uint32 ACTION_ATTACK = 0x02;
        static constexpr uint32 ACTION_OVERWATCH = 0x04;
        static constexpr uint32 ACTION_TAKE_COVER = 0x08;

    private:
        TArray<EArenaCell> Grid;
        TArray<FArenaAgent> Agents;
        int32 ActiveAgentID;
        int32 TurnNumber;

        int32 GetGridIndex(int32 X, int32 Y) const;
        bool IsValidPosition(int32 X, int32 Y) const;
        bool IsPositionOccupied(int32 X, int32 Y) const;
        bool HasLineOfSight(int32 X1, int32 Y1, int32 X2, int32 Y2) const;
        int32 CalculateDistance(int32 X1, int32 Y1, int32 X2, int32 Y2) const;
    };

    //--------------------------------------------------------------------------
    // Arena-specific heuristics
    //--------------------------------------------------------------------------

    /**
     * Tactical arena heuristics
     */
    class FArenaHeuristics : public FAASHeuristics
    {
    public:
        FArenaHeuristics();

        // Register tactical features
        void RegisterFeatures();

    private:
        // Feature extractors
        static FFixedPoint32 ExtractHealthAdvantage(const IAASGameState& State);
        static FFixedPoint32 ExtractPositionalAdvantage(const IAASGameState& State);
        static FFixedPoint32 ExtractObjectiveControl(const IAASGameState& State);
        static FFixedPoint32 ExtractCoverUtilization(const IAASGameState& State);
        static FFixedPoint32 ExtractThreatLevel(const IAASGameState& State);
    };

    /**
     * Arena phase detector
     */
    class FArenaPhaseDetector : public FAASPhaseDetector
    {
    public:
        virtual EGamePhase DetectPhase(const IAASGameState& State) const override;
    };

    //--------------------------------------------------------------------------
    // Demo runner
    //--------------------------------------------------------------------------

    /**
     * Tactical arena demo configuration
     */
    struct FArenaDemo Config
    {
        int32 NumAgentsPerTeam = 3;
        int32 MaxTurns = 50;
        int32 SearchDepth = 10;
        float SearchTimeMs = 100.0f;
        bool bLogMoves = true;
        bool bValidateDeterminism = true;
    };

    /**
     * Tactical arena demo runner
     * 
     * Demonstrates AAS-driven AI in a tactical combat scenario:
     * - Multiple agents per team
     * - Cover system
     * - Elevation advantages
     * - Objective control
     * 
     * Shows emergent coordination without explicit scripting.
     */
    class FTacticalArenaDemo
    {
    public:
        FTacticalArenaDemo();

        /** Initialize demo with configuration */
        void Initialize(const FArenaDemoConfig& Config);

        /** Run one turn (all agents on active team) */
        bool RunTurn();

        /** Run complete game */
        void RunGame();

        /** Get current state (for visualization) */
        const FArenaGameState& GetState() const { return GameState; }

        /** Get search statistics */
        TArray<FAASSearchResult> GetSearchHistory() const { return SearchHistory; }

        /** Log final results */
        void LogResults() const;

    private:
        FArenaGameState GameState;
        FArenaDemoConfig DemoConfig;
        TSharedPtr<FArenaHeuristics> Heuristics;
        TArray<TSharedPtr<FAASPlanner>> TeamPlanners;
        TArray<FAASSearchResult> SearchHistory;
        int32 CurrentTurn;
        int32 ActiveTeam;

        /** Execute agent turn using AAS */
        FAASAction PlanAgentAction(int32 AgentID);

        /** Setup arena map */
        void SetupArena();

        /** Log turn information */
        void LogTurn() const;
    };

    //--------------------------------------------------------------------------
    // Behavior Tree comparison (optional)
    //--------------------------------------------------------------------------

    /**
     * Simple behavior tree for comparison
     * 
     * Implements basic reactive logic:
     * - If health low: seek cover
     * - If enemy visible: attack
     * - If objective uncontrolled: move to objective
     * - Otherwise: patrol
     * 
     * Used to demonstrate AAS advantages over reactive AI.
     */
    class FSimpleBehaviorTree
    {
    public:
        FSimpleBehaviorTree();

        /** Get action using BT logic */
        FAASAction GetAction(const FArenaGameState& State, int32 AgentID);

        /** Compare BT decision with AAS decision */
        void CompareWithAAS(const FAASAction& AASAction, const FArenaGameState& State, 
            int32 AgentID);

    private:
        /** BT decision nodes */
        bool ShouldSeekCover(const FArenaGameState& State, int32 AgentID) const;
        bool ShouldAttack(const FArenaGameState& State, int32 AgentID) const;
        bool ShouldCaptureObjective(const FArenaGameState& State, int32 AgentID) const;

        /** Get specific actions */
        FAASAction GetSeekCoverAction(const FArenaGameState& State, int32 AgentID) const;
        FAASAction GetAttackAction(const FArenaGameState& State, int32 AgentID) const;
        FAASAction GetMoveToObjectiveAction(const FArenaGameState& State, int32 AgentID) const;
        FAASAction GetPatrolAction(const FArenaGameState& State, int32 AgentID) const;

        /** Find nearest cell of type */
        bool FindNearestCell(const FArenaGameState& State, int32 FromX, int32 FromY,
            EArenaCell CellType, int32& OutX, int32& OutY) const;
    };
}
