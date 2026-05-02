/// File: Source/QRATUM/Demo/TacticalArenaDemo.cpp
// Copyright QRATUM Platform. All Rights Reserved.
// Demo: Tactical Arena implementation

#include "Demo/TacticalArenaDemo.h"

namespace QRATUM
{
    //--------------------------------------------------------------------------
    // FArenaGameState Implementation
    //--------------------------------------------------------------------------

    FArenaGameState::FArenaGameState()
        : ActiveAgentID(0)
        , TurnNumber(0)
    {
        Grid.SetNum(ArenaWidth * ArenaHeight);
        for (int32 i = 0; i < Grid.Num(); ++i)
        {
            Grid[i] = EArenaCell::Empty;
        }
    }

    FArenaGameState::FArenaGameState(const FArenaGameState& Other)
        : Grid(Other.Grid)
        , Agents(Other.Agents)
        , ActiveAgentID(Other.ActiveAgentID)
        , TurnNumber(Other.TurnNumber)
    {
    }

    void FArenaGameState::Initialize()
    {
        // Clear grid
        for (int32 i = 0; i < Grid.Num(); ++i)
        {
            Grid[i] = EArenaCell::Empty;
        }
        Agents.Empty();
        ActiveAgentID = 0;
        TurnNumber = 0;
    }

    int32 FArenaGameState::GetGridIndex(int32 X, int32 Y) const
    {
        return Y * ArenaWidth + X;
    }

    bool FArenaGameState::IsValidPosition(int32 X, int32 Y) const
    {
        return X >= 0 && X < ArenaWidth && Y >= 0 && Y < ArenaHeight;
    }

    void FArenaGameState::SetCell(int32 X, int32 Y, EArenaCell Cell)
    {
        if (IsValidPosition(X, Y))
        {
            Grid[GetGridIndex(X, Y)] = Cell;
        }
    }

    EArenaCell FArenaGameState::GetCell(int32 X, int32 Y) const
    {
        if (IsValidPosition(X, Y))
        {
            return Grid[GetGridIndex(X, Y)];
        }
        return EArenaCell::Wall;
    }

    void FArenaGameState::AddAgent(const FArenaAgent& Agent)
    {
        Agents.Add(Agent);
    }

    FArenaAgent* FArenaGameState::GetAgent(int32 AgentID)
    {
        for (FArenaAgent& Agent : Agents)
        {
            if (Agent.AgentID == AgentID)
            {
                return &Agent;
            }
        }
        return nullptr;
    }

    const FArenaAgent* FArenaGameState::GetAgent(int32 AgentID) const
    {
        for (const FArenaAgent& Agent : Agents)
        {
            if (Agent.AgentID == AgentID)
            {
                return &Agent;
            }
        }
        return nullptr;
    }

    void FArenaGameState::SetActiveAgent(int32 AgentID)
    {
        ActiveAgentID = AgentID;
    }

    bool FArenaGameState::IsPositionOccupied(int32 X, int32 Y) const
    {
        for (const FArenaAgent& Agent : Agents)
        {
            if (Agent.Health > 0 && Agent.PositionX == X && Agent.PositionY == Y)
            {
                return true;
            }
        }
        return false;
    }

    bool FArenaGameState::HasLineOfSight(int32 X1, int32 Y1, int32 X2, int32 Y2) const
    {
        // Bresenham's line algorithm
        int32 dx = FMath::Abs(X2 - X1);
        int32 dy = FMath::Abs(Y2 - Y1);
        int32 sx = X1 < X2 ? 1 : -1;
        int32 sy = Y1 < Y2 ? 1 : -1;
        int32 err = dx - dy;

        int32 x = X1;
        int32 y = Y1;

        while (x != X2 || y != Y2)
        {
            // Check for walls blocking LOS
            EArenaCell Cell = GetCell(x, y);
            if (Cell == EArenaCell::Wall)
            {
                return false;
            }

            int32 e2 = 2 * err;
            if (e2 > -dy)
            {
                err -= dy;
                x += sx;
            }
            if (e2 < dx)
            {
                err += dx;
                y += sy;
            }
        }

        return true;
    }

    int32 FArenaGameState::CalculateDistance(int32 X1, int32 Y1, int32 X2, int32 Y2) const
    {
        // Manhattan distance
        return FMath::Abs(X2 - X1) + FMath::Abs(Y2 - Y1);
    }

    uint64 FArenaGameState::GetStateHash() const
    {
        uint64 Hash = 0;

        // Hash grid
        Hash = DeterministicHash(Grid.GetData(), Grid.Num() * sizeof(EArenaCell));

        // Hash agents
        for (const FArenaAgent& Agent : Agents)
        {
            Hash = HashCombine(Hash, Agent.AgentID);
            Hash = HashCombine(Hash, Agent.PositionX);
            Hash = HashCombine(Hash, Agent.PositionY);
            Hash = HashCombine(Hash, Agent.Health);
            Hash = HashCombine(Hash, Agent.ActionPoints);
        }

        // Hash active agent and turn
        Hash = HashCombine(Hash, ActiveAgentID);
        Hash = HashCombine(Hash, TurnNumber);

        return Hash;
    }

    void FArenaGameState::GetLegalActions(TArray<FAASAction>& OutActions) const
    {
        OutActions.Empty();

        const FArenaAgent* ActiveAgent = GetAgent(ActiveAgentID);
        if (!ActiveAgent || ActiveAgent->Health <= 0 || ActiveAgent->ActionPoints <= 0)
        {
            return;
        }

        const int32 FromPos = ActiveAgent->PositionY * ArenaWidth + ActiveAgent->PositionX;

        // Movement actions (if AP >= 1)
        if (ActiveAgent->ActionPoints >= 1)
        {
            static const int32 DX[] = {-1, 0, 1, 0};
            static const int32 DY[] = {0, -1, 0, 1};

            for (int32 i = 0; i < 4; ++i)
            {
                int32 NewX = ActiveAgent->PositionX + DX[i];
                int32 NewY = ActiveAgent->PositionY + DY[i];

                if (IsValidPosition(NewX, NewY))
                {
                    EArenaCell Cell = GetCell(NewX, NewY);
                    if (Cell != EArenaCell::Wall && !IsPositionOccupied(NewX, NewY))
                    {
                        FAASAction MoveAction;
                        MoveAction.From = FromPos;
                        MoveAction.To = NewY * ArenaWidth + NewX;
                        MoveAction.TypeFlags = ACTION_MOVE;
                        MoveAction.ActionID = OutActions.Num();

                        // Higher prior for cover and high ground
                        float Prior = 0.3f;
                        if (Cell == EArenaCell::Cover) Prior = 0.5f;
                        if (Cell == EArenaCell::HighGround) Prior = 0.6f;
                        if (Cell == EArenaCell::Objective) Prior = 0.7f;
                        MoveAction.Prior = FFixedPoint32::FromFloat(Prior);

                        OutActions.Add(MoveAction);
                    }
                }
            }
        }

        // Attack actions (if AP >= 1)
        if (ActiveAgent->ActionPoints >= 1)
        {
            for (const FArenaAgent& Target : Agents)
            {
                if (Target.TeamID != ActiveAgent->TeamID && Target.Health > 0)
                {
                    if (HasLineOfSight(ActiveAgent->PositionX, ActiveAgent->PositionY,
                                       Target.PositionX, Target.PositionY))
                    {
                        FAASAction AttackAction;
                        AttackAction.From = FromPos;
                        AttackAction.To = Target.PositionY * ArenaWidth + Target.PositionX;
                        AttackAction.TypeFlags = ACTION_ATTACK;
                        AttackAction.ActionID = OutActions.Num();

                        // Higher prior for low-health targets
                        float HealthRatio = static_cast<float>(Target.Health) / Target.MaxHealth;
                        float Prior = 0.8f - 0.3f * HealthRatio;
                        AttackAction.Prior = FFixedPoint32::FromFloat(Prior);

                        // Static score based on potential damage
                        int32 Distance = CalculateDistance(ActiveAgent->PositionX, ActiveAgent->PositionY,
                                                          Target.PositionX, Target.PositionY);
                        int32 BaseDamage = 30 - Distance * 2;
                        if (Target.bInCover) BaseDamage /= 2;
                        if (ActiveAgent->bOnHighGround) BaseDamage = BaseDamage * 3 / 2;
                        AttackAction.StaticScore = FFixedPoint32::FromInt(BaseDamage);

                        OutActions.Add(AttackAction);
                    }
                }
            }
        }

        // Take cover action
        EArenaCell CurrentCell = GetCell(ActiveAgent->PositionX, ActiveAgent->PositionY);
        if (CurrentCell == EArenaCell::Cover && !ActiveAgent->bInCover && 
            ActiveAgent->ActionPoints >= 1)
        {
            FAASAction CoverAction;
            CoverAction.From = FromPos;
            CoverAction.To = FromPos;
            CoverAction.TypeFlags = ACTION_TAKE_COVER;
            CoverAction.ActionID = OutActions.Num();
            CoverAction.Prior = FFixedPoint32::FromFloat(0.4f);
            OutActions.Add(CoverAction);
        }

        // Sort by action ID for deterministic ordering
        OutActions.Sort([](const FAASAction& A, const FAASAction& B)
        {
            return A < B;
        });
    }

    TUniquePtr<IAASGameState> FArenaGameState::ApplyAction(const FAASAction& Action) const
    {
        TUniquePtr<FArenaGameState> NewState = MakeUnique<FArenaGameState>(*this);
        
        FArenaAgent* Agent = NewState->GetAgent(ActiveAgentID);
        if (!Agent)
        {
            return NewState;
        }

        if (Action.TypeFlags & ACTION_MOVE)
        {
            // Move to new position
            Agent->PositionX = Action.To % ArenaWidth;
            Agent->PositionY = Action.To / ArenaWidth;
            Agent->ActionPoints -= 1;
            Agent->bInCover = false;

            // Check new cell
            EArenaCell Cell = NewState->GetCell(Agent->PositionX, Agent->PositionY);
            Agent->bOnHighGround = (Cell == EArenaCell::HighGround);
        }
        else if (Action.TypeFlags & ACTION_ATTACK)
        {
            // Find target
            int32 TargetX = Action.To % ArenaWidth;
            int32 TargetY = Action.To / ArenaWidth;

            for (FArenaAgent& Target : NewState->Agents)
            {
                if (Target.PositionX == TargetX && Target.PositionY == TargetY &&
                    Target.TeamID != Agent->TeamID)
                {
                    // Calculate damage
                    int32 Distance = CalculateDistance(Agent->PositionX, Agent->PositionY,
                                                      Target.PositionX, Target.PositionY);
                    int32 Damage = 30 - Distance * 2;
                    if (Target.bInCover) Damage /= 2;
                    if (Agent->bOnHighGround) Damage = Damage * 3 / 2;
                    Damage = FMath::Max(5, Damage);

                    Target.Health = FMath::Max(0, Target.Health - Damage);
                    break;
                }
            }

            Agent->ActionPoints -= 1;
        }
        else if (Action.TypeFlags & ACTION_TAKE_COVER)
        {
            Agent->bInCover = true;
            Agent->ActionPoints -= 1;
        }

        return NewState;
    }

    bool FArenaGameState::IsTerminal() const
    {
        // Game ends when one team is eliminated
        int32 Team0Alive = 0;
        int32 Team1Alive = 0;

        for (const FArenaAgent& Agent : Agents)
        {
            if (Agent.Health > 0)
            {
                if (Agent.TeamID == 0) ++Team0Alive;
                else ++Team1Alive;
            }
        }

        return Team0Alive == 0 || Team1Alive == 0;
    }

    FFixedPoint32 FArenaGameState::GetTerminalValue() const
    {
        const FArenaAgent* ActiveAgent = GetAgent(ActiveAgentID);
        if (!ActiveAgent)
        {
            return FFixedPoint32::Zero();
        }

        int32 MyTeamAlive = 0;
        int32 EnemyTeamAlive = 0;

        for (const FArenaAgent& Agent : Agents)
        {
            if (Agent.Health > 0)
            {
                if (Agent.TeamID == ActiveAgent->TeamID)
                    ++MyTeamAlive;
                else
                    ++EnemyTeamAlive;
            }
        }

        if (MyTeamAlive > 0 && EnemyTeamAlive == 0)
        {
            return FFixedPoint32::One(); // Win
        }
        if (MyTeamAlive == 0 && EnemyTeamAlive > 0)
        {
            return FFixedPoint32(-FFixedPoint32::Scale); // Loss
        }

        return FFixedPoint32::Zero(); // Draw
    }

    int32 FArenaGameState::GetActiveAgentID() const
    {
        return ActiveAgentID;
    }

    TUniquePtr<IAASGameState> FArenaGameState::Clone() const
    {
        return MakeUnique<FArenaGameState>(*this);
    }

    int32 FArenaGameState::GetTeamScore(int32 TeamID) const
    {
        int32 Score = 0;
        for (const FArenaAgent& Agent : Agents)
        {
            if (Agent.TeamID == TeamID)
            {
                Score += Agent.Health;
            }
        }
        return Score;
    }

    //--------------------------------------------------------------------------
    // FArenaHeuristics Implementation
    //--------------------------------------------------------------------------

    FArenaHeuristics::FArenaHeuristics()
    {
        RegisterFeatures();
    }

    void FArenaHeuristics::RegisterFeatures()
    {
        // Health advantage
        AddFeature(FAASHeuristicFeature(
            FName("HealthAdvantage"),
            FFixedPoint32::FromFloat(0.3f),
            ExtractHealthAdvantage
        ));

        // Positional advantage
        AddFeature(FAASHeuristicFeature(
            FName("PositionalAdvantage"),
            FFixedPoint32::FromFloat(0.25f),
            ExtractPositionalAdvantage
        ));

        // Objective control
        AddFeature(FAASHeuristicFeature(
            FName("ObjectiveControl"),
            FFixedPoint32::FromFloat(0.2f),
            ExtractObjectiveControl
        ));

        // Cover utilization
        AddFeature(FAASHeuristicFeature(
            FName("CoverUtilization"),
            FFixedPoint32::FromFloat(0.15f),
            ExtractCoverUtilization
        ));

        // Threat level
        AddFeature(FAASHeuristicFeature(
            FName("ThreatLevel"),
            FFixedPoint32::FromFloat(0.1f),
            ExtractThreatLevel
        ));
    }

    FFixedPoint32 FArenaHeuristics::ExtractHealthAdvantage(const IAASGameState& State)
    {
        const FArenaGameState* Arena = static_cast<const FArenaGameState*>(&State);
        if (!Arena)
        {
            return FFixedPoint32::Zero();
        }

        const FArenaAgent* Active = Arena->GetAgent(Arena->GetActiveAgentID());
        if (!Active)
        {
            return FFixedPoint32::Zero();
        }

        int32 MyTeamHealth = Arena->GetTeamScore(Active->TeamID);
        int32 EnemyTeamHealth = Arena->GetTeamScore(1 - Active->TeamID);

        int32 TotalHealth = MyTeamHealth + EnemyTeamHealth;
        if (TotalHealth == 0)
        {
            return FFixedPoint32::Zero();
        }

        float Advantage = static_cast<float>(MyTeamHealth - EnemyTeamHealth) / TotalHealth;
        return FFixedPoint32::FromFloat(Advantage);
    }

    FFixedPoint32 FArenaHeuristics::ExtractPositionalAdvantage(const IAASGameState& State)
    {
        const FArenaGameState* Arena = static_cast<const FArenaGameState*>(&State);
        if (!Arena)
        {
            return FFixedPoint32::Zero();
        }

        const FArenaAgent* Active = Arena->GetAgent(Arena->GetActiveAgentID());
        if (!Active)
        {
            return FFixedPoint32::Zero();
        }

        // Count high ground and cover positions
        int32 MyAdvantage = 0;
        int32 EnemyAdvantage = 0;

        // Check agents' positions
        for (int32 ID = 0; ID < 6; ++ID)
        {
            const FArenaAgent* Agent = Arena->GetAgent(ID);
            if (Agent && Agent->Health > 0)
            {
                EArenaCell Cell = Arena->GetCell(Agent->PositionX, Agent->PositionY);
                int32 Value = 0;
                if (Cell == EArenaCell::HighGround) Value += 2;
                if (Agent->bInCover) Value += 1;

                if (Agent->TeamID == Active->TeamID)
                    MyAdvantage += Value;
                else
                    EnemyAdvantage += Value;
            }
        }

        int32 Total = MyAdvantage + EnemyAdvantage;
        if (Total == 0)
        {
            return FFixedPoint32::Zero();
        }

        float Advantage = static_cast<float>(MyAdvantage - EnemyAdvantage) / FMath::Max(Total, 1);
        return FFixedPoint32::FromFloat(FMath::Clamp(Advantage, -1.0f, 1.0f));
    }

    FFixedPoint32 FArenaHeuristics::ExtractObjectiveControl(const IAASGameState& State)
    {
        // Count agents near objectives
        const FArenaGameState* Arena = static_cast<const FArenaGameState*>(&State);
        if (!Arena)
        {
            return FFixedPoint32::Zero();
        }

        // Placeholder - would check objective positions
        return FFixedPoint32::Zero();
    }

    FFixedPoint32 FArenaHeuristics::ExtractCoverUtilization(const IAASGameState& State)
    {
        const FArenaGameState* Arena = static_cast<const FArenaGameState*>(&State);
        if (!Arena)
        {
            return FFixedPoint32::Zero();
        }

        const FArenaAgent* Active = Arena->GetAgent(Arena->GetActiveAgentID());
        if (!Active)
        {
            return FFixedPoint32::Zero();
        }

        float Utilization = Active->bInCover ? 1.0f : 0.0f;
        return FFixedPoint32::FromFloat(Utilization);
    }

    FFixedPoint32 FArenaHeuristics::ExtractThreatLevel(const IAASGameState& State)
    {
        const FArenaGameState* Arena = static_cast<const FArenaGameState*>(&State);
        if (!Arena)
        {
            return FFixedPoint32::Zero();
        }

        const FArenaAgent* Active = Arena->GetAgent(Arena->GetActiveAgentID());
        if (!Active)
        {
            return FFixedPoint32::Zero();
        }

        // Count enemies with LOS
        int32 ThreatsToUs = 0;
        int32 TargetsForUs = 0;

        for (int32 ID = 0; ID < 6; ++ID)
        {
            const FArenaAgent* Agent = Arena->GetAgent(ID);
            if (Agent && Agent->Health > 0 && Agent->AgentID != Active->AgentID)
            {
                bool HasLOS = Arena->GetCell(Agent->PositionX, Agent->PositionY) != EArenaCell::Wall;
                
                if (Agent->TeamID != Active->TeamID)
                {
                    if (HasLOS) ++ThreatsToUs;
                }
                else
                {
                    if (HasLOS) ++TargetsForUs;
                }
            }
        }

        // Negative threat (we're exposed) vs positive (we can attack)
        float Threat = static_cast<float>(TargetsForUs - ThreatsToUs) / 3.0f;
        return FFixedPoint32::FromFloat(FMath::Clamp(Threat, -1.0f, 1.0f));
    }

    //--------------------------------------------------------------------------
    // FArenaPhaseDetector Implementation
    //--------------------------------------------------------------------------

    EGamePhase FArenaPhaseDetector::DetectPhase(const IAASGameState& State) const
    {
        const FArenaGameState* Arena = static_cast<const FArenaGameState*>(&State);
        if (!Arena)
        {
            return EGamePhase::Middlegame;
        }

        // Count alive agents
        int32 TotalAlive = 0;
        for (int32 ID = 0; ID < 6; ++ID)
        {
            const FArenaAgent* Agent = Arena->GetAgent(ID);
            if (Agent && Agent->Health > 0)
            {
                ++TotalAlive;
            }
        }

        if (TotalAlive <= 2)
        {
            return EGamePhase::Endgame;
        }
        if (TotalAlive >= 5)
        {
            return EGamePhase::Opening;
        }

        return EGamePhase::Middlegame;
    }

    //--------------------------------------------------------------------------
    // FTacticalArenaDemo Implementation
    //--------------------------------------------------------------------------

    FTacticalArenaDemo::FTacticalArenaDemo()
        : CurrentTurn(0)
        , ActiveTeam(0)
    {
    }

    void FTacticalArenaDemo::Initialize(const FArenaDemoConfig& Config)
    {
        DemoConfig = Config;
        CurrentTurn = 0;
        ActiveTeam = 0;
        SearchHistory.Empty();

        // Initialize heuristics
        Heuristics = MakeShared<FArenaHeuristics>();

        // Initialize planners for each team
        TeamPlanners.Empty();
        for (int32 Team = 0; Team < 2; ++Team)
        {
            TSharedPtr<FAASPlanner> Planner = MakeShared<FAASPlanner>();
            
            FAASSearchConfig SearchConfig;
            SearchConfig.BaseDepth = DemoConfig.SearchDepth;
            SearchConfig.TimeLimitMs = DemoConfig.SearchTimeMs;
            
            Planner->Initialize(Heuristics, SearchConfig);
            TeamPlanners.Add(Planner);
        }

        // Setup arena
        SetupArena();

        UE_LOG(LogTemp, Log, TEXT("[QRATUM Demo] Tactical Arena initialized"));
        UE_LOG(LogTemp, Log, TEXT("  Agents per team: %d"), DemoConfig.NumAgentsPerTeam);
        UE_LOG(LogTemp, Log, TEXT("  Search depth: %d"), DemoConfig.SearchDepth);
        UE_LOG(LogTemp, Log, TEXT("  Search time: %.1f ms"), DemoConfig.SearchTimeMs);
    }

    void FTacticalArenaDemo::SetupArena()
    {
        GameState.Initialize();

        // Create arena map with cover and high ground
        // Place walls
        for (int32 y = 5; y <= 10; ++y)
        {
            GameState.SetCell(7, y, EArenaCell::Wall);
            GameState.SetCell(8, y, EArenaCell::Wall);
        }

        // Place cover
        GameState.SetCell(3, 3, EArenaCell::Cover);
        GameState.SetCell(3, 12, EArenaCell::Cover);
        GameState.SetCell(12, 3, EArenaCell::Cover);
        GameState.SetCell(12, 12, EArenaCell::Cover);
        GameState.SetCell(5, 7, EArenaCell::Cover);
        GameState.SetCell(10, 8, EArenaCell::Cover);

        // Place high ground
        GameState.SetCell(4, 8, EArenaCell::HighGround);
        GameState.SetCell(11, 7, EArenaCell::HighGround);

        // Place objectives
        GameState.SetCell(4, 4, EArenaCell::Objective);
        GameState.SetCell(11, 11, EArenaCell::Objective);

        // Add agents
        // Team 0 (left side)
        for (int32 i = 0; i < DemoConfig.NumAgentsPerTeam; ++i)
        {
            FArenaAgent Agent;
            Agent.AgentID = i;
            Agent.TeamID = 0;
            Agent.PositionX = 2;
            Agent.PositionY = 4 + i * 4;
            Agent.Health = 100;
            Agent.MaxHealth = 100;
            Agent.ActionPoints = 2;
            Agent.MaxActionPoints = 2;
            GameState.AddAgent(Agent);
        }

        // Team 1 (right side)
        for (int32 i = 0; i < DemoConfig.NumAgentsPerTeam; ++i)
        {
            FArenaAgent Agent;
            Agent.AgentID = DemoConfig.NumAgentsPerTeam + i;
            Agent.TeamID = 1;
            Agent.PositionX = 13;
            Agent.PositionY = 4 + i * 4;
            Agent.Health = 100;
            Agent.MaxHealth = 100;
            Agent.ActionPoints = 2;
            Agent.MaxActionPoints = 2;
            GameState.AddAgent(Agent);
        }

        GameState.SetActiveAgent(0);
    }

    FAASAction FTacticalArenaDemo::PlanAgentAction(int32 AgentID)
    {
        FArenaAgent* Agent = GameState.GetAgent(AgentID);
        if (!Agent || Agent->Health <= 0)
        {
            return FAASAction();
        }

        GameState.SetActiveAgent(AgentID);

        FAASPlanner* Planner = TeamPlanners[Agent->TeamID].Get();
        
        FAASPlanningContext Context;
        Context.CurrentState = &GameState;
        Context.AvailableTimeMs = DemoConfig.SearchTimeMs;
        Context.FrameBudgetMs = DemoConfig.SearchTimeMs; // Single-frame search for demo

        // Reset and plan
        Planner->Reset();
        while (!Planner->PlanStep(Context))
        {
            // Continue until complete
        }

        FAASPlannedAction Result = Planner->GetPlannedAction();
        SearchHistory.Add(Planner->GetSearchStats());

        if (DemoConfig.bLogMoves)
        {
            UE_LOG(LogTemp, Log, TEXT("  Agent %d (Team %d): %d -> %d [flags=0x%X, conf=%.2f, depth=%d]"),
                AgentID, Agent->TeamID,
                Result.PrimaryAction.From, Result.PrimaryAction.To,
                Result.PrimaryAction.TypeFlags,
                Result.Confidence.ToFloat(),
                Result.LookaheadDepth);
        }

        return Result.PrimaryAction;
    }

    bool FTacticalArenaDemo::RunTurn()
    {
        if (GameState.IsTerminal() || CurrentTurn >= DemoConfig.MaxTurns)
        {
            return false;
        }

        ++CurrentTurn;

        if (DemoConfig.bLogMoves)
        {
            LogTurn();
        }

        // Run all agents for current team
        for (int32 i = 0; i < DemoConfig.NumAgentsPerTeam; ++i)
        {
            int32 AgentID = ActiveTeam * DemoConfig.NumAgentsPerTeam + i;
            FArenaAgent* Agent = GameState.GetAgent(AgentID);

            if (Agent && Agent->Health > 0)
            {
                // Reset action points for turn
                Agent->ActionPoints = Agent->MaxActionPoints;

                // Plan and execute actions
                while (Agent->ActionPoints > 0)
                {
                    FAASAction Action = PlanAgentAction(AgentID);
                    
                    if (Action.From == 0 && Action.To == 0 && Action.TypeFlags == 0)
                    {
                        break; // No valid action
                    }

                    // Apply action
                    TUniquePtr<IAASGameState> NewState = GameState.ApplyAction(Action);
                    GameState = *static_cast<FArenaGameState*>(NewState.Get());

                    if (GameState.IsTerminal())
                    {
                        return false;
                    }
                }
            }
        }

        // Switch teams
        ActiveTeam = 1 - ActiveTeam;

        return !GameState.IsTerminal();
    }

    void FTacticalArenaDemo::RunGame()
    {
        UE_LOG(LogTemp, Log, TEXT("[QRATUM Demo] Starting tactical arena game..."));

        while (RunTurn())
        {
            // Continue until game ends
        }

        LogResults();
    }

    void FTacticalArenaDemo::LogTurn() const
    {
        UE_LOG(LogTemp, Log, TEXT("--- Turn %d (Team %d) ---"), CurrentTurn, ActiveTeam);
        
        // Log team status
        for (int32 Team = 0; Team < 2; ++Team)
        {
            int32 TotalHealth = GameState.GetTeamScore(Team);
            UE_LOG(LogTemp, Log, TEXT("  Team %d total health: %d"), Team, TotalHealth);
        }
    }

    void FTacticalArenaDemo::LogResults() const
    {
        UE_LOG(LogTemp, Log, TEXT("=== GAME OVER ==="));
        UE_LOG(LogTemp, Log, TEXT("Turns played: %d"), CurrentTurn);

        // Final scores
        int32 Team0Score = GameState.GetTeamScore(0);
        int32 Team1Score = GameState.GetTeamScore(1);

        UE_LOG(LogTemp, Log, TEXT("Team 0 remaining health: %d"), Team0Score);
        UE_LOG(LogTemp, Log, TEXT("Team 1 remaining health: %d"), Team1Score);

        if (Team0Score > Team1Score)
        {
            UE_LOG(LogTemp, Log, TEXT("Winner: Team 0"));
        }
        else if (Team1Score > Team0Score)
        {
            UE_LOG(LogTemp, Log, TEXT("Winner: Team 1"));
        }
        else
        {
            UE_LOG(LogTemp, Log, TEXT("Result: Draw"));
        }

        // Search statistics
        if (!SearchHistory.IsEmpty())
        {
            int64 TotalNodes = 0;
            double TotalTime = 0.0;
            int32 TotalSearches = SearchHistory.Num();

            for (const FAASSearchResult& Result : SearchHistory)
            {
                TotalNodes += Result.NodesSearched;
                TotalTime += Result.TimeMs;
            }

            UE_LOG(LogTemp, Log, TEXT("Search Statistics:"));
            UE_LOG(LogTemp, Log, TEXT("  Total searches: %d"), TotalSearches);
            UE_LOG(LogTemp, Log, TEXT("  Total nodes: %lld"), TotalNodes);
            UE_LOG(LogTemp, Log, TEXT("  Total time: %.1f ms"), TotalTime);
            UE_LOG(LogTemp, Log, TEXT("  Avg nodes/search: %lld"), TotalNodes / FMath::Max(TotalSearches, 1));
            UE_LOG(LogTemp, Log, TEXT("  Avg time/search: %.1f ms"), TotalTime / FMath::Max(TotalSearches, 1));
        }
    }

    //--------------------------------------------------------------------------
    // FSimpleBehaviorTree Implementation
    //--------------------------------------------------------------------------

    FSimpleBehaviorTree::FSimpleBehaviorTree()
    {
    }

    FAASAction FSimpleBehaviorTree::GetAction(const FArenaGameState& State, int32 AgentID)
    {
        // BT priority order: Cover > Attack > Objective > Patrol
        if (ShouldSeekCover(State, AgentID))
        {
            return GetSeekCoverAction(State, AgentID);
        }
        if (ShouldAttack(State, AgentID))
        {
            return GetAttackAction(State, AgentID);
        }
        if (ShouldCaptureObjective(State, AgentID))
        {
            return GetMoveToObjectiveAction(State, AgentID);
        }
        return GetPatrolAction(State, AgentID);
    }

    void FSimpleBehaviorTree::CompareWithAAS(const FAASAction& AASAction, 
        const FArenaGameState& State, int32 AgentID)
    {
        FAASAction BTAction = GetAction(State, AgentID);
        FAASBehaviorTreeCompare::LogComparison(AASAction, BTAction, 
            FString::Printf(TEXT("Agent %d decision"), AgentID));
    }

    bool FSimpleBehaviorTree::ShouldSeekCover(const FArenaGameState& State, int32 AgentID) const
    {
        const FArenaAgent* Agent = State.GetAgent(AgentID);
        if (!Agent)
        {
            return false;
        }

        // Seek cover if health below 50%
        return Agent->Health < Agent->MaxHealth / 2;
    }

    bool FSimpleBehaviorTree::ShouldAttack(const FArenaGameState& State, int32 AgentID) const
    {
        const FArenaAgent* Agent = State.GetAgent(AgentID);
        if (!Agent)
        {
            return false;
        }

        // Attack if any enemy is visible
        for (int32 ID = 0; ID < 6; ++ID)
        {
            const FArenaAgent* Target = State.GetAgent(ID);
            if (Target && Target->Health > 0 && Target->TeamID != Agent->TeamID)
            {
                if (State.GetCell(Target->PositionX, Target->PositionY) != EArenaCell::Wall)
                {
                    return true;
                }
            }
        }
        return false;
    }

    bool FSimpleBehaviorTree::ShouldCaptureObjective(const FArenaGameState& State, int32 AgentID) const
    {
        // Always try to capture if not in combat
        return true;
    }

    FAASAction FSimpleBehaviorTree::GetSeekCoverAction(const FArenaGameState& State, int32 AgentID) const
    {
        const FArenaAgent* Agent = State.GetAgent(AgentID);
        if (!Agent)
        {
            return FAASAction();
        }

        int32 CoverX, CoverY;
        if (FindNearestCell(State, Agent->PositionX, Agent->PositionY, 
                           EArenaCell::Cover, CoverX, CoverY))
        {
            FAASAction Action;
            Action.From = Agent->PositionY * FArenaGameState::ArenaWidth + Agent->PositionX;
            Action.To = CoverY * FArenaGameState::ArenaWidth + CoverX;
            Action.TypeFlags = FArenaGameState::ACTION_MOVE;
            return Action;
        }

        return FAASAction();
    }

    FAASAction FSimpleBehaviorTree::GetAttackAction(const FArenaGameState& State, int32 AgentID) const
    {
        const FArenaAgent* Agent = State.GetAgent(AgentID);
        if (!Agent)
        {
            return FAASAction();
        }

        // Attack first visible enemy
        for (int32 ID = 0; ID < 6; ++ID)
        {
            const FArenaAgent* Target = State.GetAgent(ID);
            if (Target && Target->Health > 0 && Target->TeamID != Agent->TeamID)
            {
                FAASAction Action;
                Action.From = Agent->PositionY * FArenaGameState::ArenaWidth + Agent->PositionX;
                Action.To = Target->PositionY * FArenaGameState::ArenaWidth + Target->PositionX;
                Action.TypeFlags = FArenaGameState::ACTION_ATTACK;
                return Action;
            }
        }

        return FAASAction();
    }

    FAASAction FSimpleBehaviorTree::GetMoveToObjectiveAction(const FArenaGameState& State, int32 AgentID) const
    {
        const FArenaAgent* Agent = State.GetAgent(AgentID);
        if (!Agent)
        {
            return FAASAction();
        }

        int32 ObjX, ObjY;
        if (FindNearestCell(State, Agent->PositionX, Agent->PositionY,
                           EArenaCell::Objective, ObjX, ObjY))
        {
            FAASAction Action;
            Action.From = Agent->PositionY * FArenaGameState::ArenaWidth + Agent->PositionX;
            Action.To = ObjY * FArenaGameState::ArenaWidth + ObjX;
            Action.TypeFlags = FArenaGameState::ACTION_MOVE;
            return Action;
        }

        return FAASAction();
    }

    FAASAction FSimpleBehaviorTree::GetPatrolAction(const FArenaGameState& State, int32 AgentID) const
    {
        const FArenaAgent* Agent = State.GetAgent(AgentID);
        if (!Agent)
        {
            return FAASAction();
        }

        // Move towards center
        int32 CenterX = FArenaGameState::ArenaWidth / 2;
        int32 CenterY = FArenaGameState::ArenaHeight / 2;

        int32 DX = FMath::Sign(CenterX - Agent->PositionX);
        int32 DY = FMath::Sign(CenterY - Agent->PositionY);

        int32 NewX = Agent->PositionX + DX;
        int32 NewY = Agent->PositionY + DY;

        if (NewX >= 0 && NewX < FArenaGameState::ArenaWidth &&
            NewY >= 0 && NewY < FArenaGameState::ArenaHeight)
        {
            FAASAction Action;
            Action.From = Agent->PositionY * FArenaGameState::ArenaWidth + Agent->PositionX;
            Action.To = NewY * FArenaGameState::ArenaWidth + NewX;
            Action.TypeFlags = FArenaGameState::ACTION_MOVE;
            return Action;
        }

        return FAASAction();
    }

    bool FSimpleBehaviorTree::FindNearestCell(const FArenaGameState& State, int32 FromX, int32 FromY,
        EArenaCell CellType, int32& OutX, int32& OutY) const
    {
        int32 BestDist = INT32_MAX;
        bool Found = false;

        for (int32 Y = 0; Y < FArenaGameState::ArenaHeight; ++Y)
        {
            for (int32 X = 0; X < FArenaGameState::ArenaWidth; ++X)
            {
                if (State.GetCell(X, Y) == CellType)
                {
                    int32 Dist = FMath::Abs(X - FromX) + FMath::Abs(Y - FromY);
                    if (Dist < BestDist)
                    {
                        BestDist = Dist;
                        OutX = X;
                        OutY = Y;
                        Found = true;
                    }
                }
            }
        }

        return Found;
    }
}
