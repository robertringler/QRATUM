/// File: Source/QRATUM/Public/Core/AASNode.h
// Copyright QRATUM Platform. All Rights Reserved.
// Asymmetric Adaptive Search - Node representation

#pragma once

#include "CoreMinimal.h"
#include "Determinism/DeterministicTypes.h"
#include "Determinism/DeterministicContainers.h"

namespace QRATUM
{
    // Forward declarations
    class FAASSearch;
    struct FAASAction;

    /**
     * Node flag bits for efficient state tracking
     */
    enum class EAASNodeFlags : uint8
    {
        None            = 0,
        Expanded        = 1 << 0,  // Children have been generated
        Terminal        = 1 << 1,  // No further actions possible
        Pruned          = 1 << 2,  // Cut from search tree
        FullyEvaluated  = 1 << 3,  // All descendants searched to depth
        InPrincipalVar  = 1 << 4,  // Part of principal variation
        Transposition   = 1 << 5,  // Reached via transposition
    };

    ENUM_CLASS_FLAGS(EAASNodeFlags)

    /**
     * Transposition table entry type
     */
    enum class ETranspositionType : uint8
    {
        Exact,      // Value is exact
        LowerBound, // Value is lower bound (failed high)
        UpperBound, // Value is upper bound (failed low)
    };

    /**
     * FAASAction - Abstract action representation
     * 
     * Domain-agnostic action type that can represent any game action.
     * Concrete implementations should subclass this for type safety.
     * 
     * Memory: Packed to 32 bytes for cache efficiency
     */
    struct QRATUM_API FAASAction
    {
        /** Unique action identifier within the current state */
        uint32 ActionID;

        /** Source entity/position (domain-specific encoding) */
        uint32 From;

        /** Target entity/position (domain-specific encoding) */
        uint32 To;

        /** Action type flags (domain-specific) */
        uint32 TypeFlags;

        /** Prior probability from policy heuristic [0, 1] */
        FFixedPoint32 Prior;

        /** Cached static evaluation of this action */
        FFixedPoint32 StaticScore;

        /** Domain-specific payload (e.g., promotion piece, ability ID) */
        int32 Payload;

        /** Padding for alignment */
        uint32 Reserved;

        FAASAction()
            : ActionID(0), From(0), To(0), TypeFlags(0)
            , Prior(FFixedPoint32::Zero())
            , StaticScore(FFixedPoint32::Zero())
            , Payload(0), Reserved(0)
        {}

        FAASAction(uint32 InFrom, uint32 InTo, uint32 InTypeFlags = 0)
            : ActionID(0), From(InFrom), To(InTo), TypeFlags(InTypeFlags)
            , Prior(FFixedPoint32::Zero())
            , StaticScore(FFixedPoint32::Zero())
            , Payload(0), Reserved(0)
        {}

        /** Hash for transposition table */
        uint64 GetHash() const
        {
            return DeterministicHash(this, sizeof(FAASAction));
        }

        bool operator==(const FAASAction& Other) const
        {
            return From == Other.From && To == Other.To && TypeFlags == Other.TypeFlags && Payload == Other.Payload;
        }

        bool operator!=(const FAASAction& Other) const
        {
            return !(*this == Other);
        }

        /** Comparison for deterministic ordering */
        bool operator<(const FAASAction& Other) const
        {
            if (From != Other.From) return From < Other.From;
            if (To != Other.To) return To < Other.To;
            if (TypeFlags != Other.TypeFlags) return TypeFlags < Other.TypeFlags;
            return Payload < Other.Payload;
        }
    };

    /**
     * FAASNode - Search tree node
     * 
     * Core data structure for the AAS search tree. Each node represents
     * a game state reachable through a sequence of actions from the root.
     * 
     * Design Decisions:
     * - Intrusive tree structure (parent/child pointers) for efficient traversal
     * - Fixed-point values for determinism
     * - Compact flags for memory efficiency
     * - Explicit visit counts for UCB-style selection
     * 
     * Memory Layout: ~96 bytes per node (cache-line aligned)
     */
    class QRATUM_API FAASNode
    {
    public:
        FAASNode();
        explicit FAASNode(FAASNode* InParent, const FAASAction& InAction);
        ~FAASNode();

        // Prevent copying (tree ownership semantics)
        FAASNode(const FAASNode&) = delete;
        FAASNode& operator=(const FAASNode&) = delete;

        // Move semantics for efficient container operations
        FAASNode(FAASNode&& Other) noexcept;
        FAASNode& operator=(FAASNode&& Other) noexcept;

        /** Get parent node (nullptr for root) */
        FAASNode* GetParent() const { return Parent; }

        /** Get action that led to this node */
        const FAASAction& GetAction() const { return Action; }

        /** Get/set evaluation value */
        FFixedPoint32 GetValue() const { return Value; }
        void SetValue(FFixedPoint32 InValue) { Value = InValue; }

        /** Get visit count */
        uint32 GetVisitCount() const { return VisitCount; }

        /** Increment visit count and update value statistics */
        void RecordVisit(FFixedPoint32 VisitValue);

        /** Get average value (mean of all visits) */
        FFixedPoint32 GetAverageValue() const;

        /** Get depth in tree (0 for root) */
        int32 GetDepth() const { return Depth; }

        /** Check/set flags */
        bool HasFlag(EAASNodeFlags Flag) const { return EnumHasAnyFlags(Flags, Flag); }
        void SetFlag(EAASNodeFlags Flag) { Flags |= Flag; }
        void ClearFlag(EAASNodeFlags Flag) { Flags &= ~Flag; }

        /** State hash for transposition detection */
        uint64 GetStateHash() const { return StateHash; }
        void SetStateHash(uint64 InHash) { StateHash = InHash; }

        /** Child management */
        bool HasChildren() const { return !Children.IsEmpty(); }
        int32 GetChildCount() const { return Children.Num(); }
        
        /** Add a child node, takes ownership */
        FAASNode* AddChild(const FAASAction& ChildAction);

        /** Get children array for iteration (deterministic order) */
        const TArray<FAASNode*>& GetChildren() const { return Children; }

        /** Find child by action */
        FAASNode* FindChild(const FAASAction& ChildAction) const;

        /** Get best child by value (deterministic tiebreaking) */
        FAASNode* GetBestChild() const;

        /** Get best child by UCB1 formula (deterministic tiebreaking) */
        FAASNode* GetBestChildUCB(FFixedPoint32 ExplorationConstant) const;

        /** Detach from parent (for subtree reuse) */
        void DetachFromParent();

        /** Recursively destroy all children */
        void DestroyChildren();

        /** Get principal variation from this node */
        void GetPrincipalVariation(TArray<FAASAction>& OutPV, int32 MaxLength = 20) const;

        /** Serialization for replay/debug */
        void Serialize(FArchive& Ar);

    private:
        /** Parent node (nullptr for root) */
        FAASNode* Parent;

        /** Children in deterministic order (sorted by action) */
        TArray<FAASNode*> Children;

        /** Action that led to this node */
        FAASAction Action;

        /** Best known value (from perspective of side to move) */
        FFixedPoint32 Value;

        /** Sum of all values from visits (for averaging) */
        FFixedPoint32 ValueSum;

        /** Number of times this node was visited */
        uint32 VisitCount;

        /** Depth in tree (0 for root) */
        int32 Depth;

        /** Hash of the game state at this node */
        uint64 StateHash;

        /** Node flags */
        EAASNodeFlags Flags;
    };

    /**
     * FAASTranspositionEntry - Entry in the transposition table
     * 
     * Stores search results for previously visited positions to avoid
     * redundant computation. Uses replacement scheme based on depth.
     */
    struct QRATUM_API FAASTranspositionEntry
    {
        uint64 StateHash;
        FFixedPoint32 Value;
        FAASAction BestAction;
        int32 Depth;
        ETranspositionType Type;

        FAASTranspositionEntry()
            : StateHash(0)
            , Value(FFixedPoint32::Zero())
            , Depth(0)
            , Type(ETranspositionType::Exact)
        {}
    };

    /**
     * Transposition table with deterministic replacement policy
     */
    class QRATUM_API FAASTranspositionTable
    {
    public:
        explicit FAASTranspositionTable(int32 SizeInMB = 64);

        /** Probe the table for an entry */
        bool Probe(uint64 StateHash, FAASTranspositionEntry& OutEntry) const;

        /** Store an entry (may replace existing) */
        void Store(const FAASTranspositionEntry& Entry);

        /** Clear all entries */
        void Clear();

        /** Get statistics */
        int64 GetHitCount() const { return HitCount; }
        int64 GetProbeCount() const { return ProbeCount; }
        float GetHitRate() const { return ProbeCount > 0 ? static_cast<float>(HitCount) / ProbeCount : 0.f; }

    private:
        TArray<FAASTranspositionEntry> Table;
        int32 TableSize;
        mutable int64 HitCount;
        mutable int64 ProbeCount;
    };
}
