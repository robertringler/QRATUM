/// File: Source/QRATUM/Private/Core/AASNode.cpp
// Copyright QRATUM Platform. All Rights Reserved.
// Asymmetric Adaptive Search - Node implementation

#include "Core/AASNode.h"
#include "Algo/Sort.h"

namespace QRATUM
{
    //--------------------------------------------------------------------------
    // FAASNode Implementation
    //--------------------------------------------------------------------------

    FAASNode::FAASNode()
        : Parent(nullptr)
        , Value(FFixedPoint32::Zero())
        , ValueSum(FFixedPoint32::Zero())
        , VisitCount(0)
        , Depth(0)
        , StateHash(0)
        , Flags(EAASNodeFlags::None)
    {
    }

    FAASNode::FAASNode(FAASNode* InParent, const FAASAction& InAction)
        : Parent(InParent)
        , Action(InAction)
        , Value(FFixedPoint32::Zero())
        , ValueSum(FFixedPoint32::Zero())
        , VisitCount(0)
        , Depth(InParent ? InParent->Depth + 1 : 0)
        , StateHash(0)
        , Flags(EAASNodeFlags::None)
    {
    }

    FAASNode::~FAASNode()
    {
        DestroyChildren();
    }

    FAASNode::FAASNode(FAASNode&& Other) noexcept
        : Parent(Other.Parent)
        , Children(MoveTemp(Other.Children))
        , Action(Other.Action)
        , Value(Other.Value)
        , ValueSum(Other.ValueSum)
        , VisitCount(Other.VisitCount)
        , Depth(Other.Depth)
        , StateHash(Other.StateHash)
        , Flags(Other.Flags)
    {
        // Update children's parent pointers
        for (FAASNode* Child : Children)
        {
            if (Child)
            {
                Child->Parent = this;
            }
        }
        
        // Clear source
        Other.Parent = nullptr;
        Other.Children.Empty();
    }

    FAASNode& FAASNode::operator=(FAASNode&& Other) noexcept
    {
        if (this != &Other)
        {
            DestroyChildren();
            
            Parent = Other.Parent;
            Children = MoveTemp(Other.Children);
            Action = Other.Action;
            Value = Other.Value;
            ValueSum = Other.ValueSum;
            VisitCount = Other.VisitCount;
            Depth = Other.Depth;
            StateHash = Other.StateHash;
            Flags = Other.Flags;
            
            // Update children's parent pointers
            for (FAASNode* Child : Children)
            {
                if (Child)
                {
                    Child->Parent = this;
                }
            }
            
            // Clear source
            Other.Parent = nullptr;
            Other.Children.Empty();
        }
        return *this;
    }

    void FAASNode::RecordVisit(FFixedPoint32 VisitValue)
    {
        ++VisitCount;
        ValueSum = ValueSum + VisitValue;
        
        // Update best value if this visit is better
        if (VisitCount == 1 || VisitValue > Value)
        {
            Value = VisitValue;
        }
    }

    FFixedPoint32 FAASNode::GetAverageValue() const
    {
        if (VisitCount == 0)
        {
            return FFixedPoint32::Zero();
        }
        return FFixedPoint32(ValueSum.RawValue / static_cast<int32>(VisitCount));
    }

    FAASNode* FAASNode::AddChild(const FAASAction& ChildAction)
    {
        FAASNode* Child = new FAASNode(this, ChildAction);
        Children.Add(Child);
        
        // Sort children by action for deterministic iteration order
        // Uses action's < operator which compares From, To, TypeFlags, Payload
        Algo::Sort(Children, [](const FAASNode* A, const FAASNode* B)
        {
            return A->Action < B->Action;
        });
        
        return Child;
    }

    FAASNode* FAASNode::FindChild(const FAASAction& ChildAction) const
    {
        for (FAASNode* Child : Children)
        {
            if (Child->Action == ChildAction)
            {
                return Child;
            }
        }
        return nullptr;
    }

    FAASNode* FAASNode::GetBestChild() const
    {
        if (Children.IsEmpty())
        {
            return nullptr;
        }

        FAASNode* Best = Children[0];
        
        for (int32 i = 1; i < Children.Num(); ++i)
        {
            FAASNode* Child = Children[i];
            
            // Compare by value first
            if (Child->Value > Best->Value)
            {
                Best = Child;
            }
            // Deterministic tiebreaker: use action ordering
            else if (Child->Value == Best->Value && Child->Action < Best->Action)
            {
                Best = Child;
            }
        }
        
        return Best;
    }

    FAASNode* FAASNode::GetBestChildUCB(FFixedPoint32 ExplorationConstant) const
    {
        if (Children.IsEmpty())
        {
            return nullptr;
        }

        if (VisitCount == 0)
        {
            // No visits yet, return first child deterministically
            return Children[0];
        }

        FAASNode* Best = nullptr;
        FFixedPoint32 BestUCB = FFixedPoint32::Min();
        
        // Precompute sqrt(ln(parent_visits)) as fixed point
        // Using approximation: ln(N) ≈ 0.6931 * log2(N)
        const float LogParentVisits = FMath::Loge(static_cast<float>(VisitCount));
        const float SqrtLog = FMath::Sqrt(LogParentVisits);
        const FFixedPoint32 ExplorationTerm = FFixedPoint32::FromFloat(SqrtLog);

        for (int32 i = 0; i < Children.Num(); ++i)
        {
            FAASNode* Child = Children[i];
            
            FFixedPoint32 UCBValue;
            
            if (Child->VisitCount == 0)
            {
                // Unvisited nodes get maximum priority but with deterministic tiebreak
                UCBValue = FFixedPoint32::Max();
                // Subtract action ID for deterministic ordering among unvisited nodes
                UCBValue.RawValue -= static_cast<int32>(Child->Action.ActionID);
            }
            else
            {
                // UCB1 formula: Q + C * sqrt(ln(N) / n)
                FFixedPoint32 ChildVisits = FFixedPoint32::FromInt(static_cast<int32>(Child->VisitCount));
                FFixedPoint32 Exploration = ExplorationConstant * ExplorationTerm / ChildVisits;
                
                // Use child's prior as additional exploration bonus
                FFixedPoint32 PriorBonus = Child->Action.Prior * ExplorationConstant;
                
                UCBValue = Child->GetAverageValue() + Exploration + PriorBonus;
            }

            // Compare with deterministic tiebreaking
            if (UCBValue > BestUCB || 
                (UCBValue == BestUCB && Best != nullptr && Child->Action < Best->Action))
            {
                BestUCB = UCBValue;
                Best = Child;
            }
        }
        
        return Best ? Best : Children[0];
    }

    void FAASNode::DetachFromParent()
    {
        if (Parent)
        {
            Parent->Children.Remove(this);
            Parent = nullptr;
            Depth = 0;
        }
    }

    void FAASNode::DestroyChildren()
    {
        for (FAASNode* Child : Children)
        {
            delete Child;
        }
        Children.Empty();
    }

    void FAASNode::GetPrincipalVariation(TArray<FAASAction>& OutPV, int32 MaxLength) const
    {
        OutPV.Empty();
        
        const FAASNode* Current = this;
        while (Current && OutPV.Num() < MaxLength)
        {
            FAASNode* Best = Current->GetBestChild();
            if (!Best)
            {
                break;
            }
            
            OutPV.Add(Best->GetAction());
            Current = Best;
        }
    }

    void FAASNode::Serialize(FArchive& Ar)
    {
        Ar << Action.ActionID;
        Ar << Action.From;
        Ar << Action.To;
        Ar << Action.TypeFlags;
        Ar << Action.Prior.RawValue;
        Ar << Action.StaticScore.RawValue;
        Ar << Action.Payload;
        Ar << Value.RawValue;
        Ar << ValueSum.RawValue;
        Ar << VisitCount;
        Ar << Depth;
        Ar << StateHash;
        
        uint8 FlagsValue = static_cast<uint8>(Flags);
        Ar << FlagsValue;
        if (Ar.IsLoading())
        {
            Flags = static_cast<EAASNodeFlags>(FlagsValue);
        }
    }

    //--------------------------------------------------------------------------
    // FAASTranspositionTable Implementation
    //--------------------------------------------------------------------------

    FAASTranspositionTable::FAASTranspositionTable(int32 SizeInMB)
        : HitCount(0)
        , ProbeCount(0)
    {
        // Calculate number of entries based on size
        const int32 BytesPerEntry = sizeof(FAASTranspositionEntry);
        TableSize = (SizeInMB * 1024 * 1024) / BytesPerEntry;
        
        // Ensure power of 2 for efficient modulo via bitwise AND
        TableSize = FMath::RoundUpToPowerOfTwo(TableSize);
        TableSize = FMath::Max(TableSize, 1024); // Minimum 1024 entries
        
        Table.SetNum(TableSize);
        Clear();
        
        UE_LOG(LogTemp, Log, TEXT("[QRATUM] Transposition table initialized: %d entries (%d MB)"),
            TableSize, (TableSize * BytesPerEntry) / (1024 * 1024));
    }

    bool FAASTranspositionTable::Probe(uint64 StateHash, FAASTranspositionEntry& OutEntry) const
    {
        ++ProbeCount;
        
        const int32 Index = static_cast<int32>(StateHash & (TableSize - 1));
        const FAASTranspositionEntry& Entry = Table[Index];
        
        if (Entry.StateHash == StateHash)
        {
            ++HitCount;
            OutEntry = Entry;
            return true;
        }
        
        return false;
    }

    void FAASTranspositionTable::Store(const FAASTranspositionEntry& Entry)
    {
        const int32 Index = static_cast<int32>(Entry.StateHash & (TableSize - 1));
        FAASTranspositionEntry& Existing = Table[Index];
        
        // Replacement policy: always replace if deeper or same hash
        // This prefers deeper searches and avoids stale entries
        if (Existing.StateHash == 0 || 
            Existing.StateHash == Entry.StateHash ||
            Entry.Depth >= Existing.Depth)
        {
            Existing = Entry;
        }
    }

    void FAASTranspositionTable::Clear()
    {
        for (FAASTranspositionEntry& Entry : Table)
        {
            Entry.StateHash = 0;
            Entry.Value = FFixedPoint32::Zero();
            Entry.Depth = 0;
            Entry.Type = ETranspositionType::Exact;
        }
        HitCount = 0;
        ProbeCount = 0;
    }
}
