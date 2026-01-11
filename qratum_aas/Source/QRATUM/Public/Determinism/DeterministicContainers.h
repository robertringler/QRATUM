/// File: Source/QRATUM/Public/Determinism/DeterministicContainers.h
// Copyright QRATUM Platform. All Rights Reserved.
// Deterministic containers for stable iteration order

#pragma once

#include "CoreMinimal.h"
#include "DeterministicTypes.h"

namespace QRATUM
{
    /**
     * Ordered map with deterministic iteration.
     * 
     * Unlike TMap which uses hash-based buckets with arbitrary iteration order,
     * this container maintains insertion order for deterministic traversal.
     * 
     * Performance:
     * - Insert: O(1) amortized
     * - Find: O(n) - linear search
     * - Iteration: O(n) in insertion order
     * 
     * For large collections where find performance matters, consider using
     * a sorted array with binary search instead.
     * 
     * Note: We use a simple array-based implementation because:
     * 1. AAS node counts per search step are typically small (<1000)
     * 2. Insertion order preservation is critical for determinism
     * 3. Memory locality of array traversal benefits cache performance
     */
    template<typename KeyType, typename ValueType>
    class TDeterministicMap
    {
    public:
        struct FPair
        {
            KeyType Key;
            ValueType Value;

            FPair() = default;
            FPair(const KeyType& InKey, const ValueType& InValue)
                : Key(InKey), Value(InValue) {}
        };

        TDeterministicMap() = default;

        /** Add or update a key-value pair */
        void Add(const KeyType& Key, const ValueType& Value)
        {
            for (int32 i = 0; i < Pairs.Num(); ++i)
            {
                if (Pairs[i].Key == Key)
                {
                    Pairs[i].Value = Value;
                    return;
                }
            }
            Pairs.Emplace(Key, Value);
        }

        /** Find value by key, returns nullptr if not found */
        ValueType* Find(const KeyType& Key)
        {
            for (int32 i = 0; i < Pairs.Num(); ++i)
            {
                if (Pairs[i].Key == Key)
                {
                    return &Pairs[i].Value;
                }
            }
            return nullptr;
        }

        const ValueType* Find(const KeyType& Key) const
        {
            for (int32 i = 0; i < Pairs.Num(); ++i)
            {
                if (Pairs[i].Key == Key)
                {
                    return &Pairs[i].Value;
                }
            }
            return nullptr;
        }

        /** Check if key exists */
        bool Contains(const KeyType& Key) const
        {
            return Find(Key) != nullptr;
        }

        /** Remove key, returns true if found */
        bool Remove(const KeyType& Key)
        {
            for (int32 i = 0; i < Pairs.Num(); ++i)
            {
                if (Pairs[i].Key == Key)
                {
                    Pairs.RemoveAt(i);
                    return true;
                }
            }
            return false;
        }

        /** Get value by key, with default if not found */
        ValueType FindOrDefault(const KeyType& Key, const ValueType& Default = ValueType()) const
        {
            const ValueType* Found = Find(Key);
            return Found ? *Found : Default;
        }

        /** Bracket operator for convenient access */
        ValueType& operator[](const KeyType& Key)
        {
            for (int32 i = 0; i < Pairs.Num(); ++i)
            {
                if (Pairs[i].Key == Key)
                {
                    return Pairs[i].Value;
                }
            }
            // Add new entry with default value
            int32 Index = Pairs.Emplace(Key, ValueType());
            return Pairs[Index].Value;
        }

        /** Get number of elements */
        int32 Num() const { return Pairs.Num(); }

        /** Check if empty */
        bool IsEmpty() const { return Pairs.Num() == 0; }

        /** Clear all entries */
        void Empty() { Pairs.Empty(); }

        /** Reserve capacity */
        void Reserve(int32 Capacity) { Pairs.Reserve(Capacity); }

        /** Iteration support - deterministic order guaranteed */
        TArray<FPair>::TIterator begin() { return Pairs.begin(); }
        TArray<FPair>::TIterator end() { return Pairs.end(); }
        TArray<FPair>::TConstIterator begin() const { return Pairs.begin(); }
        TArray<FPair>::TConstIterator end() const { return Pairs.end(); }

        /** Get underlying array for direct access */
        const TArray<FPair>& GetPairs() const { return Pairs; }

    private:
        TArray<FPair> Pairs;
    };

    /**
     * Priority queue with deterministic ordering.
     * 
     * Uses a binary heap with stable comparison. When two elements have
     * equal priority, insertion order is used as tiebreaker to ensure
     * deterministic extraction order.
     * 
     * Default is max-heap (highest priority first). Use TDeterministicMinHeap
     * for min-heap behavior.
     */
    template<typename ElementType, typename PriorityType = float>
    class TDeterministicPriorityQueue
    {
    public:
        struct FEntry
        {
            ElementType Element;
            PriorityType Priority;
            uint64 InsertionOrder; // Tiebreaker for determinism

            bool operator<(const FEntry& Other) const
            {
                if (Priority != Other.Priority)
                {
                    return Priority < Other.Priority;
                }
                // Equal priority: use insertion order (earlier = higher)
                return InsertionOrder > Other.InsertionOrder;
            }
        };

        TDeterministicPriorityQueue() : InsertionCounter(0) {}

        /** Add element with given priority */
        void Push(const ElementType& Element, PriorityType Priority)
        {
            FEntry Entry;
            Entry.Element = Element;
            Entry.Priority = Priority;
            Entry.InsertionOrder = InsertionCounter++;
            
            Heap.Push(Entry);
            HeapifyUp(Heap.Num() - 1);
        }

        /** Remove and return highest priority element */
        ElementType Pop()
        {
            check(Heap.Num() > 0);

            ElementType Result = Heap[0].Element;
            
            if (Heap.Num() > 1)
            {
                Heap[0] = Heap.Last();
            }
            Heap.Pop(EAllowShrinking::No);
            
            if (Heap.Num() > 0)
            {
                HeapifyDown(0);
            }
            
            return Result;
        }

        /** Peek at highest priority element without removing */
        const ElementType& Top() const
        {
            check(Heap.Num() > 0);
            return Heap[0].Element;
        }

        /** Get priority of top element */
        PriorityType TopPriority() const
        {
            check(Heap.Num() > 0);
            return Heap[0].Priority;
        }

        int32 Num() const { return Heap.Num(); }
        bool IsEmpty() const { return Heap.Num() == 0; }
        void Empty() { Heap.Empty(); InsertionCounter = 0; }
        void Reserve(int32 Capacity) { Heap.Reserve(Capacity); }

    private:
        TArray<FEntry> Heap;
        uint64 InsertionCounter;

        void HeapifyUp(int32 Index)
        {
            while (Index > 0)
            {
                int32 Parent = (Index - 1) / 2;
                if (Heap[Parent] < Heap[Index])
                {
                    Heap.Swap(Parent, Index);
                    Index = Parent;
                }
                else
                {
                    break;
                }
            }
        }

        void HeapifyDown(int32 Index)
        {
            const int32 Size = Heap.Num();
            while (true)
            {
                int32 Largest = Index;
                int32 Left = 2 * Index + 1;
                int32 Right = 2 * Index + 2;

                if (Left < Size && Heap[Largest] < Heap[Left])
                {
                    Largest = Left;
                }
                if (Right < Size && Heap[Largest] < Heap[Right])
                {
                    Largest = Right;
                }
                
                if (Largest != Index)
                {
                    Heap.Swap(Index, Largest);
                    Index = Largest;
                }
                else
                {
                    break;
                }
            }
        }
    };

    /**
     * Deterministic set with ordered iteration.
     * Simple wrapper around TArray that prevents duplicates.
     */
    template<typename ElementType>
    class TDeterministicSet
    {
    public:
        /** Add element if not present, returns true if added */
        bool Add(const ElementType& Element)
        {
            if (Contains(Element))
            {
                return false;
            }
            Elements.Add(Element);
            return true;
        }

        /** Check if element exists */
        bool Contains(const ElementType& Element) const
        {
            return Elements.Contains(Element);
        }

        /** Remove element, returns true if found */
        bool Remove(const ElementType& Element)
        {
            return Elements.Remove(Element) > 0;
        }

        int32 Num() const { return Elements.Num(); }
        bool IsEmpty() const { return Elements.Num() == 0; }
        void Empty() { Elements.Empty(); }
        void Reserve(int32 Capacity) { Elements.Reserve(Capacity); }

        /** Iteration support */
        auto begin() { return Elements.begin(); }
        auto end() { return Elements.end(); }
        auto begin() const { return Elements.begin(); }
        auto end() const { return Elements.end(); }

        /** Get element by index */
        const ElementType& operator[](int32 Index) const { return Elements[Index]; }

    private:
        TArray<ElementType> Elements;
    };
}
