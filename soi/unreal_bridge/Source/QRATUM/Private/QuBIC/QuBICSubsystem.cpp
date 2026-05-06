// QuBICSubsystem.cpp — Φ=2: 64-node ring graph; deterministic activation diffusion.

#include "QuBIC/QuBICSubsystem.h"

void UQuBICSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);

    constexpr int32 N = 64;
    Graph.Nodes.Reset();
    Graph.EdgesFrom.Reset();
    Graph.EdgesTo.Reset();

    for (int32 i = 0; i < N; ++i)
    {
        FGenomicNode Node;
        Node.NodeId    = i;
        Node.StratumId = i % 4;
        Node.Weight    = (i == 0) ? 1.0f : 0.0f; // unit impulse at node 0
        Graph.Nodes.Add(Node);
        Graph.EdgesFrom.Add(i);
        Graph.EdgesTo.Add((i + 1) % N);
    }
    Epoch = 0;
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/QuBIC] Initialized %d-node ring"), N);
}

void UQuBICSubsystem::Deinitialize()
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/QuBIC] Deinitialize"));
    Super::Deinitialize();
}

void UQuBICSubsystem::Tick(float /*DeltaTime*/)
{
    const int32 N = Graph.Nodes.Num();
    if (N == 0) return;

    TArray<float> Next; Next.SetNumZeroed(N);
    for (int32 e = 0; e < Graph.EdgesFrom.Num(); ++e)
    {
        const int32 From = static_cast<int32>(Graph.EdgesFrom[e]);
        const int32 To   = static_cast<int32>(Graph.EdgesTo[e]);
        if (Graph.Nodes.IsValidIndex(From) && Next.IsValidIndex(To))
        {
            Next[To] += 0.5f * Graph.Nodes[From].Weight;
        }
    }
    for (int32 i = 0; i < N; ++i) Next[i] += 0.5f * Graph.Nodes[i].Weight;

    float Sum = 0.0f;
    for (float V : Next) Sum += V;
    if (Sum > 1e-9f)
    {
        for (float& V : Next) V /= Sum;
    }
    for (int32 i = 0; i < N; ++i) Graph.Nodes[i].Weight = Next[i];
    ++Epoch;
}

TStatId UQuBICSubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(UQuBICSubsystem, STATGROUP_Tickables);
}
