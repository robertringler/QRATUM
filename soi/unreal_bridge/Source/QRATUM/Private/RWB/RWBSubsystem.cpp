// RWBSubsystem.cpp — Φ=2: synthetic deterministic sensor stream.

#include "RWB/RWBSubsystem.h"
#include "MVRI/MVRISubsystem.h"
#include "Engine/World.h"

void URWBSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);
    Sequence       = 0;
    AccumulatedTime = 0.0f;
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/RWB] Initialize"));
}

void URWBSubsystem::Deinitialize()
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/RWB] Deinitialize"));
    Super::Deinitialize();
}

void URWBSubsystem::Tick(float DeltaTime)
{
    AccumulatedTime += DeltaTime;

    FRWBPacket Packet;
    Packet.Sequence         = ++Sequence;
    Packet.TimestampSeconds = AccumulatedTime;
    Packet.SensorVector.SetNum(8);
    for (int32 i = 0; i < 8; ++i)
    {
        const float Phase = AccumulatedTime + 0.25f * i;
        Packet.SensorVector[i] = FMath::Sin(Phase);
    }
    Packet.CanonicalHashHex = FString::Printf(TEXT("%016llx"), static_cast<uint64>(Sequence));

    OnPacketEmitted.Broadcast(Packet);

    if (UWorld* World = GetWorld())
    {
        if (UMVRISubsystem* MVRI = World->GetSubsystem<UMVRISubsystem>())
        {
            MVRI->Ingest(Packet);
        }
    }
}

TStatId URWBSubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(URWBSubsystem, STATGROUP_Tickables);
}
