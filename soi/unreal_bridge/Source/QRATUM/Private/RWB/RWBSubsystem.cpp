// RWBSubsystem.cpp — Φ=1 shell (no behavior).

#include "RWB/RWBSubsystem.h"

void URWBSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/RWB] Initialize"));
}

void URWBSubsystem::Deinitialize()
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/RWB] Deinitialize"));
    Super::Deinitialize();
}

void URWBSubsystem::Tick(float /*DeltaTime*/)
{
    // Φ=2 will poll sensors and emit FRWBPacket here.
}

TStatId URWBSubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(URWBSubsystem, STATGROUP_Tickables);
}
