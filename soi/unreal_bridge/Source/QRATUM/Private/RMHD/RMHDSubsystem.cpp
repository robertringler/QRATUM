// RMHDSubsystem.cpp — Φ=1 shell (no behavior).

#include "RMHD/RMHDSubsystem.h"

void URMHDSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/RMHD] Initialize"));
}

void URMHDSubsystem::Deinitialize()
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/RMHD] Deinitialize"));
    Super::Deinitialize();
}

void URMHDSubsystem::Tick(float /*DeltaTime*/) {}

TStatId URMHDSubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(URMHDSubsystem, STATGROUP_Tickables);
}
