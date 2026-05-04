// CRSSubsystem.cpp — Φ=1 shell (no behavior).

#include "CRS/CRSSubsystem.h"

void UCRSSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/CRS] Initialize"));
}

void UCRSSubsystem::Deinitialize()
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/CRS] Deinitialize"));
    Super::Deinitialize();
}

void UCRSSubsystem::Tick(float /*DeltaTime*/) {}

TStatId UCRSSubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(UCRSSubsystem, STATGROUP_Tickables);
}

void UCRSSubsystem::UpdateStratumBoundary(const FWhitneyStratum& /*Stratum*/)
{
    // Φ=2: recompute admissible-action set against CGL spectrum.
}
