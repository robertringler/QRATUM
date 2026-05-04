// QuBICSubsystem.cpp — Φ=1 shell (no behavior).

#include "QuBIC/QuBICSubsystem.h"

void UQuBICSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/QuBIC] Initialize"));
}

void UQuBICSubsystem::Deinitialize()
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/QuBIC] Deinitialize"));
    Super::Deinitialize();
}

void UQuBICSubsystem::Tick(float /*DeltaTime*/) {}

TStatId UQuBICSubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(UQuBICSubsystem, STATGROUP_Tickables);
}
