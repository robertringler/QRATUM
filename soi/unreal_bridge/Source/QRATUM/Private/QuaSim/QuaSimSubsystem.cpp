// QuaSimSubsystem.cpp — Φ=1 shell (no behavior).

#include "QuaSim/QuaSimSubsystem.h"

void UQuaSimSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/QuaSim] Initialize"));
}

void UQuaSimSubsystem::Deinitialize()
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/QuaSim] Deinitialize"));
    Super::Deinitialize();
}

void UQuaSimSubsystem::Tick(float /*DeltaTime*/) {}

TStatId UQuaSimSubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(UQuaSimSubsystem, STATGROUP_Tickables);
}
