// CIIRSubsystem.cpp — Φ=1 shell (no behavior).

#include "CIIR/CIIRSubsystem.h"

void UCIIRSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/CIIR] Initialize"));
}

void UCIIRSubsystem::Deinitialize()
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/CIIR] Deinitialize"));
    Super::Deinitialize();
}

void UCIIRSubsystem::Tick(float /*DeltaTime*/) {}

TStatId UCIIRSubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(UCIIRSubsystem, STATGROUP_Tickables);
}
