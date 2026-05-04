// CGLSubsystem.cpp — Φ=1 shell (no behavior).

#include "CGL/CGLSubsystem.h"

void UCGLSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/CGL] Initialize"));
}

void UCGLSubsystem::Deinitialize()
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/CGL] Deinitialize"));
    Super::Deinitialize();
}

void UCGLSubsystem::Tick(float /*DeltaTime*/) {}

TStatId UCGLSubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(UCGLSubsystem, STATGROUP_Tickables);
}
