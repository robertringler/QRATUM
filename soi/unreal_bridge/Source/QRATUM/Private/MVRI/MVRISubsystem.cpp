// MVRISubsystem.cpp — Φ=1 shell (no behavior).

#include "MVRI/MVRISubsystem.h"

void UMVRISubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/MVRI] Initialize"));
}

void UMVRISubsystem::Deinitialize()
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/MVRI] Deinitialize"));
    Super::Deinitialize();
}

void UMVRISubsystem::Tick(float /*DeltaTime*/)
{
    // Φ=2: drain RWB ring buffer, run constraint gate, emit FMVRIChannel.
}

TStatId UMVRISubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(UMVRISubsystem, STATGROUP_Tickables);
}

EMVRIInjectionStatus UMVRISubsystem::Ingest(const FRWBPacket& /*Packet*/)
{
    // Φ=1 stub: accept-by-default; Φ=2 will run ER/CS/SS/IC/AUD gate.
    return EMVRIInjectionStatus::Accepted;
}
