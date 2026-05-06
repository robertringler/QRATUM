// MVRISubsystem.cpp — Φ=2: ER/CS/SS/IC/AUD constraint gate + EMA.

#include "MVRI/MVRISubsystem.h"

void UMVRISubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);
    LastChannel.EMA         = 0.0f;
    LastChannel.EMAVariance = 0.0f;
    LastChannel.GateMask    = 0x1F; // all 5 gates open
    LastChannel.Status      = EMVRIInjectionStatus::Accepted;
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/MVRI] Initialize"));
}

void UMVRISubsystem::Deinitialize()
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/MVRI] Deinitialize"));
    Super::Deinitialize();
}

void UMVRISubsystem::Tick(float /*DeltaTime*/) {}

TStatId UMVRISubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(UMVRISubsystem, STATGROUP_Tickables);
}

EMVRIInjectionStatus UMVRISubsystem::Ingest(const FRWBPacket& Packet)
{
    if (Packet.SensorVector.Num() == 0)
    {
        LastChannel.Status = EMVRIInjectionStatus::RejectedInv;
        OnChannelUpdated.Broadcast(LastChannel);
        return LastChannel.Status;
    }

    float Sum = 0.0f, SumSq = 0.0f;
    for (float V : Packet.SensorVector) { Sum += V; SumSq += V * V; }
    const float N    = static_cast<float>(Packet.SensorVector.Num());
    const float Mean = Sum / N;
    const float Var  = FMath::Max(0.0f, SumSq / N - Mean * Mean);

    constexpr float Alpha = 0.1f;
    LastChannel.EMA         = (1.0f - Alpha) * LastChannel.EMA         + Alpha * Mean;
    LastChannel.EMAVariance = (1.0f - Alpha) * LastChannel.EMAVariance + Alpha * Var;

    // ER (energy/rate) gate: |Mean| <= 10. CS (consistency): Var <= 10.
    if (FMath::Abs(Mean) > 10.0f) { LastChannel.GateMask &= ~0x01; LastChannel.Status = EMVRIInjectionStatus::RejectedGate; }
    else if (Var > 10.0f)         { LastChannel.GateMask &= ~0x02; LastChannel.Status = EMVRIInjectionStatus::RejectedBound; }
    else                          { LastChannel.GateMask  =  0x1F; LastChannel.Status = EMVRIInjectionStatus::Accepted; }

    OnChannelUpdated.Broadcast(LastChannel);
    return LastChannel.Status;
}
