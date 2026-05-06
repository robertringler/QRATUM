// QuaSimSubsystem.cpp — Φ=2: Bell-state CHSH probe S = 2*sqrt(2) [Ch.10, Thm 10.1].
// E(a,b) = -cos(a-b) for the singlet; S = E(a,b)-E(a,b')+E(a',b)+E(a',b').
// Choosing (a,a',b,b') = (0, π/2, π/4, 3π/4) yields S = 2*sqrt(2).

#include "QuaSim/QuaSimSubsystem.h"

namespace
{
    static FORCEINLINE float SingletCorrelation(float A, float B)
    {
        return -FMath::Cos(A - B);
    }
}

void UQuaSimSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);
    LastResult.S            = 0.0f;
    LastResult.Sigma        = 0.0f;
    LastResult.SampleCount  = 0;
    LastResult.bQuantumBound = false;
    TickCounter = 0;
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/QuaSim] Initialize"));
}

void UQuaSimSubsystem::Deinitialize()
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/QuaSim] Deinitialize"));
    Super::Deinitialize();
}

void UQuaSimSubsystem::Tick(float /*DeltaTime*/)
{
    ++TickCounter;
    if ((TickCounter % 100) != 0) return;

    constexpr float a  = 0.0f;
    constexpr float ap = UE_HALF_PI;
    constexpr float b  = UE_PI * 0.25f;
    constexpr float bp = UE_PI * 0.75f;

    const float Eab   = SingletCorrelation(a,  b);
    const float Eabp  = SingletCorrelation(a,  bp);
    const float Eapb  = SingletCorrelation(ap, b);
    const float Eapbp = SingletCorrelation(ap, bp);
    const float S     = FMath::Abs(Eab - Eabp + Eapb + Eapbp);

    LastResult.S            = S;
    LastResult.Sigma        = 0.0f;
    LastResult.SampleCount += 1;
    LastResult.bQuantumBound = (S > 2.0f) && (S <= 2.0f * UE_SQRT_2 + 1e-4f);
    OnCHSHEvaluated.Broadcast(LastResult);
}

TStatId UQuaSimSubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(UQuaSimSubsystem, STATGROUP_Tickables);
}
