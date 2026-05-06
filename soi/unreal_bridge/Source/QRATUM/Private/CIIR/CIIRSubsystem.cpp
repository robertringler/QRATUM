// CIIRSubsystem.cpp — Φ=2 deterministic observer fixed-point kernel.
// φ_{n+1} = T(φ_n) where T is a contraction on R^4 (‖A‖ < 1).
// Convergence: |φ_n - φ*| < 1e-6 within 1000 iterations [Ch.22, Thm 22.1].

#include "CIIR/CIIRSubsystem.h"

namespace
{
    static FORCEINLINE void ApplyContractionMap(float* InOut)
    {
        const float A[4][4] = {
            { 0.55f, -0.10f,  0.05f,  0.00f },
            { 0.10f,  0.50f, -0.05f,  0.00f },
            { 0.00f,  0.05f,  0.45f,  0.10f },
            { 0.00f,  0.00f, -0.10f,  0.40f },
        };
        const float b[4] = { 0.30f, -0.20f, 0.15f, 0.05f };
        float Out[4] = { b[0], b[1], b[2], b[3] };
        for (int32 i = 0; i < 4; ++i)
        {
            for (int32 j = 0; j < 4; ++j)
            {
                Out[i] += A[i][j] * InOut[j];
            }
        }
        for (int32 i = 0; i < 4; ++i) InOut[i] = Out[i];
    }
}

void UCIIRSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);

    float Phi[4]     = { 0.0f, 0.0f, 0.0f, 0.0f };
    float PhiPrev[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float Delta = 0.0f;
    int32 Iter = 0;
    constexpr int32 MaxIter = 1000;
    constexpr float Tol = 1.0e-6f;

    for (; Iter < MaxIter; ++Iter)
    {
        for (int32 i = 0; i < 4; ++i) PhiPrev[i] = Phi[i];
        ApplyContractionMap(Phi);
        Delta = 0.0f;
        for (int32 i = 0; i < 4; ++i)
        {
            const float d = Phi[i] - PhiPrev[i];
            Delta += d * d;
        }
        Delta = FMath::Sqrt(Delta);
        if (Delta < Tol) { ++Iter; break; }
    }

    PhiFixedPoint[0] = Phi[0];
    PhiFixedPoint[1] = Phi[1];
    PhiFixedPoint[2] = Phi[2];
    PhiFixedPoint[3] = Phi[3];
    LastPhi[0] = Phi[0]; LastPhi[1] = Phi[1]; LastPhi[2] = Phi[2]; LastPhi[3] = Phi[3];

    LastState.EpochTick         = 0;
    LastState.Phi               = FMath::Sqrt(Phi[0]*Phi[0] + Phi[1]*Phi[1] + Phi[2]*Phi[2] + Phi[3]*Phi[3]);
    LastState.DeltaWNorm        = Delta;
    LastState.ActiveConstraints = 4;

    UE_LOG(LogTemp, Log, TEXT("[QRATUM/CIIR] phi* converged in %d iters, |Delta|=%.3e"), Iter, Delta);
}

void UCIIRSubsystem::Deinitialize()
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/CIIR] Deinitialize"));
    Super::Deinitialize();
}

void UCIIRSubsystem::Tick(float /*DeltaTime*/)
{
    // Live observer: Phi <- 0.99 Phi + 0.01 Phi*.
    constexpr float Decay = 0.99f;
    float Phi[4];
    for (int32 i = 0; i < 4; ++i)
    {
        Phi[i] = Decay * LastPhi[i] + (1.0f - Decay) * PhiFixedPoint[i];
    }

    float Delta = 0.0f;
    for (int32 i = 0; i < 4; ++i)
    {
        const float d = Phi[i] - LastPhi[i];
        Delta += d * d;
    }
    Delta = FMath::Sqrt(Delta);

    for (int32 i = 0; i < 4; ++i) LastPhi[i] = Phi[i];

    LastState.EpochTick        += 1;
    LastState.Phi               = FMath::Sqrt(Phi[0]*Phi[0] + Phi[1]*Phi[1] + Phi[2]*Phi[2] + Phi[3]*Phi[3]);
    LastState.DeltaWNorm        = Delta;
    LastState.ActiveConstraints = 4;

    OnTopologyUpdated.Broadcast(LastState);
}

TStatId UCIIRSubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(UCIIRSubsystem, STATGROUP_Tickables);
}
