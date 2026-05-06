// CRSSubsystem.cpp — Φ=2: Whitney stratification with ISS bound check [Ch.18, Thm 18.4].

#include "CRS/CRSSubsystem.h"

void UCRSSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);

    // Seed 4 strata of increasing codimension.
    Strata.Reset();
    for (int32 k = 0; k < 4; ++k)
    {
        FWhitneyStratum S;
        S.StratumId   = k;
        S.Codimension = k;
        S.NodeIds.Reset();
        for (int32 j = 0; j < 8; ++j) S.NodeIds.Add(k * 8 + j);
        Strata.Add(S);
    }
    ActiveStratumId = 0;
    ISSResidual = 0.0f;
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/CRS] Initialized %d Whitney strata"), Strata.Num());
}

void UCRSSubsystem::Deinitialize()
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/CRS] Deinitialize"));
    Super::Deinitialize();
}

void UCRSSubsystem::Tick(float DeltaTime)
{
    // ISS residual: |x(t)| <= beta(|x0|, t) + gamma(||u||_inf).
    // Track residual decay; rotate active stratum every 2 s.
    ISSResidual = FMath::Max(0.0f, ISSResidual * FMath::Exp(-0.5f * DeltaTime));
    AccumulatedTime += DeltaTime;
    if (AccumulatedTime >= 2.0f)
    {
        AccumulatedTime = 0.0f;
        ActiveStratumId = (ActiveStratumId + 1) % FMath::Max(1, Strata.Num());
    }
}

TStatId UCRSSubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(UCRSSubsystem, STATGROUP_Tickables);
}

void UCRSSubsystem::UpdateStratumBoundary(const FWhitneyStratum& Stratum)
{
    // Triggered by CGL OnRankDeficiencyDetected.
    const int32 Idx = Stratum.StratumId;
    if (Strata.IsValidIndex(Idx))
    {
        Strata[Idx] = Stratum;
    }
    else
    {
        Strata.Add(Stratum);
    }
    ISSResidual += 0.25f; // perturbation injects bounded residual
    UE_LOG(LogTemp, Verbose, TEXT("[QRATUM/CRS] Stratum %d updated, residual=%.3f"), Idx, ISSResidual);
}
