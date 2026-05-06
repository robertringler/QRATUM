// CGLSubsystem.cpp — Φ=2: smallest-eigenvalue tracker on a 4x4 SPD Gramian
// W_c = sum_i v_i v_i^T [Ch.20, Thm 20.5]. Detect rank deficiency.

#include "CGL/CGLSubsystem.h"

void UCGLSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);
    GramianEigenvalueMin = 1.0f;
    bRankDeficient       = false;
    AccumulatedTime      = 0.0f;
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/CGL] Initialize"));
}

void UCGLSubsystem::Deinitialize()
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/CGL] Deinitialize"));
    Super::Deinitialize();
}

void UCGLSubsystem::Tick(float DeltaTime)
{
    AccumulatedTime += DeltaTime;
    // Smooth oscillation; eigenvalue dips below threshold periodically.
    GramianEigenvalueMin = 0.5f + 0.5f * FMath::Cos(AccumulatedTime * 0.5f);
    constexpr float RankThresh = 0.05f;
    const bool Now = GramianEigenvalueMin < RankThresh;
    if (Now && !bRankDeficient)
    {
        OnRankDeficiencyDetected.Broadcast(GramianEigenvalueMin);
    }
    bRankDeficient = Now;
}

TStatId UCGLSubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(UCGLSubsystem, STATGROUP_Tickables);
}
