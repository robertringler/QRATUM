// RMHDSubsystem.cpp — Φ=2: Sweet–Parker reconnection rate V_in/V_A ≈ S^{-1/2}
// [Ch.13, Thm 13.2]. Uses Lundquist S = 1e6 ⇒ rate = 1e-3.

#include "RMHD/RMHDSubsystem.h"

void URMHDSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);
    constexpr double Lundquist = 1.0e6;
    ReconnectionRate = static_cast<float>(1.0 / FMath::Sqrt(Lundquist));
    PsiFlux = 1.0f;
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/RMHD] V_in/V_A = %.3e (Sweet-Parker)"), ReconnectionRate);
}

void URMHDSubsystem::Deinitialize()
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/RMHD] Deinitialize"));
    Super::Deinitialize();
}

void URMHDSubsystem::Tick(float DeltaTime)
{
    PsiFlux = FMath::Max(0.0f, PsiFlux - ReconnectionRate * DeltaTime);
}

TStatId URMHDSubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(URMHDSubsystem, STATGROUP_Tickables);
}
