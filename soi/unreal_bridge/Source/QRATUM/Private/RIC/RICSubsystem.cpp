// RICSubsystem.cpp — Φ=2: bias-corrected dispatch [Ch.27, Thm 27.3].
// score_c = score_r / (1 + k_bias * w_age)

#include "RIC/RICSubsystem.h"

void URICSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);

    DispatchMatrix.Cells.SetNum(4);
    DispatchMatrix.Cells[0].Source = EQRATUMDispatchSource::MVRI;
    DispatchMatrix.Cells[1].Source = EQRATUMDispatchSource::QuaSim;
    DispatchMatrix.Cells[2].Source = EQRATUMDispatchSource::QuBIC;
    DispatchMatrix.Cells[3].Source = EQRATUMDispatchSource::RMHD;
    for (FDispatchMatrixCell& C : DispatchMatrix.Cells) { C.Weight = 0.25f; C.LastWriteTick = 0; }
    GlobalTick = 0;

    UE_LOG(LogTemp, Log, TEXT("[QRATUM/RIC] Initialize (4-cell DispatchMatrix)"));
}

void URICSubsystem::Deinitialize()
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/RIC] Deinitialize"));
    Super::Deinitialize();
}

void URICSubsystem::Tick(float /*DeltaTime*/)
{
    ++GlobalTick;
    constexpr float KBias = 0.05f;
    float Sum = 0.0f;
    for (FDispatchMatrixCell& C : DispatchMatrix.Cells)
    {
        const int64 Age = GlobalTick - C.LastWriteTick;
        const float WAge = static_cast<float>(Age);
        const float Corrected = C.Weight / (1.0f + KBias * WAge);
        C.Weight = Corrected;
        Sum += C.Weight;
    }
    if (Sum > 1e-6f)
    {
        for (FDispatchMatrixCell& C : DispatchMatrix.Cells) C.Weight /= Sum;
    }
}

TStatId URICSubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(URICSubsystem, STATGROUP_Tickables);
}

void URICSubsystem::SubmitObservation(EQRATUMDispatchSource Source, float Score)
{
    for (FDispatchMatrixCell& C : DispatchMatrix.Cells)
    {
        if (C.Source == Source)
        {
            C.Weight        = FMath::Clamp(Score, 0.0f, 1.0f);
            C.LastWriteTick = GlobalTick;
            return;
        }
    }
}
