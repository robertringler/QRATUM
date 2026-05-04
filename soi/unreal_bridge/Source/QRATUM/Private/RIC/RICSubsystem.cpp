// RICSubsystem.cpp — Φ=1 shell (4-cell DispatchMatrix initialized empty).

#include "RIC/RICSubsystem.h"

void URICSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);

    DispatchMatrix.Cells.SetNum(4);
    DispatchMatrix.Cells[0].Source = EQRATUMDispatchSource::MVRI;
    DispatchMatrix.Cells[1].Source = EQRATUMDispatchSource::QuaSim;
    DispatchMatrix.Cells[2].Source = EQRATUMDispatchSource::QuBIC;
    DispatchMatrix.Cells[3].Source = EQRATUMDispatchSource::RMHD;

    UE_LOG(LogTemp, Log, TEXT("[QRATUM/RIC] Initialize (4-cell DispatchMatrix)"));
}

void URICSubsystem::Deinitialize()
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/RIC] Deinitialize"));
    Super::Deinitialize();
}

void URICSubsystem::Tick(float /*DeltaTime*/) {}

TStatId URICSubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(URICSubsystem, STATGROUP_Tickables);
}
