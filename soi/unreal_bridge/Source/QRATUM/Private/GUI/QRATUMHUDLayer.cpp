// QRATUMHUDLayer.cpp — Φ=1 shell.

#include "GUI/QRATUMHUDLayer.h"

void UQRATUMHUDLayer::NativeConstruct()
{
    Super::NativeConstruct();
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/GUI] HUD layer constructed"));
}

void UQRATUMHUDLayer::NativeDestruct()
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/GUI] HUD layer destructed"));
    Super::NativeDestruct();
}

void UQRATUMHUDLayer::RegisterPanel(UQRATUMPanelWidget* Panel)
{
    if (Panel)
    {
        Panels.Add(Panel);
    }
}

int32 UQRATUMHUDLayer::GetPanelLiveCount() const
{
    int32 Count = 0;
    for (const TObjectPtr<UQRATUMPanelWidget>& P : Panels)
    {
        if (P && P->bLive) ++Count;
    }
    return Count;
}
