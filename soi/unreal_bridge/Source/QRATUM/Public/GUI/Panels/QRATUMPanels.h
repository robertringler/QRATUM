// QRATUMPanels.h — 12 panel widget subclasses P-01 .. P-12 (Φ=1 shells).
// Owner: A-10. Each panel binds to a producing ViewModel in Φ=2.

#pragma once

#include "CoreMinimal.h"
#include "GUI/QRATUMPanelWidget.h"
#include "QRATUMPanels.generated.h"

UCLASS() class QRATUM_API UQRATUMPanel_P01_SystemHealth        : public UQRATUMPanelWidget { GENERATED_BODY() };
UCLASS() class QRATUM_API UQRATUMPanel_P02_ConstraintLoad      : public UQRATUMPanelWidget { GENERATED_BODY() };
UCLASS() class QRATUM_API UQRATUMPanel_P03_DispatchMatrix      : public UQRATUMPanelWidget { GENERATED_BODY() };
UCLASS() class QRATUM_API UQRATUMPanel_P04_CHSHHeatmap         : public UQRATUMPanelWidget { GENERATED_BODY() };
UCLASS() class QRATUM_API UQRATUMPanel_P05_GenomicGraph        : public UQRATUMPanelWidget { GENERATED_BODY() };
UCLASS() class QRATUM_API UQRATUMPanel_P06_RMHDFieldLines      : public UQRATUMPanelWidget { GENERATED_BODY() };
UCLASS() class QRATUM_API UQRATUMPanel_P07_PhiTracer           : public UQRATUMPanelWidget { GENERATED_BODY() };
UCLASS() class QRATUM_API UQRATUMPanel_P08_GramianSpectrum     : public UQRATUMPanelWidget { GENERATED_BODY() };
UCLASS() class QRATUM_API UQRATUMPanel_P09_RWBMVRIFeed         : public UQRATUMPanelWidget { GENERATED_BODY() };
UCLASS() class QRATUM_API UQRATUMPanel_P10_FalsifierVerdict    : public UQRATUMPanelWidget { GENERATED_BODY() };
UCLASS() class QRATUM_API UQRATUMPanel_P11_SwarmHealth         : public UQRATUMPanelWidget { GENERATED_BODY() };
UCLASS() class QRATUM_API UQRATUMPanel_P12_Console             : public UQRATUMPanelWidget { GENERATED_BODY() };
