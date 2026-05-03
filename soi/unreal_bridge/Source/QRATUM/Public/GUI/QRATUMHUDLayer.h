// QRATUMHUDLayer.h — UQRATUMHUDLayer (Φ=1 shell).
// Owner: A-10. Hosts the 12 P-01..P-12 panel widgets.

#pragma once

#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "GUI/QRATUMPanelWidget.h"
#include "QRATUMHUDLayer.generated.h"

UCLASS(BlueprintType, Blueprintable)
class QRATUM_API UQRATUMHUDLayer : public UUserWidget
{
    GENERATED_BODY()

public:
    virtual void NativeConstruct() override;
    virtual void NativeDestruct() override;

    UFUNCTION(BlueprintCallable, Category = "QRATUM|GUI")
    void RegisterPanel(UQRATUMPanelWidget* Panel);

    UFUNCTION(BlueprintPure, Category = "QRATUM|GUI")
    int32 GetPanelLiveCount() const;

protected:
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|GUI")
    TArray<TObjectPtr<UQRATUMPanelWidget>> Panels;
};
