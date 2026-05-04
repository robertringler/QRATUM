// QRATUMPanelWidget.h — UQRATUMPanelWidget base widget (Φ=1 shell).
// Owner: A-10. All 12 panel subclasses inherit from this.

#pragma once

#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "QRATUMPanelWidget.generated.h"

UCLASS(Abstract, BlueprintType, Blueprintable)
class QRATUM_API UQRATUMPanelWidget : public UUserWidget
{
    GENERATED_BODY()

public:
    UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category = "QRATUM|GUI")
    FName PanelId;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|GUI")
    bool bLive = false;

    UFUNCTION(BlueprintNativeEvent, Category = "QRATUM|GUI")
    void BindViewModel();
    virtual void BindViewModel_Implementation() {}
};
