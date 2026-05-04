#pragma once

#include "CoreMinimal.h"
#include "Engine/DeveloperSettings.h"
#include "QRATUMDeveloperSettings.generated.h"

// Exposed in Project Settings -> Plugins -> QRATUM
// Assign DebugWidgetClass to your WBP_QRATUMDebug asset before
// using the QRATUM.Debug console command.
UCLASS(Config=Game, DefaultConfig, meta=(DisplayName="QRATUM"))
class UQRATUMDeveloperSettings : public UDeveloperSettings
{
	GENERATED_BODY()

public:

	UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category="QRATUM Debug",
		meta=(DisplayName="Debug Widget Class",
			  ToolTip="Assign WBP_QRATUMDebug here. Required for QRATUM.Debug console command."))
	TSubclassOf<class UUserWidget> DebugWidgetClass;
};
