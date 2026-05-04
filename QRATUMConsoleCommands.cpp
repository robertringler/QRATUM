// Place this in your QRATUMCoreModule.cpp or equivalent module startup file.
// Registers QRATUM.Debug 1 / QRATUM.Debug 0 console commands.

#include "QRATUMDeveloperSettings.h"
#include "Blueprint/UserWidget.h"
#include "Engine/Engine.h"
#include "GameFramework/PlayerController.h"

static UUserWidget* ActiveDebugWidget = nullptr;

static void ToggleQRATUMDebug(UWorld* World, bool bEnable)
{
	if (!World) return;

	APlayerController* PC = World->GetFirstPlayerController();
	if (!PC) return;

	if (bEnable)
	{
		if (ActiveDebugWidget == nullptr)
		{
			const UQRATUMDeveloperSettings* Settings = GetDefault<UQRATUMDeveloperSettings>();

			if (!Settings || !Settings->DebugWidgetClass)
			{
				UE_LOG(LogTemp, Error,
					TEXT("QRATUM.Debug: DebugWidgetClass not assigned in Project Settings -> QRATUM."));
				return;
			}

			ActiveDebugWidget = CreateWidget<UUserWidget>(PC, Settings->DebugWidgetClass);

			if (ActiveDebugWidget)
			{
				ActiveDebugWidget->AddToViewport();
			}
		}
	}
	else
	{
		if (ActiveDebugWidget)
		{
			ActiveDebugWidget->RemoveFromParent();
			ActiveDebugWidget = nullptr;
		}
	}
}

static FAutoConsoleCommand QRATUMDebugOn(
	TEXT("QRATUM.Debug 1"),
	TEXT("Enable QRATUM Debug Overlay. Requires DebugWidgetClass set in Project Settings."),
	FConsoleCommandDelegate::CreateLambda([]()
	{
		if (GEngine)
		{
			if (UWorld* World = GEngine->GetCurrentPlayWorld())
			{
				ToggleQRATUMDebug(World, true);
			}
		}
	})
);

static FAutoConsoleCommand QRATUMDebugOff(
	TEXT("QRATUM.Debug 0"),
	TEXT("Disable QRATUM Debug Overlay."),
	FConsoleCommandDelegate::CreateLambda([]()
	{
		if (GEngine)
		{
			if (UWorld* World = GEngine->GetCurrentPlayWorld())
			{
				ToggleQRATUMDebug(World, false);
			}
		}
	})
);
