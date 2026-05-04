#pragma once

#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "MemoryStructs.h"
#include "QRATUMDebugWidget.generated.h"

class UQRATUMBrainComponent;
class UQRATUMMemoryComponent;
class UQRATUMPerceptionComponent;

UCLASS()
class UQRATUMDebugWidget : public UUserWidget
{
	GENERATED_BODY()

public:

	// Call this from the console command or Blueprint after adding widget to viewport.
	// Binds all delegates. No tick. No polling.
	UFUNCTION(BlueprintCallable, Category="QRATUM|Debug")
	void BindToOwner(AActor* Owner);

protected:

	virtual void NativeConstruct() override;

private:

	UPROPERTY()
	AActor* BoundOwner;

	UPROPERTY()
	UQRATUMBrainComponent* Brain;

	UPROPERTY()
	UQRATUMMemoryComponent* Memory;

	UPROPERTY()
	UQRATUMPerceptionComponent* Perception;

	// UMG BindWidget — must match widget name in Blueprint
	UPROPERTY(BlueprintReadOnly, meta=(BindWidget), meta=(AllowPrivateAccess="true"))
	class UTextBlock* CurrentIntentText;

	// UMG BindWidget — must match widget name in Blueprint
	UPROPERTY(BlueprintReadOnly, meta=(BindWidget), meta=(AllowPrivateAccess="true"))
	class UTextBlock* LastPerceptionText;

	// UMG BindWidget — must match widget name in Blueprint
	UPROPERTY(BlueprintReadOnly, meta=(BindWidget), meta=(AllowPrivateAccess="true"))
	class UVerticalBox* MemoryList;

	UFUNCTION()
	void HandleDecision(FName NewIntent);

	UFUNCTION()
	void HandleMemoryUpdated();

	UFUNCTION()
	void HandlePerception(AActor* Actor);

	void RefreshMemoryUI();
};
