#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "QRATUMActionExecutor.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnActionStarted,   FName, ActionName);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnActionCompleted, FName, ActionName);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnCombatTriggered, AActor*, Target);

UCLASS(ClassGroup=(QRATUM), meta=(BlueprintSpawnableComponent))
class UQRATUMActionExecutor : public UActorComponent
{
	GENERATED_BODY()

public:

	UQRATUMActionExecutor();

	UPROPERTY(BlueprintAssignable, Category="QRATUM|Actions")
	FOnActionStarted OnActionStarted;

	UPROPERTY(BlueprintAssignable, Category="QRATUM|Actions")
	FOnActionCompleted OnActionCompleted;

	// Fires when combat range is entered. Bind your damage / animation logic here.
	UPROPERTY(BlueprintAssignable, Category="QRATUM|Actions")
	FOnCombatTriggered OnCombatTriggered;

	// How long (seconds) the NPC searches before falling back to Patrol
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="QRATUM|Actions")
	float SearchTimeout = 5.0f;

	UFUNCTION(BlueprintCallable, Category="QRATUM|Actions")
	void ExecutePatrol(const TArray<FVector>& Waypoints);

	UFUNCTION(BlueprintCallable, Category="QRATUM|Actions")
	void ExecuteChase(AActor* Target);

	UFUNCTION(BlueprintCallable, Category="QRATUM|Actions")
	void ExecuteFlee(AActor* Threat);

	// Pure signal. Bind OnCombatTriggered for damage / animation logic.
	UFUNCTION(BlueprintCallable, Category="QRATUM|Actions")
	void ExecuteCombat(AActor* Target);

	// Moves to last known location, then broadcasts OnActionCompleted("Search") on timeout
	UFUNCTION(BlueprintCallable, Category="QRATUM|Actions")
	void ExecuteSearch(FVector LastKnownLocation);

protected:

	virtual void BeginPlay() override;

private:

	UPROPERTY()
	class AAIController* CachedController;

	int32 CurrentWaypointIndex;
	FTimerHandle SearchTimerHandle;

	void MoveToLocation(const FVector& Location);
	FVector GetFleeLocation(AActor* Threat);
	void OnSearchTimeoutComplete();
};
