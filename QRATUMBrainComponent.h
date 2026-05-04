#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "MemoryStructs.h"
#include "IntentEvaluator.h"
#include "QRATUMBrainComponent.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnDecisionMade, FName, DecisionName);

UCLASS(ClassGroup=(QRATUM), meta=(BlueprintSpawnableComponent))
class UQRATUMBrainComponent : public UActorComponent
{
	GENERATED_BODY()

public:

	UQRATUMBrainComponent();

	// How often (seconds) the decision loop evaluates intents
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="QRATUM|Brain")
	float DecisionInterval = 0.5f;

	// Constraint weights used by IntentEvaluator
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="QRATUM|Brain")
	FConstraintVector ConstraintWeights;

	// Currently active intent name (read-only at runtime)
	UPROPERTY(BlueprintReadOnly, Category="QRATUM|Brain")
	FName CurrentIntent;

	// Fires every decision cycle with the selected intent name
	UPROPERTY(BlueprintAssignable, Category="QRATUM|Brain")
	FOnDecisionMade OnDecisionMade;

	// Add an intent to the evaluation queue
	UFUNCTION(BlueprintCallable, Category="QRATUM|Brain")
	void AddIntent(FQRATUMIntent Intent);

	// Remove all intents matching IntentName
	UFUNCTION(BlueprintCallable, Category="QRATUM|Brain")
	void RemoveIntent(FName IntentName);

	// Force an immediate decision evaluation
	UFUNCTION(BlueprintCallable, Category="QRATUM|Brain")
	void EvaluateDecision();

protected:

	virtual void BeginPlay() override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

private:

	TArray<FQRATUMIntent> IntentQueue;
	float TimeAccumulator;

	FQRATUMIntent SelectBestIntent();
	float ComputeDistanceScore(const FQRATUMIntent& Intent);
	float ComputeHealthProxy();
	float ComputeThreatProxy(const FQRATUMIntent& Intent);
};
