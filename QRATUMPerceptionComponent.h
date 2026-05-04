#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "QRATUMPerceptionComponent.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnPerceivedActor, AActor*, Actor);

USTRUCT(BlueprintType)
struct FQRATUMPerceptionEvent
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	AActor* Actor = nullptr;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	FVector Location = FVector::ZeroVector;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	float Strength = 0.f;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	FName EventType;
};

UCLASS(ClassGroup=(QRATUM), meta=(BlueprintSpawnableComponent))
class UQRATUMPerceptionComponent : public UActorComponent
{
	GENERATED_BODY()

public:

	UQRATUMPerceptionComponent();

	// Maximum sight distance in world units
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="QRATUM|Perception")
	float SightRange = 2000.0f;

	// Half-angle of sight cone in degrees
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="QRATUM|Perception")
	float SightAngle = 70.0f;

	// Object types included in sight overlap query
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="QRATUM|Perception")
	TArray<TEnumAsByte<EObjectTypeQuery>> PerceptionObjectTypes;

	// Fires when any actor enters the sight cone or hearing range.
	// ARCHITECTURE: Brain connection is delegate-only.
	// Bind OnPerceivedActor -> BrainComponent::AddIntent in Blueprint.
	UPROPERTY(BlueprintAssignable, Category="QRATUM|Perception")
	FOnPerceivedActor OnPerceivedActor;

	// Call on a timer or from an external trigger to run sight evaluation
	UFUNCTION(BlueprintCallable, Category="QRATUM|Perception")
	void ProcessSight();

	// Inject a hearing event directly (no spatial check)
	UFUNCTION(BlueprintCallable, Category="QRATUM|Perception")
	void ProcessHearing(FQRATUMPerceptionEvent Event);

protected:

	virtual void BeginPlay() override;

private:

	TArray<FQRATUMPerceptionEvent> EventQueue;

	bool IsInSightCone(AActor* Target) const;
	void EmitPerception(const FQRATUMPerceptionEvent& Event);
};
