#pragma once

#include "CoreMinimal.h"
#include "MemoryStructs.generated.h"

UENUM(BlueprintType)
enum class EQRATUMEventType : uint8
{
	Seen,
	Heard,
	Damaged,
	Helped,
	Threatened
};

USTRUCT(BlueprintType)
struct FMemoryEntry
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	int32 ActorID = -1;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	FName ActorTag = NAME_None;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	EQRATUMEventType EventType;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	FVector Location = FVector::ZeroVector;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	float Timestamp = 0.f;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	float Salience = 0.f;
};

USTRUCT(BlueprintType)
struct FConstraintVector
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	float HealthWeight = 1.0f;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	float ThreatWeight = 1.0f;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	float DistanceWeight = 1.0f;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	float AggressionBias = 1.0f;
};

USTRUCT(BlueprintType)
struct FQRATUMIntent
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	FName IntentName;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	int32 TargetActorID;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	FVector TargetLocation;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	float BasePriority;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	float TTL = 5.0f;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	float TimeAdded = 0.0f;
};
