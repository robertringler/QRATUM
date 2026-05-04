#pragma once

#include "CoreMinimal.h"
#include "MemoryStructs.h"
#include "IntentEvaluator.generated.h"

UCLASS()
class UIntentEvaluator : public UObject
{
	GENERATED_BODY()

public:

	// Evaluates a single intent and returns a score (0.0-100.0).
	// Health must be normalized 0.0-1.0.
	// DistanceToTarget is in world units.
	// ThreatLevel is 0.0-100.0.
	UFUNCTION(BlueprintCallable)
	static float EvaluateIntent(
		const FQRATUMIntent& Intent,
		const FConstraintVector& Constraints,
		float DistanceToTarget,
		float Health,
		float ThreatLevel
	);
};
