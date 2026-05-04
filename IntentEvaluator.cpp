#include "IntentEvaluator.h"
#include "Math/UnrealMathUtility.h"

float UIntentEvaluator::EvaluateIntent(
	const FQRATUMIntent& Intent,
	const FConstraintVector& Constraints,
	float DistanceToTarget,
	float Health,
	float ThreatLevel
)
{
	float Score = Intent.BasePriority;

	// Distance pressure — farther target reduces score
	Score -= DistanceToTarget * Constraints.DistanceWeight * 0.01f;

	// Normalized health (0.0-1.0) — higher health supports aggressive intents
	Score += Health * Constraints.HealthWeight * 10.0f;

	// Threat penalty — high threat suppresses non-flee intents
	Score -= ThreatLevel * Constraints.ThreatWeight;

	// Aggression multiplier
	Score *= Constraints.AggressionBias;

	return FMath::Clamp(Score, 0.0f, 100.0f);
}
