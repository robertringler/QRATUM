#include "QRATUMBrainComponent.h"
#include "QRATUMHealthProvider.h"
#include "GameFramework/Actor.h"

UQRATUMBrainComponent::UQRATUMBrainComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	TimeAccumulator = 0.0f;
}

void UQRATUMBrainComponent::BeginPlay()
{
	Super::BeginPlay();
}

void UQRATUMBrainComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	TimeAccumulator += DeltaTime;

	if (TimeAccumulator >= DecisionInterval)
	{
		TimeAccumulator = 0.0f;
		EvaluateDecision();
	}
}

void UQRATUMBrainComponent::AddIntent(FQRATUMIntent Intent)
{
	if (!GetWorld()) return;

	Intent.TimeAdded = GetWorld()->GetTimeSeconds();
	IntentQueue.Add(Intent);
}

void UQRATUMBrainComponent::RemoveIntent(FName IntentName)
{
	for (int32 i = IntentQueue.Num() - 1; i >= 0; --i)
	{
		if (IntentQueue[i].IntentName == IntentName)
		{
			IntentQueue.RemoveAt(i);
		}
	}
}

void UQRATUMBrainComponent::EvaluateDecision()
{
	if (IntentQueue.Num() == 0)
	{
		CurrentIntent = "Idle";
		OnDecisionMade.Broadcast(CurrentIntent);
		return;
	}

	float CurrentTime = GetWorld()->GetTimeSeconds();

	// Prune expired intents by TTL
	for (int32 i = IntentQueue.Num() - 1; i >= 0; --i)
	{
		if ((CurrentTime - IntentQueue[i].TimeAdded) > IntentQueue[i].TTL)
		{
			IntentQueue.RemoveAt(i);
		}
	}

	if (IntentQueue.Num() == 0)
	{
		CurrentIntent = "Fallback_Idle";
		OnDecisionMade.Broadcast(CurrentIntent);
		return;
	}

	FQRATUMIntent BestIntent = SelectBestIntent();

	CurrentIntent = BestIntent.IntentName;
	OnDecisionMade.Broadcast(CurrentIntent);
}

FQRATUMIntent UQRATUMBrainComponent::SelectBestIntent()
{
	float BestScore = -FLT_MAX;
	FQRATUMIntent BestIntent;

	for (const FQRATUMIntent& Intent : IntentQueue)
	{
		float Distance = ComputeDistanceScore(Intent);
		float Health   = ComputeHealthProxy();
		float Threat   = ComputeThreatProxy(Intent);

		float Score = UIntentEvaluator::EvaluateIntent(
			Intent,
			ConstraintWeights,
			Distance,
			Health,
			Threat
		);

		if (Score > BestScore)
		{
			BestScore  = Score;
			BestIntent = Intent;
		}
	}

	return BestIntent;
}

float UQRATUMBrainComponent::ComputeDistanceScore(const FQRATUMIntent& Intent)
{
	if (!GetOwner()) return 0.0f;
	return FVector::Dist(GetOwner()->GetActorLocation(), Intent.TargetLocation);
}

float UQRATUMBrainComponent::ComputeHealthProxy()
{
	if (!GetOwner()) return 1.0f;

	IQRATUMHealthProvider* HP = Cast<IQRATUMHealthProvider>(GetOwner());
	if (HP)
	{
		return FMath::Clamp(HP->GetNormalizedHealth(), 0.0f, 1.0f);
	}

	return 1.0f;
}

float UQRATUMBrainComponent::ComputeThreatProxy(const FQRATUMIntent& Intent)
{
	if (!GetOwner()) return 0.0f;

	float Dist = FVector::Dist(GetOwner()->GetActorLocation(), Intent.TargetLocation);
	return FMath::Clamp(1000.0f / (Dist + 1.0f), 0.0f, 100.0f);
}
