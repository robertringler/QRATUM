#include "QRATUMPerceptionComponent.h"
#include "GameFramework/Actor.h"
#include "Kismet/KismetSystemLibrary.h"

UQRATUMPerceptionComponent::UQRATUMPerceptionComponent()
{
	PrimaryComponentTick.bCanEverTick = false; // event-driven only
}

void UQRATUMPerceptionComponent::BeginPlay()
{
	Super::BeginPlay();
}

void UQRATUMPerceptionComponent::ProcessSight()
{
	if (!GetOwner()) return;

	UWorld* World = GetWorld();
	if (!World) return;

	TArray<AActor*> IgnoredActors;
	IgnoredActors.Add(GetOwner());

	TArray<AActor*> OutActors;

	UKismetSystemLibrary::SphereOverlapActors(
		World,
		GetOwner()->GetActorLocation(),
		SightRange,
		PerceptionObjectTypes,
		nullptr,
		IgnoredActors,
		OutActors
	);

	for (AActor* Actor : OutActors)
	{
		if (!Actor) continue;

		if (!IsInSightCone(Actor)) continue;

		FQRATUMPerceptionEvent Event;
		Event.Actor     = Actor;
		Event.Location  = Actor->GetActorLocation();
		Event.Strength  = FVector::Dist(GetOwner()->GetActorLocation(), Actor->GetActorLocation());
		Event.EventType = "Sighted";

		EmitPerception(Event);
	}
}

void UQRATUMPerceptionComponent::ProcessHearing(FQRATUMPerceptionEvent Event)
{
	Event.EventType = "Heard";
	EventQueue.Add(Event);
	EmitPerception(Event);
}

void UQRATUMPerceptionComponent::EmitPerception(const FQRATUMPerceptionEvent& Event)
{
	// ARCHITECTURE: Perception does not know about BrainComponent.
	// Bind OnPerceivedActor -> QRATUMBrainComponent::AddIntent in Blueprint.
	// Example mappings: Sighted -> Chase, Heard -> Investigate, Threat -> Flee
	OnPerceivedActor.Broadcast(Event.Actor);
}

bool UQRATUMPerceptionComponent::IsInSightCone(AActor* Target) const
{
	if (!GetOwner() || !Target) return false;

	FVector OwnerLoc  = GetOwner()->GetActorLocation();
	FVector Forward   = GetOwner()->GetActorForwardVector();
	FVector ToTarget  = (Target->GetActorLocation() - OwnerLoc).GetSafeNormal();

	float Distance = FVector::Dist(OwnerLoc, Target->GetActorLocation());
	if (Distance > SightRange) return false;

	float Dot          = FVector::DotProduct(Forward, ToTarget);
	float AngleDegrees = FMath::RadiansToDegrees(FMath::Acos(Dot));

	return AngleDegrees <= SightAngle;
}
