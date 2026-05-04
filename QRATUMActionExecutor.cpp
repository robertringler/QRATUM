#include "QRATUMActionExecutor.h"
#include "AIController.h"
#include "GameFramework/Actor.h"
#include "GameFramework/Pawn.h"
#include "Navigation/PathFollowingComponent.h"

UQRATUMActionExecutor::UQRATUMActionExecutor()
{
	PrimaryComponentTick.bCanEverTick = false;
	CurrentWaypointIndex = 0;
}

void UQRATUMActionExecutor::BeginPlay()
{
	Super::BeginPlay();

	APawn* OwnerPawn = Cast<APawn>(GetOwner());
	if (OwnerPawn)
	{
		CachedController = Cast<AAIController>(OwnerPawn->GetController());
	}
}

void UQRATUMActionExecutor::ExecutePatrol(const TArray<FVector>& Waypoints)
{
	if (!CachedController || Waypoints.Num() == 0) return;

	OnActionStarted.Broadcast("Patrol");

	FVector Target = Waypoints[CurrentWaypointIndex];
	MoveToLocation(Target);

	CurrentWaypointIndex = (CurrentWaypointIndex + 1) % Waypoints.Num();

	OnActionCompleted.Broadcast("Patrol");
}

void UQRATUMActionExecutor::ExecuteChase(AActor* Target)
{
	if (!CachedController || !Target) return;

	OnActionStarted.Broadcast("Chase");
	MoveToLocation(Target->GetActorLocation());
	OnActionCompleted.Broadcast("Chase");
}

void UQRATUMActionExecutor::ExecuteFlee(AActor* Threat)
{
	if (!CachedController || !Threat || !GetOwner()) return;

	OnActionStarted.Broadcast("Flee");

	FVector FleeLocation = GetFleeLocation(Threat);
	MoveToLocation(FleeLocation);

	OnActionCompleted.Broadcast("Flee");
}

void UQRATUMActionExecutor::ExecuteCombat(AActor* Target)
{
	if (!Target) return;

	OnActionStarted.Broadcast("Combat");

	// QRATUM signals only. Bind OnCombatTriggered to your damage / animation system.
	OnCombatTriggered.Broadcast(Target);

	OnActionCompleted.Broadcast("Combat");
}

void UQRATUMActionExecutor::ExecuteSearch(FVector LastKnownLocation)
{
	if (!CachedController || !GetWorld()) return;

	// Clear any existing search timer to prevent stacking
	GetWorld()->GetTimerManager().ClearTimer(SearchTimerHandle);

	OnActionStarted.Broadcast("Search");

	MoveToLocation(LastKnownLocation);

	GetWorld()->GetTimerManager().SetTimer(
		SearchTimerHandle,
		this,
		&UQRATUMActionExecutor::OnSearchTimeoutComplete,
		SearchTimeout,
		false
	);
}

void UQRATUMActionExecutor::OnSearchTimeoutComplete()
{
	OnActionCompleted.Broadcast("Search");
}

void UQRATUMActionExecutor::MoveToLocation(const FVector& Location)
{
	if (!CachedController) return;

	FAIMoveRequest Request;
	Request.SetGoalLocation(Location);
	Request.SetAcceptanceRadius(50.0f);

	CachedController->MoveTo(Request);
}

FVector UQRATUMActionExecutor::GetFleeLocation(AActor* Threat)
{
	if (!GetOwner() || !Threat) return FVector::ZeroVector;

	FVector Dir = (GetOwner()->GetActorLocation() - Threat->GetActorLocation()).GetSafeNormal();
	return GetOwner()->GetActorLocation() + (Dir * 1200.0f);
}
