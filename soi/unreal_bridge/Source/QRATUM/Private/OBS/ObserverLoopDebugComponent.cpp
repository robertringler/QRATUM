// ObserverLoopDebugComponent.cpp — Φ=1 shell (no behavior).

#include "OBS/ObserverLoopDebugComponent.h"

UObserverLoopDebugComponent::UObserverLoopDebugComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
}

void UObserverLoopDebugComponent::BeginPlay()
{
    Super::BeginPlay();
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/OBS] BeginPlay"));
}

void UObserverLoopDebugComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/OBS] EndPlay"));
    Super::EndPlay(EndPlayReason);
}

void UObserverLoopDebugComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
}

void UObserverLoopDebugComponent::HandleObserverLoopState(const FObserverLoopState& /*State*/)
{
    // Φ=2: feed PhiTrace ring buffer.
}
