// ObserverLoopDebugComponent.h — UObserverLoopDebugComponent (Φ=1 shell).
// Owner: A-09. Subscribes to CIIR FObserverLoopState; renders Φ trace.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "CIIR/ObserverLoopState.h"
#include "ObserverLoopDebugComponent.generated.h"

UCLASS(ClassGroup = (QRATUM), meta = (BlueprintSpawnableComponent))
class QRATUM_API UObserverLoopDebugComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UObserverLoopDebugComponent();

    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

    UFUNCTION()
    void HandleObserverLoopState(const FObserverLoopState& State);
};
