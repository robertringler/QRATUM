// CIIRSubsystem.h — UCIIRSubsystem (Φ=1 shell).
// Owner: A-02. Outbound: FObserverLoopState → OBS, topology updates → CRS/QuaSim.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "CIIR/ObserverLoopState.h"
#include "CIIRSubsystem.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnTopologyUpdated, const FObserverLoopState&, State);

UCLASS()
class QRATUM_API UCIIRSubsystem : public UTickableWorldSubsystem
{
    GENERATED_BODY()

public:
    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;
    virtual void Tick(float DeltaTime) override;
    virtual TStatId GetStatId() const override;

    UPROPERTY(BlueprintAssignable, Category = "QRATUM|CIIR")
    FOnTopologyUpdated OnTopologyUpdated;
};
