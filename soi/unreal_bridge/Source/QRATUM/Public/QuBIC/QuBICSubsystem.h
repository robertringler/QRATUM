// QuBICSubsystem.h — UQuBICSubsystem (Φ=1 shell).
// Owner: A-06. Outbound: FGenomicGraphState → RIC regime weights.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "QuBIC/GenomicGraphState.h"
#include "QuBICSubsystem.generated.h"

UCLASS()
class QRATUM_API UQuBICSubsystem : public UTickableWorldSubsystem
{
    GENERATED_BODY()

public:
    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;
    virtual void Tick(float DeltaTime) override;
    virtual TStatId GetStatId() const override;
};
