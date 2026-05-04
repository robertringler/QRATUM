// CGLSubsystem.h — UCGLSubsystem (Φ=1 shell).
// Owner: A-09. Outbound: OnRankDeficiencyDetected → CRS.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "CGLSubsystem.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnRankDeficiencyDetected, float, ConditionRatio);

UCLASS()
class QRATUM_API UCGLSubsystem : public UTickableWorldSubsystem
{
    GENERATED_BODY()

public:
    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;
    virtual void Tick(float DeltaTime) override;
    virtual TStatId GetStatId() const override;

    UPROPERTY(BlueprintAssignable, Category = "QRATUM|CGL")
    FOnRankDeficiencyDetected OnRankDeficiencyDetected;
};
