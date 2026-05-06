// CRSSubsystem.h — UCRSSubsystem (Φ=1 shell).
// Owner: A-03. Inbound: CGL rank-deficiency. Outbound: stratum updates → QuBIC, RMHD.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "CRS/WhitneyStratum.h"
#include "CRSSubsystem.generated.h"

UCLASS()
class QRATUM_API UCRSSubsystem : public UTickableWorldSubsystem
{
    GENERATED_BODY()

public:
    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;
    virtual void Tick(float DeltaTime) override;
    virtual TStatId GetStatId() const override;

    UFUNCTION(BlueprintCallable, Category = "QRATUM|CRS")
    void UpdateStratumBoundary(const FWhitneyStratum& Stratum);

    UFUNCTION(BlueprintPure, Category = "QRATUM|CRS")
    int32 GetActiveStratumId() const { return ActiveStratumId; }

    UFUNCTION(BlueprintPure, Category = "QRATUM|CRS")
    float GetISSResidual() const { return ISSResidual; }

    UFUNCTION(BlueprintPure, Category = "QRATUM|CRS")
    int32 GetStratumCount() const { return Strata.Num(); }

protected:
    UPROPERTY()
    TArray<FWhitneyStratum> Strata;

    int32 ActiveStratumId = 0;
    float ISSResidual     = 0.0f;
    float AccumulatedTime = 0.0f;
};
