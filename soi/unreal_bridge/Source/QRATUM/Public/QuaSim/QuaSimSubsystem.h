// QuaSimSubsystem.h — UQuaSimSubsystem (Φ=1 shell).
// Owner: A-05. Outbound: FCHSHResult → RIC.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "QuaSim/CHSHResult.h"
#include "QuaSimSubsystem.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnCHSHEvaluated, const FCHSHResult&, Result);

UCLASS()
class QRATUM_API UQuaSimSubsystem : public UTickableWorldSubsystem
{
    GENERATED_BODY()

public:
    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;
    virtual void Tick(float DeltaTime) override;
    virtual TStatId GetStatId() const override;

    UPROPERTY(BlueprintAssignable, Category = "QRATUM|QuaSim")
    FOnCHSHEvaluated OnCHSHEvaluated;

    UFUNCTION(BlueprintPure, Category = "QRATUM|QuaSim")
    const FCHSHResult& GetLastResult() const { return LastResult; }

protected:
    UPROPERTY()
    FCHSHResult LastResult;

    int64 TickCounter = 0;
};
