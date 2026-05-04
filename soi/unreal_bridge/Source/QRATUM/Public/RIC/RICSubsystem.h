// RICSubsystem.h — URICSubsystem (Φ=1 shell).
// Owner: A-04. 4-cell DispatchMatrix; outbound to GUI P-03 ViewModel.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "RIC/DispatchMatrix.h"
#include "RICSubsystem.generated.h"

UCLASS()
class QRATUM_API URICSubsystem : public UTickableWorldSubsystem
{
    GENERATED_BODY()

public:
    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;
    virtual void Tick(float DeltaTime) override;
    virtual TStatId GetStatId() const override;

    UFUNCTION(BlueprintPure, Category = "QRATUM|RIC")
    const FDispatchMatrix& GetDispatchMatrix() const { return DispatchMatrix; }

protected:
    UPROPERTY()
    FDispatchMatrix DispatchMatrix;
};
