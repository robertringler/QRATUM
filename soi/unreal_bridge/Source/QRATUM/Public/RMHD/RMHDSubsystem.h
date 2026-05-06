// RMHDSubsystem.h — URMHDSubsystem (Φ=1 shell).
// Owner: A-07. Outbound: reconnection_rate → RIC + Niagara NS_RMHDFieldLines.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "RMHDSubsystem.generated.h"

UCLASS()
class QRATUM_API URMHDSubsystem : public UTickableWorldSubsystem
{
    GENERATED_BODY()

public:
    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;
    virtual void Tick(float DeltaTime) override;
    virtual TStatId GetStatId() const override;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|RMHD")
    float ReconnectionRate = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|RMHD")
    float PsiFlux = 0.0f;
};
