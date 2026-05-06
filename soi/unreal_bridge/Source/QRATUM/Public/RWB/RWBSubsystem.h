// RWBSubsystem.h — URWBSubsystem (Φ=1 shell).
// Owner: A-08. Outbound: FRWBPacket → MVRI.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "RWB/RWBPacket.h"
#include "RWBSubsystem.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnRWBPacketEmitted, const FRWBPacket&, Packet);

UCLASS()
class QRATUM_API URWBSubsystem : public UTickableWorldSubsystem
{
    GENERATED_BODY()

public:
    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;
    virtual void Tick(float DeltaTime) override;
    virtual TStatId GetStatId() const override;

    UPROPERTY(BlueprintAssignable, Category = "QRATUM|RWB")
    FOnRWBPacketEmitted OnPacketEmitted;

    UFUNCTION(BlueprintPure, Category = "QRATUM|RWB")
    int64 GetSequence() const { return Sequence; }

protected:
    int64 Sequence       = 0;
    float AccumulatedTime = 0.0f;
};
