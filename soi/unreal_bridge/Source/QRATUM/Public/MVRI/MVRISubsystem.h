// MVRISubsystem.h — UMVRISubsystem (Φ=1 shell).
// Owner: A-08. Inbound: FRWBPacket. Outbound: FMVRIChannel → RIC.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "MVRI/MVRIChannel.h"
#include "RWB/RWBPacket.h"
#include "MVRISubsystem.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnMVRIChannelUpdated, const FMVRIChannel&, Channel);

UCLASS()
class QRATUM_API UMVRISubsystem : public UTickableWorldSubsystem
{
    GENERATED_BODY()

public:
    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;
    virtual void Tick(float DeltaTime) override;
    virtual TStatId GetStatId() const override;

    UFUNCTION(BlueprintCallable, Category = "QRATUM|MVRI")
    EMVRIInjectionStatus Ingest(const FRWBPacket& Packet);

    UPROPERTY(BlueprintAssignable, Category = "QRATUM|MVRI")
    FOnMVRIChannelUpdated OnChannelUpdated;

    UFUNCTION(BlueprintPure, Category = "QRATUM|MVRI")
    const FMVRIChannel& GetLastChannel() const { return LastChannel; }

protected:
    UPROPERTY()
    FMVRIChannel LastChannel;
};
