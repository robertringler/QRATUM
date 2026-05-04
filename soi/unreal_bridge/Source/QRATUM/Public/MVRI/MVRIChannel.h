// MVRIChannel.h — FMVRIChannel typed wire (MVRI → RIC.DispatchMatrix).

#pragma once

#include "CoreMinimal.h"
#include "MVRIChannel.generated.h"

UENUM(BlueprintType)
enum class EMVRIInjectionStatus : uint8
{
    Accepted        UMETA(DisplayName = "Accepted"),
    RejectedGate    UMETA(DisplayName = "RejectedGate"),
    RejectedInv     UMETA(DisplayName = "RejectedInv"),
    RejectedBound   UMETA(DisplayName = "RejectedBound"),
};

USTRUCT(BlueprintType)
struct QRATUM_API FMVRIChannel
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|MVRI")
    float EMA = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|MVRI")
    float EMAVariance = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|MVRI")
    int32 GateMask = 0;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|MVRI")
    EMVRIInjectionStatus Status = EMVRIInjectionStatus::Accepted;
};
