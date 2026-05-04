// RWBPacket.h — FRWBPacket typed wire (RWB → MVRI).
// Φ=1 shell: fields declared, no behavior.

#pragma once

#include "CoreMinimal.h"
#include "RWBPacket.generated.h"

USTRUCT(BlueprintType)
struct QRATUM_API FRWBPacket
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|RWB")
    int64 Sequence = 0;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|RWB")
    double TimestampSeconds = 0.0;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|RWB")
    TArray<float> SensorVector;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|RWB")
    FString CanonicalHashHex;
};
