// ObserverLoopState.h — FObserverLoopState typed wire (CIIR ↔ OBS).

#pragma once

#include "CoreMinimal.h"
#include "ObserverLoopState.generated.h"

USTRUCT(BlueprintType)
struct QRATUM_API FObserverLoopState
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|CIIR")
    int64 EpochTick = 0;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|CIIR")
    float Phi = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|CIIR")
    float DeltaWNorm = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|CIIR")
    int32 ActiveConstraints = 0;
};
