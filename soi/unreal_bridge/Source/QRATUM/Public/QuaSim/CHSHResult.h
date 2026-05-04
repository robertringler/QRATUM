// CHSHResult.h — FCHSHResult typed wire (QuaSim → RIC).

#pragma once

#include "CoreMinimal.h"
#include "CHSHResult.generated.h"

USTRUCT(BlueprintType)
struct QRATUM_API FCHSHResult
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|QuaSim")
    float S = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|QuaSim")
    float Sigma = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|QuaSim")
    int32 SampleCount = 0;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|QuaSim")
    bool bQuantumBound = false;
};
