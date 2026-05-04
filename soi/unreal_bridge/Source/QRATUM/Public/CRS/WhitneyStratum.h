// WhitneyStratum.h — FWhitneyStratum typed wire (CRS → QuBIC, CRS → RMHD).

#pragma once

#include "CoreMinimal.h"
#include "WhitneyStratum.generated.h"

USTRUCT(BlueprintType)
struct QRATUM_API FWhitneyStratum
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|CRS")
    int32 StratumId = 0;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|CRS")
    int32 Codimension = 0;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|CRS")
    TArray<int32> NodeIds;
};
