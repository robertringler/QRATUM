// GenomicGraphState.h — FGenomicNode + FGenomicGraphState typed wire (QuBIC → RIC).

#pragma once

#include "CoreMinimal.h"
#include "GenomicGraphState.generated.h"

USTRUCT(BlueprintType)
struct QRATUM_API FGenomicNode
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|QuBIC")
    int64 NodeId = 0;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|QuBIC")
    int32 StratumId = 0;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|QuBIC")
    float Weight = 0.0f;
};

USTRUCT(BlueprintType)
struct QRATUM_API FGenomicGraphState
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|QuBIC")
    TArray<FGenomicNode> Nodes;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|QuBIC")
    TArray<int64> EdgesFrom;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|QuBIC")
    TArray<int64> EdgesTo;
};
