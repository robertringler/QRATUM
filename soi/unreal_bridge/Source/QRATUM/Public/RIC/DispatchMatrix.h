// DispatchMatrix.h — FDispatchMatrix (RIC inbound: MVRI/QuaSim/QuBIC/RMHD × regime weights).

#pragma once

#include "CoreMinimal.h"
#include "DispatchMatrix.generated.h"

UENUM(BlueprintType)
enum class EQRATUMDispatchSource : uint8
{
    MVRI    UMETA(DisplayName = "MVRI"),
    QuaSim  UMETA(DisplayName = "QuaSim"),
    QuBIC   UMETA(DisplayName = "QuBIC"),
    RMHD    UMETA(DisplayName = "RMHD"),
};

USTRUCT(BlueprintType)
struct QRATUM_API FDispatchMatrixCell
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|RIC")
    EQRATUMDispatchSource Source = EQRATUMDispatchSource::MVRI;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|RIC")
    float Weight = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|RIC")
    int64 LastWriteTick = 0;
};

USTRUCT(BlueprintType)
struct QRATUM_API FDispatchMatrix
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|RIC")
    TArray<FDispatchMatrixCell> Cells;
};
