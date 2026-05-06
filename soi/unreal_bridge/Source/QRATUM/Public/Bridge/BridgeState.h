// BridgeState.h — payload schema mirrored to qratum_desktop via JSON IPC.

#pragma once

#include "CoreMinimal.h"
#include "BridgeState.generated.h"

USTRUCT(BlueprintType)
struct QRATUM_API FQRATUMBridgeState
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") int64  EpochTick = 0;
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") double WallClockSeconds = 0.0;
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") FString Phase;            // "POST"|"INIT"|"RUN"|"HALT"

    // CIIR
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") float CIIRPhi          = 0.0f;
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") float CIIRDeltaWNorm   = 0.0f;
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") int32 CIIRConstraints  = 0;

    // CRS
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") int32 CRSActiveStratum = 0;
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") float CRSISSResidual   = 0.0f;
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") int32 CRSStratumCount  = 0;

    // RIC
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") TArray<float> RICDispatchWeights;

    // QuaSim
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") float QuaSimS               = 0.0f;
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") bool  QuaSimQuantumBound    = false;
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") int32 QuaSimSamples         = 0;

    // QuBIC
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") int64 QuBICEpoch        = 0;
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") int32 QuBICNodes        = 0;

    // RMHD
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") float RMHDReconnectRate = 0.0f;
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") float RMHDPsiFlux       = 0.0f;

    // MVRI / RWB
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") float MVRIema           = 0.0f;
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") int32 MVRIGateMask      = 0;
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") FString MVRIStatus;
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") int64 RWBSequence       = 0;

    // CGL
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") float CGLLambdaMin      = 0.0f;
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Bridge") bool  CGLRankDeficient  = false;
};
