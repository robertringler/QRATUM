// QRATUMIntent.h — frozen intent contract carried over the AI↔UE bridge.
// Format mirrors the Python launcher push protocol:
//   {"type":"intent","name":"spawn_vehicle","params":{...},"id":"<uuid>","epoch":1234}

#pragma once

#include "CoreMinimal.h"
#include "Dom/JsonObject.h"
#include "QRATUMIntent.generated.h"

USTRUCT(BlueprintType)
struct QRATUM_API FQRATUMIntent
{
    GENERATED_BODY()

    /** Intent name (e.g. "spawn_vehicle", "set_parameter"). Maps to a router handler. */
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Intent")
    FString Name;

    /** Optional client-supplied id used to correlate ack/result. */
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Intent")
    FString Id;

    /** Originating epoch tick from the kernel, if known. */
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Intent")
    int64 Epoch = 0;

    /** Raw JSON params payload — kept opaque on game thread; handlers parse. */
    TSharedPtr<FJsonObject> Params;

    /** Original unparsed line, for logging/audit. */
    UPROPERTY(BlueprintReadOnly, Category = "QRATUM|Intent")
    FString RawJson;
};

/** Result returned by a command-router handler. Stays inside UE for now. */
USTRUCT()
struct QRATUM_API FQRATUMIntentResult
{
    GENERATED_BODY()

    bool   bOk        = false;
    FString Id;
    FString Error;
    TSharedPtr<FJsonObject> Data;
};
