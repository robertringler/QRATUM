// QRATUMCommandRouter.h — Translation layer.
//
// Maps an intent name -> a handler function bound by gameplay code.
// Game code (e.g. SoiGame's mission system, vehicle spawner, parameter
// service) registers handlers at startup; the bridge dispatches in Tick.
//
// CRITICAL: handlers run on the game thread. They must be cheap or
// schedule async work themselves.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "Bridge/QRATUMIntent.h"
#include "QRATUMCommandRouter.generated.h"

DECLARE_DYNAMIC_DELEGATE_OneParam(FQRATUMIntentDelegate, const FQRATUMIntent&, Intent);
DECLARE_LOG_CATEGORY_EXTERN(LogQRATUMRouter, Log, All);

class FQRATUMIntentClient;

UCLASS()
class QRATUM_API UQRATUMCommandRouter : public UWorldSubsystem
{
    GENERATED_BODY()

public:
    /** Register a handler for an intent name. Replacing is allowed. */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Intent")
    void RegisterHandler(const FString& IntentName, const FQRATUMIntentDelegate& Handler);

    UFUNCTION(BlueprintCallable, Category = "QRATUM|Intent")
    void UnregisterHandler(const FString& IntentName);

    /** Dispatch one intent to its handler. Returns false if no handler bound. */
    bool Dispatch(const FQRATUMIntent& Intent);

    /** Provide the outbound transport for ack/result frames. Non-owning. */
    void SetIntentClient(FQRATUMIntentClient* InClient) { IntentClient = InClient; }

    /** Diagnostics. */
    UFUNCTION(BlueprintPure, Category = "QRATUM|Intent")
    int32 GetHandlerCount() const { return Handlers.Num(); }

    UFUNCTION(BlueprintPure, Category = "QRATUM|Intent")
    int64 GetDispatchedCount() const { return DispatchedCount; }

    UFUNCTION(BlueprintPure, Category = "QRATUM|Intent")
    int64 GetUnhandledCount() const { return UnhandledCount; }

    UFUNCTION(BlueprintPure, Category = "QRATUM|Intent")
    float GetLastDispatchMs() const { return LastDispatchMs; }

private:
    void EmitFrame(const TCHAR* Type, const FString& Id, bool bOk,
                   const FString& Error, double DispatchMs) const;

    TMap<FString, FQRATUMIntentDelegate> Handlers;
    int64 DispatchedCount = 0;
    int64 UnhandledCount  = 0;
    float LastDispatchMs  = 0.0f;
    FQRATUMIntentClient* IntentClient = nullptr;
};
