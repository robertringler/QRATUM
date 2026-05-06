// QRATUMCommandRouter.cpp — §6 + §7: namespaced dispatch with handler isolation.

#include "Bridge/QRATUMCommandRouter.h"
#include "Bridge/QRATUMIntentClient.h"
#include "HAL/PlatformTime.h"

DEFINE_LOG_CATEGORY(LogQRATUMRouter);

void UQRATUMCommandRouter::RegisterHandler(const FString& IntentName, const FQRATUMIntentDelegate& Handler)
{
    if (IntentName.IsEmpty() || !Handler.IsBound()) return;
    Handlers.Add(IntentName, Handler);
    UE_LOG(LogQRATUMRouter, Log, TEXT("registered handler '%s' (total=%d)"),
           *IntentName, Handlers.Num());
}

void UQRATUMCommandRouter::UnregisterHandler(const FString& IntentName)
{
    Handlers.Remove(IntentName);
}

void UQRATUMCommandRouter::EmitFrame(const TCHAR* Type, const FString& Id,
                                     bool bOk, const FString& Error,
                                     double DispatchMs) const
{
    if (!IntentClient || Id.IsEmpty()) return;
    FString EscErr = Error.Replace(TEXT("\""), TEXT("\\\""));
    FString Line = FString::Printf(
        TEXT("{\"v\":\"1.0\",\"type\":\"%s\",\"id\":\"%s\",\"ok\":%s,\"error\":\"%s\",\"dispatch_ms\":%.3f}"),
        Type, *Id, bOk ? TEXT("true") : TEXT("false"), *EscErr, DispatchMs);
    IntentClient->SendLine(Line);
}

bool UQRATUMCommandRouter::Dispatch(const FQRATUMIntent& Intent)
{
    const double T0 = FPlatformTime::Seconds();
    FQRATUMIntentDelegate* H = Handlers.Find(Intent.Name);
    if (!H)
    {
        ++UnhandledCount;
        UE_LOG(LogQRATUMRouter, Verbose,
               TEXT("unhandled intent '%s' (id=%s)"), *Intent.Name, *Intent.Id);
        EmitFrame(TEXT("result"), Intent.Id, /*bOk*/false, TEXT("E_UNKNOWN"), 0.0);
        return false;
    }

    // §3 — emit ack the moment we accept the intent for dispatch.
    EmitFrame(TEXT("ack"), Intent.Id, /*bOk*/true, FString(), 0.0);

    // §6 — handler isolation.
    bool    bOk = true;
    FString ErrorStr;
#if PLATFORM_EXCEPTIONS_DISABLED
    H->ExecuteIfBound(Intent);
#else
    try
    {
        H->ExecuteIfBound(Intent);
    }
    catch (const std::exception& Ex)
    {
        bOk = false;
        ErrorStr = FString::Printf(TEXT("handler_exception: %hs"), Ex.what());
        UE_LOG(LogQRATUMRouter, Error, TEXT("handler '%s' raised: %hs"),
               *Intent.Name, Ex.what());
    }
    catch (...)
    {
        bOk = false;
        ErrorStr = TEXT("handler_exception");
        UE_LOG(LogQRATUMRouter, Error,
               TEXT("handler '%s' raised unknown exception"), *Intent.Name);
    }
#endif

    const double DtMs = (FPlatformTime::Seconds() - T0) * 1000.0;
    LastDispatchMs = static_cast<float>(DtMs);
    ++DispatchedCount;
    EmitFrame(TEXT("result"), Intent.Id, bOk, ErrorStr, DtMs);
    return bOk;
}
