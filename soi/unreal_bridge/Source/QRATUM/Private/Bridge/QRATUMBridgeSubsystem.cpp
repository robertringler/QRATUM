// QRATUMBridgeSubsystem.cpp — JSON snapshot writer (atomic temp+rename).

#include "Bridge/QRATUMBridgeSubsystem.h"
#include "Bridge/QRATUMIntentClient.h"
#include "Bridge/QRATUMCommandRouter.h"
#include "HAL/PlatformTime.h"
#include "CIIR/CIIRSubsystem.h"
#include "CRS/CRSSubsystem.h"
#include "RIC/RICSubsystem.h"
#include "QuaSim/QuaSimSubsystem.h"
#include "QuBIC/QuBICSubsystem.h"
#include "RMHD/RMHDSubsystem.h"
#include "MVRI/MVRISubsystem.h"
#include "RWB/RWBSubsystem.h"
#include "CGL/CGLSubsystem.h"

#include "Engine/World.h"
#include "HAL/PlatformFileManager.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformMisc.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

namespace
{
    static FString MVRIStatusToString(EMVRIInjectionStatus S)
    {
        switch (S)
        {
            case EMVRIInjectionStatus::Accepted:      return TEXT("Accepted");
            case EMVRIInjectionStatus::RejectedGate:  return TEXT("RejectedGate");
            case EMVRIInjectionStatus::RejectedInv:   return TEXT("RejectedInv");
            case EMVRIInjectionStatus::RejectedBound: return TEXT("RejectedBound");
        }
        return TEXT("Unknown");
    }
}

FString UQRATUMBridgeSubsystem::ResolveBridgeDirectory()
{
#if PLATFORM_WINDOWS
    FString Base = FPlatformMisc::GetEnvironmentVariable(TEXT("LOCALAPPDATA"));
    if (Base.IsEmpty())
    {
        Base = FPaths::ProjectSavedDir();
    }
#else
    FString Base = FPlatformMisc::GetEnvironmentVariable(TEXT("HOME"));
    if (Base.IsEmpty()) Base = FPaths::ProjectSavedDir();
    Base = FPaths::Combine(Base, TEXT(".local/share"));
#endif
    return FPaths::Combine(Base, TEXT("QRATUM"), TEXT("bridge"));
}

void UQRATUMBridgeSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);

    const FString Dir = ResolveBridgeDirectory();
    IPlatformFile& PF = FPlatformFileManager::Get().GetPlatformFile();
    if (!PF.DirectoryExists(*Dir))
    {
        PF.CreateDirectoryTree(*Dir);
    }
    BridgePath = FPaths::Combine(Dir, TEXT("state.json"));
    LastState.Phase = TEXT("INIT");

    UE_LOG(LogTemp, Log, TEXT("[QRATUM/Bridge] Publishing to %s"), *BridgePath);

    // Spin up the launcher subscription on a worker thread.
    // Host/port can be overridden via env vars QRATUM_LAUNCHER_HOST/PORT.
    FString Host = FPlatformMisc::GetEnvironmentVariable(TEXT("QRATUM_LAUNCHER_HOST"));
    if (Host.IsEmpty()) Host = TEXT("127.0.0.1");
    int32 Port = 5555;
    const FString PortStr = FPlatformMisc::GetEnvironmentVariable(TEXT("QRATUM_LAUNCHER_PORT"));
    if (!PortStr.IsEmpty()) { Port = FCString::Atoi(*PortStr); }
    IntentClient = MakeUnique<FQRATUMIntentClient>(Host, Port);
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/Bridge] intent client subscribing to %s:%d"), *Host, Port);
}

void UQRATUMBridgeSubsystem::Deinitialize()
{
    LastState.Phase = TEXT("HALT");
    SnapshotAndPublish();
    if (IntentClient.IsValid())
    {
        IntentClient->Stop();
        IntentClient.Reset();
    }
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/Bridge] Deinitialize"));
    Super::Deinitialize();
}

void UQRATUMBridgeSubsystem::Tick(float /*DeltaTime*/)
{
    // Always drain pending intents (cheap when idle).
    DrainIntents();

    if (++TickCounter < TicksPerWrite) return;
    TickCounter = 0;
    LastState.Phase = TEXT("RUN");
    SnapshotAndPublish();
}

bool UQRATUMBridgeSubsystem::IsLauncherConnected() const
{
    return IntentClient.IsValid() && IntentClient->IsConnected();
}

int64 UQRATUMBridgeSubsystem::GetIntentsReceived() const
{
    return IntentClient.IsValid() ? IntentClient->GetReceivedCount() : 0;
}

void UQRATUMBridgeSubsystem::DrainIntents()
{
    if (!IntentClient.IsValid()) return;
    UWorld* World = GetWorld();
    UQRATUMCommandRouter* Router = World ? World->GetSubsystem<UQRATUMCommandRouter>() : nullptr;

    // Ensure the router can emit ack/result frames back over the same socket.
    if (Router)
    {
        Router->SetIntentClient(IntentClient.Get());
    }

    FQRATUMIntent Intent;
    int32 Drained = 0;
    const double T0 = FPlatformTime::Seconds();
    constexpr double DispatchBudgetMs = 2.0; // §8

    while (Drained < MaxIntentsPerTick && IntentClient->PopIntent(Intent))
    {
        ++Drained;
        if (Router)
        {
            Router->Dispatch(Intent);
        }
        else
        {
            UE_LOG(LogTemp, Verbose,
                   TEXT("[QRATUM/Bridge] router unavailable, intent '%s' dropped"),
                   *Intent.Name);
        }
        const double ElapsedMs = (FPlatformTime::Seconds() - T0) * 1000.0;
        if (ElapsedMs >= DispatchBudgetMs)
        {
            // Budget exceeded — defer remaining intents to next tick.
            break;
        }
    }
}

TStatId UQRATUMBridgeSubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(UQRATUMBridgeSubsystem, STATGROUP_Tickables);
}

void UQRATUMBridgeSubsystem::SnapshotAndPublish()
{
    UWorld* World = GetWorld();
    if (!World) return;

    if (UCIIRSubsystem* CIIR = World->GetSubsystem<UCIIRSubsystem>())
    {
        const FObserverLoopState& S = CIIR->GetLastState();
        LastState.EpochTick       = S.EpochTick;
        LastState.CIIRPhi         = S.Phi;
        LastState.CIIRDeltaWNorm  = S.DeltaWNorm;
        LastState.CIIRConstraints = S.ActiveConstraints;
    }
    if (UCRSSubsystem* CRS = World->GetSubsystem<UCRSSubsystem>())
    {
        LastState.CRSActiveStratum = CRS->GetActiveStratumId();
        LastState.CRSISSResidual   = CRS->GetISSResidual();
        LastState.CRSStratumCount  = CRS->GetStratumCount();
    }
    if (URICSubsystem* RIC = World->GetSubsystem<URICSubsystem>())
    {
        const FDispatchMatrix& M = RIC->GetDispatchMatrix();
        LastState.RICDispatchWeights.Reset();
        for (const FDispatchMatrixCell& C : M.Cells)
        {
            LastState.RICDispatchWeights.Add(C.Weight);
        }
    }
    if (UQuaSimSubsystem* Q = World->GetSubsystem<UQuaSimSubsystem>())
    {
        const FCHSHResult& R = Q->GetLastResult();
        LastState.QuaSimS            = R.S;
        LastState.QuaSimQuantumBound = R.bQuantumBound;
        LastState.QuaSimSamples      = R.SampleCount;
    }
    if (UQuBICSubsystem* Q = World->GetSubsystem<UQuBICSubsystem>())
    {
        LastState.QuBICEpoch = Q->GetEpoch();
        LastState.QuBICNodes = Q->GetGraph().Nodes.Num();
    }
    if (URMHDSubsystem* R = World->GetSubsystem<URMHDSubsystem>())
    {
        LastState.RMHDReconnectRate = R->ReconnectionRate;
        LastState.RMHDPsiFlux       = R->PsiFlux;
    }
    if (UMVRISubsystem* M = World->GetSubsystem<UMVRISubsystem>())
    {
        const FMVRIChannel& C = M->GetLastChannel();
        LastState.MVRIema      = C.EMA;
        LastState.MVRIGateMask = C.GateMask;
        LastState.MVRIStatus   = MVRIStatusToString(C.Status);
    }
    if (URWBSubsystem* R = World->GetSubsystem<URWBSubsystem>())
    {
        LastState.RWBSequence = R->GetSequence();
    }
    if (UCGLSubsystem* C = World->GetSubsystem<UCGLSubsystem>())
    {
        LastState.CGLLambdaMin     = C->GramianEigenvalueMin;
        LastState.CGLRankDeficient = C->bRankDeficient;
    }
    LastState.WallClockSeconds = FPlatformTime::Seconds();

    // Serialize.
    FString Out;
    TSharedRef<TJsonWriter<>> W = TJsonWriterFactory<>::Create(&Out);
    W->WriteObjectStart();
    W->WriteValue(TEXT("phase"),             LastState.Phase);
    W->WriteValue(TEXT("epoch_tick"),        LastState.EpochTick);
    W->WriteValue(TEXT("wall_clock_seconds"), LastState.WallClockSeconds);
    W->WriteObjectStart(TEXT("ciir"));
        W->WriteValue(TEXT("phi"), LastState.CIIRPhi);
        W->WriteValue(TEXT("delta_w_norm"), LastState.CIIRDeltaWNorm);
        W->WriteValue(TEXT("active_constraints"), LastState.CIIRConstraints);
    W->WriteObjectEnd();
    W->WriteObjectStart(TEXT("crs"));
        W->WriteValue(TEXT("active_stratum"), LastState.CRSActiveStratum);
        W->WriteValue(TEXT("iss_residual"), LastState.CRSISSResidual);
        W->WriteValue(TEXT("stratum_count"), LastState.CRSStratumCount);
    W->WriteObjectEnd();
    W->WriteArrayStart(TEXT("ric_dispatch_weights"));
        for (float V : LastState.RICDispatchWeights) W->WriteValue(V);
    W->WriteArrayEnd();
    W->WriteObjectStart(TEXT("quasim"));
        W->WriteValue(TEXT("s"), LastState.QuaSimS);
        W->WriteValue(TEXT("quantum_bound"), LastState.QuaSimQuantumBound);
        W->WriteValue(TEXT("samples"), LastState.QuaSimSamples);
    W->WriteObjectEnd();
    W->WriteObjectStart(TEXT("qubic"));
        W->WriteValue(TEXT("epoch"), LastState.QuBICEpoch);
        W->WriteValue(TEXT("nodes"), LastState.QuBICNodes);
    W->WriteObjectEnd();
    W->WriteObjectStart(TEXT("rmhd"));
        W->WriteValue(TEXT("reconnection_rate"), LastState.RMHDReconnectRate);
        W->WriteValue(TEXT("psi_flux"), LastState.RMHDPsiFlux);
    W->WriteObjectEnd();
    W->WriteObjectStart(TEXT("mvri"));
        W->WriteValue(TEXT("ema"), LastState.MVRIema);
        W->WriteValue(TEXT("gate_mask"), LastState.MVRIGateMask);
        W->WriteValue(TEXT("status"), LastState.MVRIStatus);
    W->WriteObjectEnd();
    W->WriteObjectStart(TEXT("rwb"));
        W->WriteValue(TEXT("sequence"), LastState.RWBSequence);
    W->WriteObjectEnd();
    W->WriteObjectStart(TEXT("cgl"));
        W->WriteValue(TEXT("lambda_min"), LastState.CGLLambdaMin);
        W->WriteValue(TEXT("rank_deficient"), LastState.CGLRankDeficient);
    W->WriteObjectEnd();
    W->WriteObjectEnd();
    W->Close();

    // Atomic write: temp file + rename.
    const FString TmpPath = BridgePath + TEXT(".tmp");
    if (FFileHelper::SaveStringToFile(Out, *TmpPath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM))
    {
        IPlatformFile& PF = FPlatformFileManager::Get().GetPlatformFile();
        PF.DeleteFile(*BridgePath);
        PF.MoveFile(*BridgePath, *TmpPath);
    }
}
