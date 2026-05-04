// SoiTelemetrySubsystem.cpp
// Copyright QRATUM Platform. All Rights Reserved.

#include "SoiTelemetrySubsystem.h"
#include "TimerManager.h"
#include "Engine/World.h"

void USoiTelemetrySubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);
    
    UE_LOG(LogTemp, Log, TEXT("[SOI] Telemetry Subsystem Initialized"));
    
    // Initialize cached zone heats
    CachedZoneHeats.SetNum(4);
    for (int32 i = 0; i < 4; ++i)
    {
        CachedZoneHeats[i] = 0.0f;
    }
}

void USoiTelemetrySubsystem::Deinitialize()
{
    // Clean up timer
    if (PollTimerHandle.IsValid())
    {
        if (UWorld* World = GetWorld())
        {
            World->GetTimerManager().ClearTimer(PollTimerHandle);
        }
    }
    
    // Shutdown Rust telemetry core
    if (bIsConnected)
    {
        soi_shutdown();
        bIsConnected = false;
    }
    
    UE_LOG(LogTemp, Log, TEXT("[SOI] Telemetry Subsystem Deinitialized"));
    
    Super::Deinitialize();
}

void USoiTelemetrySubsystem::ConnectToAethernet(FString Endpoint)
{
    UE_LOG(LogTemp, Log, TEXT("[SOI] Connecting to Aethernet: %s"), *Endpoint);
    
    // Convert FString to C string
    std::string StdEndpoint(TCHAR_TO_UTF8(*Endpoint));
    const char* CEndpoint = StdEndpoint.c_str();
    
    // Initialize Rust telemetry core
    soi_initialize(CEndpoint);
    bIsConnected = true;
    
    // Start polling timer (60Hz)
    if (UWorld* World = GetWorld())
    {
        World->GetTimerManager().SetTimer(
            PollTimerHandle,
            this,
            &USoiTelemetrySubsystem::PollRustState,
            PollInterval,
            true // Loop
        );
    }
    
    UE_LOG(LogTemp, Log, TEXT("[SOI] Connected and polling at %.2f Hz"), 1.0f / PollInterval);
}

void USoiTelemetrySubsystem::PollRustState()
{
    if (!bIsConnected || !soi_is_initialized())
    {
        return;
    }
    
    // Read epoch
    const int64 NewEpoch = static_cast<int64>(soi_get_epoch());
    
    // Read slashing vector
    const float NewSlashingVector = soi_get_slashing_vector();
    
    // Check for state changes and broadcast
    if (NewEpoch != CachedEpoch || FMath::Abs(NewSlashingVector - CachedSlashingVector) > 0.01f)
    {
        CachedEpoch = NewEpoch;
        CachedSlashingVector = NewSlashingVector;
        
        // Broadcast state update
        OnStateUpdated.Broadcast(CachedEpoch, CachedSlashingVector);
        
        UE_LOG(LogTemp, Verbose, TEXT("[SOI] State Update - Epoch: %lld, Slashing: %.3f"), 
               CachedEpoch, CachedSlashingVector);
    }
    
    // Poll zone heats
    for (int32 ZoneIdx = 0; ZoneIdx < 4; ++ZoneIdx)
    {
        const float NewHeat = soi_get_zone_heat(static_cast<size_t>(ZoneIdx));
        
        if (FMath::Abs(NewHeat - CachedZoneHeats[ZoneIdx]) > 0.01f)
        {
            CachedZoneHeats[ZoneIdx] = NewHeat;
            OnZoneHeatUpdated.Broadcast(ZoneIdx, NewHeat);
            
            UE_LOG(LogTemp, VeryVerbose, TEXT("[SOI] Zone %d Heat: %.3f"), ZoneIdx, NewHeat);
        }
    }
    
    // Poll proof (less frequently - only when epoch changes)
    if (NewEpoch != CachedEpoch)
    {
        char ProofBuffer[256];
        soi_get_proof(ProofBuffer, sizeof(ProofBuffer));
        FString NewProof = UTF8_TO_TCHAR(ProofBuffer);
        
        if (NewProof != CachedProof && !NewProof.IsEmpty())
        {
            CachedProof = NewProof;
            OnProofVerified.Broadcast(CachedProof);
            
            UE_LOG(LogTemp, Verbose, TEXT("[SOI] New Proof: %s"), *CachedProof);
        }
    }
}

int64 USoiTelemetrySubsystem::GetCurrentEpoch() const
{
    if (!bIsConnected)
    {
        return 0;
    }
    return static_cast<int64>(soi_get_epoch());
}

float USoiTelemetrySubsystem::GetZoneHeat(int32 ZoneIndex) const
{
    if (!bIsConnected || ZoneIndex < 0 || ZoneIndex >= 4)
    {
        return 0.0f;
    }
    return soi_get_zone_heat(static_cast<size_t>(ZoneIndex));
}

float USoiTelemetrySubsystem::GetSlashingVector() const
{
    if (!bIsConnected)
    {
        return 0.0f;
    }
    return soi_get_slashing_vector();
}

FString USoiTelemetrySubsystem::GetLatestProof() const
{
    if (!bIsConnected)
    {
        return TEXT("");
    }
    
    char ProofBuffer[256];
    soi_get_proof(ProofBuffer, sizeof(ProofBuffer));
    return FString(UTF8_TO_TCHAR(ProofBuffer));
}

FString USoiTelemetrySubsystem::GetStateJSON() const
{
    if (!bIsConnected)
    {
        return TEXT("{}");
    }
    
    char JsonBuffer[4096];
    const int32 BytesWritten = soi_get_status_json(JsonBuffer, sizeof(JsonBuffer));
    
    if (BytesWritten > 0)
    {
        return FString(UTF8_TO_TCHAR(JsonBuffer));
    }
    
    return TEXT("{}");
}

bool USoiTelemetrySubsystem::IsConnected() const
{
    return bIsConnected && soi_is_initialized();
}
