// SoiTelemetrySubsystem.h
// Copyright QRATUM Platform. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "SoiTelemetrySubsystem.generated.h"

// Import Rust Functions via FFI
extern "C" {
    void soi_initialize(const char* endpoint);
    uint64 soi_get_epoch();
    float soi_get_zone_heat(size_t zone_idx);
    float soi_get_slashing_vector();
    void soi_get_proof(char* buffer, size_t length);
    int32 soi_get_status_json(char* buffer, size_t length);
    bool soi_is_initialized();
    void soi_shutdown();
}

/**
 * State update delegate - broadcasts when telemetry state changes
 * @param Epoch - Current blockchain epoch
 * @param SlashingVector - Current slashing risk metric (0.0-1.0)
 */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnStateUpdate, int64, Epoch, float, SlashingVector);

/**
 * Zone heat update delegate - broadcasts zone-specific validator activity
 * @param ZoneIndex - Zone identifier (0=Z0, 1=Z1, 2=Z2, 3=Z3)
 * @param HeatValue - Heat metric for the zone (0.0-1.0)
 */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnZoneHeatUpdate, int32, ZoneIndex, float, HeatValue);

/**
 * Proof verification delegate - broadcasts when new ZK proof is verified
 * @param ProofHash - Hash of the verified proof
 */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnProofVerified, FString, ProofHash);

/**
 * SOI Telemetry Subsystem
 * 
 * High-performance bridge between Rust telemetry core and Unreal Engine.
 * Provides real-time access to QRADLE state, Aethernet consensus,
 * and ZK proof streams for cinematic visualization.
 * 
 * Architecture:
 * - Rust Core: Async WebSocket handler, zero-copy deserialization
 * - This Bridge: FFI interface, polling timer, Blueprint integration
 * - Unreal UI: Niagara particles, CommonUI, Materials
 */
UCLASS()
class SOIGAME_API USoiTelemetrySubsystem : public UGameInstanceSubsystem
{
    GENERATED_BODY()

public:
    // Subsystem lifecycle
    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;

    /**
     * Connect to the Aethernet telemetry endpoint
     * @param Endpoint - WebSocket URL (e.g., "ws://localhost:8000/soi/telemetry")
     */
    UFUNCTION(BlueprintCallable, Category = "SOI|Telemetry")
    void ConnectToAethernet(FString Endpoint);

    /**
     * Get the current blockchain epoch
     * @return Current epoch number
     */
    UFUNCTION(BlueprintPure, Category = "SOI|Telemetry")
    int64 GetCurrentEpoch() const;

    /**
     * Get validator heat for a specific zone
     * @param ZoneIndex - Zone to query (0=Z0, 1=Z1, 2=Z2, 3=Z3)
     * @return Heat value (0.0 = idle, 1.0 = maximum activity)
     */
    UFUNCTION(BlueprintPure, Category = "SOI|Telemetry")
    float GetZoneHeat(int32 ZoneIndex) const;

    /**
     * Get the current slashing vector (network risk metric)
     * @return Slashing risk (0.0 = safe, 1.0 = critical)
     */
    UFUNCTION(BlueprintPure, Category = "SOI|Telemetry")
    float GetSlashingVector() const;

    /**
     * Get the latest ZK proof hash
     * @return Proof hash as hex string
     */
    UFUNCTION(BlueprintPure, Category = "SOI|Telemetry")
    FString GetLatestProof() const;

    /**
     * Get full telemetry state as JSON
     * @return JSON-formatted state
     */
    UFUNCTION(BlueprintCallable, Category = "SOI|Telemetry")
    FString GetStateJSON() const;

    /**
     * Check if telemetry system is connected and initialized
     * @return True if connected
     */
    UFUNCTION(BlueprintPure, Category = "SOI|Telemetry")
    bool IsConnected() const;

    // Event delegates for Blueprint
    UPROPERTY(BlueprintAssignable, Category = "SOI|Telemetry|Events")
    FOnStateUpdate OnStateUpdated;

    UPROPERTY(BlueprintAssignable, Category = "SOI|Telemetry|Events")
    FOnZoneHeatUpdate OnZoneHeatUpdated;

    UPROPERTY(BlueprintAssignable, Category = "SOI|Telemetry|Events")
    FOnProofVerified OnProofVerified;

private:
    /**
     * Timer callback to poll Rust state at 60Hz
     * This runs on the game thread and reads from Rust's lock-free state
     */
    void PollRustState();

    // Timer handle for polling
    FTimerHandle PollTimerHandle;

    // Cached state for change detection
    int64 CachedEpoch = 0;
    float CachedSlashingVector = 0.0f;
    TArray<float> CachedZoneHeats;
    FString CachedProof;

    // Connection state
    bool bIsConnected = false;

    // Poll rate (60 times per second)
    const float PollInterval = 1.0f / 60.0f;
};
