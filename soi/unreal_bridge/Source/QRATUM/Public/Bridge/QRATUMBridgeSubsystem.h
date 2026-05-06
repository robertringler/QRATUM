// QRATUMBridgeSubsystem.h — File-based JSON IPC publisher to qratum_desktop.
// Owner: A-01. Writes %LOCALAPPDATA%/QRATUM/bridge/state.json atomically every N ticks.
// Also hosts the AI-intent ingress: connects to the launcher socket on a worker
// thread and dispatches received intents through UQRATUMCommandRouter on the
// game thread.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "Bridge/BridgeState.h"
#include "QRATUMBridgeSubsystem.generated.h"

class FQRATUMIntentClient;

UCLASS()
class QRATUM_API UQRATUMBridgeSubsystem : public UTickableWorldSubsystem
{
    GENERATED_BODY()

public:
    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;
    virtual void Tick(float DeltaTime) override;
    virtual TStatId GetStatId() const override;

    UFUNCTION(BlueprintPure, Category = "QRATUM|Bridge")
    const FQRATUMBridgeState& GetLastState() const { return LastState; }

    UFUNCTION(BlueprintPure, Category = "QRATUM|Bridge")
    FString GetBridgePath() const { return BridgePath; }

    UFUNCTION(BlueprintPure, Category = "QRATUM|Bridge")
    bool IsLauncherConnected() const;

    UFUNCTION(BlueprintPure, Category = "QRATUM|Bridge")
    int64 GetIntentsReceived() const;

protected:
    void SnapshotAndPublish();
    void DrainIntents();
    static FString ResolveBridgeDirectory();

    UPROPERTY()
    FQRATUMBridgeState LastState;

    FString BridgePath;
    int32   TickCounter   = 0;
    int32   TicksPerWrite = 6; // ~0.1s at 60Hz

    /** Owned worker thread that subscribes to the launcher and feeds intents. */
    TUniquePtr<FQRATUMIntentClient> IntentClient;

    /** Cap how many intents we drain per tick to bound game-thread cost. */
    int32 MaxIntentsPerTick = 32;
};

