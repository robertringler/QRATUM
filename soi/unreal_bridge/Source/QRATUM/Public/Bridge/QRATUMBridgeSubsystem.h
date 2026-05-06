// QRATUMBridgeSubsystem.h — File-based JSON IPC publisher to qratum_desktop.
// Owner: A-01. Writes %LOCALAPPDATA%/QRATUM/bridge/state.json atomically every N ticks.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "Bridge/BridgeState.h"
#include "QRATUMBridgeSubsystem.generated.h"

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

protected:
    void SnapshotAndPublish();
    static FString ResolveBridgeDirectory();

    UPROPERTY()
    FQRATUMBridgeState LastState;

    FString BridgePath;
    int32   TickCounter   = 0;
    int32   TicksPerWrite = 6; // ~0.1s at 60Hz
};
