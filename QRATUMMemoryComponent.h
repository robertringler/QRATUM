#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "MemoryStructs.h"
#include "QRATUMMemoryComponent.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnMemoryUpdated);

UCLASS(ClassGroup=(QRATUM), meta=(BlueprintSpawnableComponent))
class UQRATUMMemoryComponent : public UActorComponent
{
	GENERATED_BODY()

public:

	UQRATUMMemoryComponent();

	// Fires whenever memory pool changes (add, decay, prune)
	UPROPERTY(BlueprintAssignable, Category="QRATUM|Memory")
	FOnMemoryUpdated OnMemoryUpdated;

	// Salience reduction per MemoryTickInterval (0.0-1.0 range)
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="QRATUM|Memory")
	float DecayRate = 0.1f;

	// Entries at or below this salience are pruned
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="QRATUM|Memory")
	float PruneThreshold = 0.05f;

	// How often (seconds) the decay+prune cycle runs. Default 1s.
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="QRATUM|Memory")
	float MemoryTickInterval = 1.0f;

	// Add or merge a memory entry
	UFUNCTION(BlueprintCallable, Category="QRATUM|Memory")
	void AddMemory(const FMemoryEntry& Entry);

	// Returns all entries for a specific ActorID above PruneThreshold
	UFUNCTION(BlueprintCallable, Category="QRATUM|Memory")
	TArray<FMemoryEntry> GetMemoryOf(int32 ActorID) const;

	// Returns full memory pool (used by debug overlay)
	UFUNCTION(BlueprintCallable, Category="QRATUM|Memory")
	TArray<FMemoryEntry> GetAllMemories() const;

	// Force an immediate prune pass
	UFUNCTION(BlueprintCallable, Category="QRATUM|Memory")
	void PruneMemory();

protected:

	virtual void BeginPlay() override;

private:

	TArray<FMemoryEntry> MemoryPool;
	FTimerHandle MemoryTimerHandle;

	void DecayMemory(float DeltaTime);
};
