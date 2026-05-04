#include "QRATUMMemoryComponent.h"

UQRATUMMemoryComponent::UQRATUMMemoryComponent()
{
	PrimaryComponentTick.bCanEverTick = false; // timer-driven, not tick-driven
}

void UQRATUMMemoryComponent::BeginPlay()
{
	Super::BeginPlay();

	if (!GetWorld()) return;

	FTimerDelegate TimerDel;
	TimerDel.BindLambda([this]()
	{
		this->DecayMemory(MemoryTickInterval);
		this->PruneMemory();
	});

	GetWorld()->GetTimerManager().SetTimer(
		MemoryTimerHandle,
		TimerDel,
		MemoryTickInterval,
		true
	);
}

void UQRATUMMemoryComponent::AddMemory(const FMemoryEntry& Entry)
{
	// Dedup: if ActorID + EventType already exists, merge salience and update location/timestamp
	for (FMemoryEntry& Existing : MemoryPool)
	{
		if (Existing.ActorID == Entry.ActorID &&
			Existing.EventType == Entry.EventType)
		{
			Existing.Salience  = FMath::Max(Existing.Salience, Entry.Salience);
			Existing.Location  = Entry.Location;
			Existing.Timestamp = Entry.Timestamp;
			if (Entry.ActorTag != NAME_None)
			{
				Existing.ActorTag = Entry.ActorTag;
			}
			OnMemoryUpdated.Broadcast();
			return;
		}
	}

	MemoryPool.Add(Entry);
	OnMemoryUpdated.Broadcast();
}

TArray<FMemoryEntry> UQRATUMMemoryComponent::GetMemoryOf(int32 ActorID) const
{
	TArray<FMemoryEntry> Results;

	for (const FMemoryEntry& Entry : MemoryPool)
	{
		if (Entry.ActorID == ActorID && Entry.Salience > PruneThreshold)
		{
			Results.Add(Entry);
		}
	}

	return Results;
}

TArray<FMemoryEntry> UQRATUMMemoryComponent::GetAllMemories() const
{
	return MemoryPool;
}

void UQRATUMMemoryComponent::DecayMemory(float DeltaTime)
{
	for (FMemoryEntry& Entry : MemoryPool)
	{
		Entry.Salience -= DecayRate * DeltaTime;
		Entry.Salience  = FMath::Clamp(Entry.Salience, 0.0f, 1.0f);
	}
}

void UQRATUMMemoryComponent::PruneMemory()
{
	for (int32 i = MemoryPool.Num() - 1; i >= 0; --i)
	{
		if (MemoryPool[i].Salience <= PruneThreshold)
		{
			MemoryPool.RemoveAt(i);
		}
	}

	OnMemoryUpdated.Broadcast();
}
