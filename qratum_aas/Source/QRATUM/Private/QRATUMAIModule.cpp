/// File: Source/QRATUM/Private/QRATUMAIModule.cpp
// Copyright QRATUM Platform. All Rights Reserved.
// Asymmetric Adaptive Search (AAS) Module for Unreal Engine 5
// Unreal Fest Chicago 2026

#include "QRATUMAIModule.h"
#include "Modules/ModuleManager.h"

IMPLEMENT_MODULE(FQRATUMAIModule, QRATUM)

void FQRATUMAIModule::StartupModule()
{
    // Initialize with a fixed default seed for determinism
    // Production games should call SetGlobalSeed with match-specific seed
    GlobalSeed = 0x51415455; // "QATU" in ASCII as hex
    DeterministicTickCounter = 0;
    bIsInitialized = true;

    UE_LOG(LogTemp, Log, TEXT("[QRATUM] AAS Module initialized. Default seed: 0x%016llX"), GlobalSeed);
    UE_LOG(LogTemp, Log, TEXT("[QRATUM] Asymmetric Adaptive Search ready for tactical planning."));
}

void FQRATUMAIModule::ShutdownModule()
{
    bIsInitialized = false;
    UE_LOG(LogTemp, Log, TEXT("[QRATUM] AAS Module shutdown."));
}

FQRATUMAIModule& FQRATUMAIModule::Get()
{
    return FModuleManager::LoadModuleChecked<FQRATUMAIModule>("QRATUM");
}

bool FQRATUMAIModule::IsAvailable()
{
    return FModuleManager::Get().IsModuleLoaded("QRATUM");
}

void FQRATUMAIModule::SetGlobalSeed(uint64 NewSeed)
{
    GlobalSeed = NewSeed;
    // Reset tick counter when seed changes (new session/replay)
    DeterministicTickCounter = 0;
    
    UE_LOG(LogTemp, Log, TEXT("[QRATUM] Global seed set to: 0x%016llX"), GlobalSeed);
}

void FQRATUMAIModule::AdvanceTickCounter()
{
    ++DeterministicTickCounter;
}
