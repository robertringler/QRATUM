// QRATUM.Build.cs
// Copyright QRATUM Platform. All Rights Reserved.
// Asymmetric Adaptive Search (AAS) Module for Unreal Engine 5
// Unreal Fest Chicago 2026

using UnrealBuildTool;

public class QRATUM : ModuleRules
{
    public QRATUM(ReadOnlyTargetRules Target) : base(Target)
    {
        // Use shared PCH for faster compilation
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        
        // Enable C++20 for modern features
        CppStandard = CppStandardVersion.Cpp20;
        
        // Module configuration
        bEnableExceptions = false;
        bUseRTTI = false;

        // Public dependencies - modules used in public headers
        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "AIModule",
            "GameplayTasks",
            "NavigationSystem"
        });

        // Private dependencies - modules used only in implementation
        PrivateDependencyModuleNames.AddRange(new string[]
        {
            "InputCore",
            "GameplayAbilities"
        });

        // Define module API macro
        PublicDefinitions.Add("QRATUM_API=DLLEXPORT");
        
        // Include paths for core AAS (engine-agnostic)
        PublicIncludePaths.AddRange(new string[]
        {
            System.IO.Path.Combine(ModuleDirectory, "Public"),
            System.IO.Path.Combine(ModuleDirectory, "Public", "Core"),
            System.IO.Path.Combine(ModuleDirectory, "Public", "Integration"),
            System.IO.Path.Combine(ModuleDirectory, "Public", "Determinism")
        });

        PrivateIncludePaths.AddRange(new string[]
        {
            System.IO.Path.Combine(ModuleDirectory, "Private"),
            System.IO.Path.Combine(ModuleDirectory, "Private", "Core"),
            System.IO.Path.Combine(ModuleDirectory, "Private", "Integration"),
            System.IO.Path.Combine(ModuleDirectory, "Private", "Determinism")
        });

        // Determinism settings - critical for replay-safe AI
        // Disable SIMD optimizations that may vary between platforms
        if (Target.Configuration == UnrealTargetConfiguration.Shipping)
        {
            // Use strict floating-point in shipping builds for determinism
            PublicDefinitions.Add("QRATUM_STRICT_DETERMINISM=1");
        }
    }
}
