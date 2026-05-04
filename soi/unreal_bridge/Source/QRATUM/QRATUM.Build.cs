// QRATUM.Build.cs
// QRATUM Swarm — Φ=1 skeleton module.
// Owner: A-01 (Swarm Supervisor). Behavior added in Φ=2; this module is shells only.

using UnrealBuildTool;

public class QRATUM : ModuleRules
{
    public QRATUM(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "InputCore",
            "UMG",
            "CommonUI",
            "Niagara",
        });

        PrivateDependencyModuleNames.AddRange(new string[]
        {
            "Slate",
            "SlateCore",
            "RenderCore",
            "RHI",
        });

        PublicIncludePaths.AddRange(new string[]
        {
            System.IO.Path.Combine(ModuleDirectory, "Public"),
        });

        PrivateIncludePaths.AddRange(new string[]
        {
            System.IO.Path.Combine(ModuleDirectory, "Private"),
        });
    }
}
