// QRATUM.Build.cs
// QRATUM Swarm — Φ=2 module: deterministic kernels + Bridge subsystem.
// Owner: A-01 (Swarm Supervisor).

using UnrealBuildTool;

public class QRATUM : ModuleRules
{
    public QRATUM(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        CppStandard = CppStandardVersion.Cpp20;

        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "InputCore",
            "UMG",
            "CommonUI",
            "Niagara",
            "Json",
            "JsonUtilities",
        });

        PrivateDependencyModuleNames.AddRange(new string[]
        {
            "Slate",
            "SlateCore",
            "RenderCore",
            "RHI",
            "Projects",
            "Sockets",
            "Networking",
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
