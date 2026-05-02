using UnrealBuildTool;

public class SoiGame : ModuleRules
{
    public SoiGame(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] 
        { 
            "Core", 
            "CoreUObject", 
            "Engine", 
            "InputCore",
            "CommonUI",
            "UMG",
            "Niagara"
        });

        PrivateDependencyModuleNames.AddRange(new string[] { });

        // Link to Rust dynamic library
        string RustLibPath = System.IO.Path.Combine(ModuleDirectory, "..", "..", "..", "rust_core", "soi_telemetry_core", "target", "release");
        
        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            PublicAdditionalLibraries.Add(System.IO.Path.Combine(RustLibPath, "soi_telemetry_core.dll.lib"));
            RuntimeDependencies.Add(System.IO.Path.Combine(RustLibPath, "soi_telemetry_core.dll"));
        }
        else if (Target.Platform == UnrealTargetPlatform.Linux)
        {
            PublicAdditionalLibraries.Add(System.IO.Path.Combine(RustLibPath, "libsoi_telemetry_core.so"));
            RuntimeDependencies.Add(System.IO.Path.Combine(RustLibPath, "libsoi_telemetry_core.so"));
        }
        else if (Target.Platform == UnrealTargetPlatform.Mac)
        {
            PublicAdditionalLibraries.Add(System.IO.Path.Combine(RustLibPath, "libsoi_telemetry_core.dylib"));
            RuntimeDependencies.Add(System.IO.Path.Combine(RustLibPath, "libsoi_telemetry_core.dylib"));
        }
    }
}
