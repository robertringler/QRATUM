// IntentOS.Build.cs
// QRATUM Sovereign Operations Interface — UE5 module.
// Rust telemetry core linkage is GATED on the presence of the prebuilt
// shared library on disk. If absent, the module compiles with
// SOI_RUST_AVAILABLE=0 and telemetry calls become no-ops. Build the Rust
// crate (soi/rust_core/soi_telemetry_core) and re-run UBT to enable.

using System.IO;
using UnrealBuildTool;

public class IntentOS : ModuleRules
{
    public IntentOS(ReadOnlyTargetRules Target) : base(Target)
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

        // Path to Rust dynamic library (built via `cargo build --release`).
        string RustLibPath = Path.Combine(ModuleDirectory, "..", "..", "..", "rust_core", "soi_telemetry_core", "target", "release");

        bool bRustAvailable = false;

        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            string ImportLib = Path.Combine(RustLibPath, "soi_telemetry_core.dll.lib");
            string Dll       = Path.Combine(RustLibPath, "soi_telemetry_core.dll");
            if (File.Exists(ImportLib) && File.Exists(Dll))
            {
                PublicAdditionalLibraries.Add(ImportLib);
                RuntimeDependencies.Add(Dll);
                bRustAvailable = true;
            }
        }
        else if (Target.Platform == UnrealTargetPlatform.Linux)
        {
            string So = Path.Combine(RustLibPath, "libsoi_telemetry_core.so");
            if (File.Exists(So))
            {
                PublicAdditionalLibraries.Add(So);
                RuntimeDependencies.Add(So);
                bRustAvailable = true;
            }
        }
        else if (Target.Platform == UnrealTargetPlatform.Mac)
        {
            string Dylib = Path.Combine(RustLibPath, "libsoi_telemetry_core.dylib");
            if (File.Exists(Dylib))
            {
                PublicAdditionalLibraries.Add(Dylib);
                RuntimeDependencies.Add(Dylib);
                bRustAvailable = true;
            }
        }

        PublicDefinitions.Add("SOI_RUST_AVAILABLE=" + (bRustAvailable ? "1" : "0"));
    }
}
