using UnrealBuildTool;

public class QRATUMMemory : ModuleRules
{
	public QRATUMMemory(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"CoreUObject",
				"Engine",
				"QRATUMCore"
			}
		);
	}
}
