using UnrealBuildTool;

public class QRATUMActions : ModuleRules
{
	public QRATUMActions(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"CoreUObject",
				"Engine",
				"AIModule",
				"NavigationSystem",
				"QRATUMCore"
			}
		);
	}
}
