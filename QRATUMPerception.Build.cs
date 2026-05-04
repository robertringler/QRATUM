using UnrealBuildTool;

public class QRATUMPerception : ModuleRules
{
	public QRATUMPerception(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"CoreUObject",
				"Engine",
				"AIModule",
				"QRATUMCore"
			}
		);
	}
}
