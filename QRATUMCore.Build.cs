using UnrealBuildTool;

public class QRATUMCore : ModuleRules
{
	public QRATUMCore(ReadOnlyTargetRules Target) : base(Target)
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
				"GameplayTasks"
			}
		);

		// No dependencies on sibling QRATUM modules.
		// Dependency graph is strictly one-directional:
		// QRATUMPerception / QRATUMMemory / QRATUMActions -> QRATUMCore
	}
}
