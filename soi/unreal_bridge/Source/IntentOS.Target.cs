// IntentOS.Target.cs — Game (standalone) target.

using UnrealBuildTool;
using System.Collections.Generic;

public class IntentOSTarget : TargetRules
{
    public IntentOSTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Game;
        DefaultBuildSettings = BuildSettingsVersion.V5;
        IncludeOrderVersion  = EngineIncludeOrderVersion.Latest;
        ExtraModuleNames.AddRange(new string[] { "IntentOS", "QRATUM" });
    }
}
