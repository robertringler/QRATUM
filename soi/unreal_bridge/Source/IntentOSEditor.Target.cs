// IntentOSEditor.Target.cs — Editor target.

using UnrealBuildTool;
using System.Collections.Generic;

public class IntentOSEditorTarget : TargetRules
{
    public IntentOSEditorTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Editor;
        DefaultBuildSettings = BuildSettingsVersion.V5;
        IncludeOrderVersion  = EngineIncludeOrderVersion.Latest;
        ExtraModuleNames.AddRange(new string[] { "IntentOS", "QRATUM" });
    }
}
