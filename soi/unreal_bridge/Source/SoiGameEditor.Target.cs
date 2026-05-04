// SoiGameEditor.Target.cs — Editor target.

using UnrealBuildTool;
using System.Collections.Generic;

public class SoiGameEditorTarget : TargetRules
{
    public SoiGameEditorTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Editor;
        DefaultBuildSettings = BuildSettingsVersion.V5;
        IncludeOrderVersion  = EngineIncludeOrderVersion.Latest;
        ExtraModuleNames.AddRange(new string[] { "SoiGame", "QRATUM" });
    }
}
