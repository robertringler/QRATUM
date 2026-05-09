[Setup]
AppName=IntentOS
AppVersion=1.0
DefaultDirName={pf}\IntentOS
DefaultGroupName=IntentOS
OutputDir=.
OutputBaseFilename=IntentOS_Installer
Compression=lzma
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "autostart"; Description: "Auto-start IntentOS on boot"; GroupDescription: "Auto-start"; Flags: checkedonce

[Files]
Source: "PackagedGame\Windows\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\IntentOS"; Filename: "{app}\IntentOS.exe"
Name: "{commondesktop}\IntentOS"; Filename: "{app}\IntentOS.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\IntentOS.exe"; Description: "{cm:LaunchProgram,IntentOS}"; Flags: nowait postinstall skipifsilent

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if WizardIsTaskSelected('autostart') then
    begin
      // Run the install_win.ps1 script
      Exec('powershell.exe', '-ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\install_win.ps1') + '"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    end;
  end;
end;