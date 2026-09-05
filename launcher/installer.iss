; Inno Setup script for the WanGP Windows installer.
;
; Compile with (values are supplied by the CI workflow):
;   iscc /DMyAppVersion=1.0.0 /DPayloadDir=..\build\payload ^
;        /DLauncherDir=..\dist\launcher-bin /DOutputDir=..\dist installer.iss

#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif
#ifndef PayloadDir
  #define PayloadDir "..\build\payload"
#endif
#ifndef LauncherDir
  #define LauncherDir "..\dist\launcher-bin"
#endif
#ifndef OutputDir
  #define OutputDir "..\dist"
#endif

#define MyAppName "WanGP"
#define MyAppPublisher "Blencia"
#define MyAppURL "https://github.com/blencia/Wan2Gplus"
#define MyAppExeName "WanGP.exe"

[Setup]
; Keep this GUID stable: it is how Windows recognises upgrades of this app.
AppId={{7C5E1A54-9B2D-4F0A-9E33-2D6F1B4C8A71}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
VersionInfoVersion={#MyAppVersion}

; Per-user install: no UAC prompt, and the app can write into its own folder,
; which matters because WanGP downloads models next to itself.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
DisableDirPage=no
AllowNoIcons=yes

OutputDir={#OutputDir}
OutputBaseFilename=WanGP-Setup-{#MyAppVersion}
SetupIconFile=assets\wangp.ico
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\launcher-bin\{#MyAppExeName}
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The launcher bundle.
Source: "{#LauncherDir}\*"; DestDir: "{app}\launcher-bin"; Flags: ignoreversion recursesubdirs createallsubdirs
; The WanGP source tree (CI strips .git, caches, environments and outputs).
Source: "{#PayloadDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\launcher-bin\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\launcher-bin\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\launcher-bin\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Python bytecode caches are generated after install, so Inno does not track them.
Type: filesandordirs; Name: "{app}\__pycache__"

[Messages]
english.WelcomeLabel2=This will install [name/ver] on your computer.%n%nWanGP downloads its Python environment and its AI models on first launch. Plan for at least 40 GB of free space on the drive you install to, and pick a drive with room to spare.
french.WelcomeLabel2=Ceci va installer [name/ver] sur votre ordinateur.%n%nWanGP télécharge son environnement Python et ses modèles d'IA au premier lancement. Prévoyez au moins 40 Go d'espace libre sur le disque choisi.

[Code]
var
  DownloadPage: TDownloadWizardPage;
  NeedsWebView2: Boolean;

function WebView2Installed(): Boolean;
var
  Version: String;
begin
  { The Evergreen runtime registers its version under the WebView2 client GUID. }
  Result :=
    RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) or
    RegQueryStringValue(HKLM, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) or
    RegQueryStringValue(HKCU, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version);
  if Result then
    Result := (Version <> '') and (Version <> '0.0.0.0');
end;

function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  Result := True;
end;

procedure InitializeWizard();
begin
  DownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing), SetupMessage(msgPreparingDesc), @OnDownloadProgress);
  NeedsWebView2 := not WebView2Installed();
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  ResultCode: Integer;
begin
  Result := True;
  if (CurPageID = wpReady) and NeedsWebView2 then
  begin
    { WanGP's window is a WebView2 control; Windows 10 may not ship one. }
    DownloadPage.Clear;
    DownloadPage.Add('https://go.microsoft.com/fwlink/p/?LinkId=2124703', 'MicrosoftEdgeWebview2Setup.exe', '');
    DownloadPage.Show;
    try
      try
        DownloadPage.Download;
        if not Exec(ExpandConstant('{tmp}\MicrosoftEdgeWebview2Setup.exe'), '/silent /install', '',
                    SW_SHOW, ewWaitUntilTerminated, ResultCode) or (ResultCode <> 0) then
          MsgBox('The Microsoft Edge WebView2 Runtime could not be installed automatically.' + #13#10 +
                 'WanGP will install anyway, but its window may fail to open until you install WebView2 manually.',
                 mbInformation, MB_OK);
      except
        MsgBox('The Microsoft Edge WebView2 Runtime could not be downloaded.' + #13#10 +
               'Setup will continue; install WebView2 manually if the WanGP window does not open.',
               mbInformation, MB_OK);
      end;
    finally
      DownloadPage.Hide;
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  AppDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    AppDir := ExpandConstant('{app}');
    { The Python environment, model weights and generated media are created
      after installation, so Inno never tracks them. Removing tens of GB
      silently would be hostile: ask, and default to keeping them. }
    if DirExists(AppDir) then
    begin
      if MsgBox('Also delete the Python environment, the downloaded models and everything else in:' + #13#10 +
                AppDir + #13#10#13#10 +
                'This can free tens of GB and includes your generated videos and images.' + #13#10 +
                'Choose No to keep that folder.',
                mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(AppDir, True, True, True);
    end;
  end;
end;
